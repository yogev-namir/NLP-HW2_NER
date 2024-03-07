import os
import pickle
import random
import re
import torch
from collections import Counter
from torch.optim import Adam, AdamW
import numpy as np
import torch.nn as nn
from gensim import downloader
from gensim.models import KeyedVectors
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import StepLR, CyclicLR, OneCycleLR

EPOCHS = 15
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.25
GAMMA = 0.1
STEP_SIZE = 5
THRESHOLD = 0.4

PATTERNS = {
    'isSignes': re.compile(r'[!@#$%^&*()_+={}\[\];:\'",<.>/?\\|`~\-]'),
    "isInitCapitalWord": re.compile(r'^[A-Z][a-z]+'),
    "isAllCapitalWord": re.compile(r'^[A-Z]+$'),  # Simplified to match any all-caps word
    "isAllSmallCase": re.compile(r'^[a-z]+$'),
    "isWord": re.compile(r'^[a-zA-Z]+$'),  # Corrected to match any alphabetical word
    "isAlphaNumeric": re.compile(r'^[a-zA-Z0-9]+$'),  # Corrected pattern
    'isAlphabetic': re.compile(r'^[a-zA-Z]+$'),
    "isSingleCapLetter": re.compile(r'^[A-Z]$'),
    "containsDashes": re.compile(r'.*--.*'),
    "containsDash": re.compile(r'.*\-.*'),
    "singlePunctuation": re.compile(r'^\W$'),  # Assuming single non-alphanumeric character
    "repeatedPunctuation": re.compile(r'^[\.\,!\?"\':;_\-]{2,}$'),
    "singleDot": re.compile(r'[.]'),
    "singleComma": re.compile(r'[,]'),
    "singleQuote": re.compile(r'[\']'),
    "isSpecialCharacter": re.compile(r'^[#;:\-/<>\'\"()&]$'),
    "fourDigits": re.compile(r'^\d{4}$'),  # Simplified with quantifier
    "isDigits": re.compile(r'^\d+$'),
    "isNumber": re.compile(r'^\d+(,\d{3})*(\.\d+)?$'),  # Adjusted for typical number formats
    "containsDigit": re.compile(r'.*\d.*'),  # Simplified
    "endsWithDot": re.compile(r'.+\.$'),  # Simplified
    "isURL": re.compile(r'^https?://'),  # Simplified protocol part
    "isMention": re.compile(r'^(RT)?@[\w]+$'),  # Using \w for alphanumeric and underscore
    "isHashtag": re.compile(r'^#\w+$'),  # Simplified using \w
    "isMoney": re.compile(r'^\$\d+(,\d{3})*(\.\d+)?$'),  # Adjusted to match money format
    "isEmail": re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
    "isPhone": re.compile(
        r'^(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?([2-9]1[02-9]|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?$'),
    "isDate": re.compile(r'^(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})$'),
    "isTime": re.compile(r'^(?:[01]\d|2[0-3]):[0-5]\d(?:\s*[ap]m)?$'),
    "isIPAddress": re.compile(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'),
    "isEmoji": re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+'),
    "isAbbreviation": re.compile(r'^(?:[A-Za-z]\.){2,}$'),
    "isAcronym": re.compile(r'^(?:[A-Z]\.){2,}$'),
    "isNumericWithUnit": re.compile(
        r'^\d+(\.\d+)?\s*(?:cm|mm|m|km|in|ft|yd|mi|g|kg|mg|lb|oz|l|ml|gal|pt|qt|cup|°C|°F|K|°)$'),
    "isVersionNumber": re.compile(r'^(?:\d+\.){1,3}\d+$'),
}


def mute_non_entities(predictions, sentence, non_entity_dictionary):
    predictions = predictions
    sentence = sentence
    non_entity_dictionary = non_entity_dictionary
    for i, word in enumerate(sentence):
        for pattern in non_entity_dictionary.values():
            if pattern[0].search(word) and pattern[1] >= 0.95:
                predictions[i] = 0
                break
        if word == '@' and i != len(sentence) - 1:
            predictions[i + 1] = 0
    return predictions


def regex_tokenizer(word):
    # Produce a One-Hot vector based on the Regex patterns
    one_hot_vector = [0] * len(PATTERNS)

    for idx, (_, pattern_regex) in enumerate(PATTERNS.items()):
        if pattern_regex.search(word):
            one_hot_vector[idx] = 1

    return torch.as_tensor(one_hot_vector, dtype=torch.float)


def under_sample_data(sentences, labels):
    # Combine sentences and labels into a single list of tuples for easy shuffling
    sentence_label_pairs = list(zip(sentences, labels))
    original_amount = len(sentence_label_pairs)

    # Flatten all labels to calculate the distribution
    all_labels = [label for _, sentence_labels in sentence_label_pairs for label in sentence_labels]
    label_counts = Counter(all_labels)

    # Find the minority and majority class and its count
    min_class = min(label_counts, key=label_counts.get)
    max_class = max(label_counts, key=label_counts.get)
    min_class_count = label_counts[min_class]

    # Filter sentences to achieve a balanced dataset
    filtered_sentences = []
    filtered_labels = []

    counts = {label: 0 for label in label_counts}
    for sentence, sentence_labels in sentence_label_pairs:
        # Check if adding this sentence would still keep the counts balanced
        if (sentence_labels.count(min_class) > 0):
            # if (sentence_labels.count(min_class)/len(sentence_labels) >= 0.05):
            filtered_sentences.append(sentence)
            filtered_labels.append(sentence_labels)
            counts[min_class] += sentence_labels.count(min_class)
            counts[max_class] += sentence_labels.count(max_class)

    return filtered_sentences, filtered_labels


def read_data(file_path, is_train):
    non_entity_counter = 0
    non_entity_patterns_histogram = {key: [pattern, 0, 0] for key, pattern in PATTERNS.items()}
    sentences_list = []
    labels_list = []
    sentence = []
    labels = []

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        line = re.sub(r'\ufeff', '', line)
        if line == '':
            if sentence and labels:  # Ensure the sentence is not empty
                sentences_list.append(sentence)
                labels_list.append(labels)
                sentence = []
                labels = []
        else:
            word, label = line.split('\t')
            sentence.append(word)
            labels.append(0 if label == "O" else 1)
            for pattern in non_entity_patterns_histogram.values():
                if pattern[0].search(word):
                    pattern[1] += 1
                    if label == "O":
                        pattern[2] += 1
    if is_train:
        for key, pattern in non_entity_patterns_histogram.items():
            certainty_ratio = pattern[2] / pattern[1] if pattern[1] else 0
            non_entity_patterns_histogram[key] = [pattern[0], certainty_ratio]
            print(f'{key}: {certainty_ratio:.3f}')
        if sentence and labels:  # Add last sentence if file does not end with a newline
            sentences_list.append(sentence)
            labels_list.append(labels)

        return sentences_list, labels_list, non_entity_patterns_histogram
    return sentences_list, labels_list, None

class NERDataset(Dataset):
    def __init__(self, file_path, models, is_train=False):
        self.sentences, self.labels, self.non_entity_patterns_histogram = read_data(file_path, is_train)
        # if under_sample:
        #     self.sentences, self.labels = under_sample_data(self.sentences, self.labels)
        self.models = models
        self.embedding_dimension = sum([model.vector_size for model in models])  # +len(PATTERNS)
        self.embedded_sentences = []
        self.mapped_labels = []
        self.tokenize()

    def tokenize(self):
        word_embedding_history = {}
        mapped_labels = []
        for sentence, labels in zip(self.sentences, self.labels):
            embedded_sentence = []
            sentence_labels = []
            for word, label in zip(sentence, labels):
                # word = word.lower() if label == "O" else word
                embedded_word = None
                if word in word_embedding_history.keys():
                    embedded_word = word_embedding_history[word]
                    embedded_sentence.append(embedded_word)
                else:
                    models = self.models
                    embeddings = []

                    for model in models:
                        word_vec = self.get_word_vector(model=model, word=word, vector_size=model.vector_size)
                        embeddings.append(word_vec)
                    regex_one_hot = regex_tokenizer(word)
                    embeddings.append(regex_one_hot)
                    embedded_word = torch.cat(tuple(embeddings), dim=0)
                    word_embedding_history[word] = embedded_word
                    embedded_sentence.append(embedded_word)
                sentence_labels.append(label)

            self.embedded_sentences.append(torch.stack(embedded_sentence))
            mapped_labels.append(torch.LongTensor(sentence_labels))
        self.mapped_labels = mapped_labels

    def get_word_vector(self, model, word, vector_size):
        """Get the word vector from the model or return a zero vector if the word is OOV."""
        if word in model.key_to_index:
            return torch.tensor(model.vectors[model.key_to_index[word]], dtype=torch.float)
        else:
            return torch.as_tensor(np.zeros(vector_size), dtype=torch.float)

    def get_embedding_dimension(self):
        return self.embedding_dimension

    def __len__(self):
        # Return the number of sentences in the dataset
        return len(self.sentences)

    def __getitem__(self, idx):
        # Return the processed sentence and its corresponding labels at the given index
        return self.embedded_sentences[idx], self.mapped_labels[idx]


class NERLSTM(nn.Module):
    def __init__(self, vec_dim, num_classes, weights, pos_weight, hidden_dim=64, dropout_rate=DROPOUT_RATE,
                 num_layers=2):
        super(NERLSTM, self).__init__()
        # Embedding layer if needed (e.g., if input_ids are token indices)
        # self.embedding = nn.Embedding(num_embeddings, vec_dim)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=vec_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=DROPOUT_RATE if num_layers > 1 else 0,
                            # apply dropout between LSTM layers if num_layers > 1
                            bidirectional=True)  # Using a bidirectional LSTM
        self.lstm_word2vec = nn.LSTM(input_size=300,
                                     hidden_size=hidden_dim,
                                     num_layers=num_layers,
                                     batch_first=True,
                                     dropout=DROPOUT_RATE if num_layers > 1 else 0,
                                     # apply dropout between LSTM layers if num_layers > 1
                                     bidirectional=True)  # Using a bidirectional LSTM
        self.lstm_glove = nn.LSTM(input_size=vec_dim - 300,
                                  hidden_size=hidden_dim,
                                  num_layers=num_layers,
                                  batch_first=True,
                                  dropout=DROPOUT_RATE if num_layers > 1 else 0,
                                  # apply dropout between LSTM layers if num_layers > 1
                                  bidirectional=True)  # Using a bidirectional LSTM
        # Dropout layer applied to the outputs of the LSTM
        self.dropout = nn.Dropout(dropout_rate)
        # Apply LayerNorm after LSTM
        # self.layer_norm = nn.LayerNorm(hidden_dim * 4)  # * 4 for dual bi-direction
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)  # * 2 for bi-direction

        # Fully connected layer that maps LSTM outputs to class scores
        # self.fc1 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, num_classes)
        self.fc3 = nn.Linear(hidden_dim * 2, 1)
        self.fc4 = nn.Linear(hidden_dim, num_classes)
        self.fc5 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim * 2, hidden_dim // 2)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc8 = nn.Linear(hidden_dim // 2, 1)

        self.activation = nn.Tanh()
        # self.activation = nn.LeakyReLU()
        self.normalization = nn.Sigmoid()
        # self.loss = nn.CrossEntropyLoss(weight=weights, reduction='mean')
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    # identity
    def forward(self, input_ids, labels=None):
        x, (hidden, cell) = self.lstm(input_ids.unsqueeze(1))
        norm_out = self.layer_norm(x)
        identity = self.activation(self.fc6(norm_out))
        x = self.activation(self.fc5(norm_out))
        x = self.activation(self.fc7(x))
        x = x + identity
        x = self.activation(self.fc8(x))
        x = x.squeeze(1)

        if labels is not None:
            labels = labels.unsqueeze(1).float()
            loss = self.loss(x, labels)
            return x, loss
        return x, None

    # ~0.56
    # def forward(self, input_ids, labels=None):
    #     x, (hidden, cell) = self.lstm(input_ids.unsqueeze(1))
    #     x = self.layer_norm(x)
    #     x = self.dropout(x)
    #     x = self.activation(self.fc1(x))
    #     x = self.activation(self.fc2(x))
    #     x = x.squeeze(1)

    #     if labels is not None:
    #         # labels = labels.unsqueeze(1).float()
    #         loss = self.loss(x, labels)
    #         return x, loss
    #     return x, None


def train_and_dev(model, data_sets, optimizer, scheduler, num_epochs: int, batch_size=16, plot=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    softmax = nn.Softmax(dim=-1)
    sigmoid = nn.Sigmoid()

    best_f1 = 0.0
    scores = {'train': [], 'dev': []}
    losses = {'train': [], 'dev': []}
    non_entity_dictionary = data_sets['train'].non_entity_patterns_histogram

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            data = data_sets[phase]
            running_loss = 0.0
            labels, preds = [], []

            for original_sentence, sentence, sentence_labels in zip(data.sentences, data.embedded_sentences,
                                                                    data.mapped_labels):
                sentence = sentence.to(device)
                sentence_labels = sentence_labels.to(device)
                if phase == 'train':
                    outputs, loss = model(sentence, sentence_labels)
                    predictions = (outputs[:] > THRESHOLD).long()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    labels += sentence_labels.cpu().view(-1).tolist()
                    preds += predictions.view(-1).tolist()
                    running_loss += loss.item()
                else:
                    with torch.no_grad():
                        outputs, loss = model(sentence, sentence_labels)
                        predictions = sigmoid(outputs)
                        predictions = (predictions[:] > THRESHOLD).long()  # binary cross entropy
                        predictions = mute_non_entities(predictions, original_sentence, non_entity_dictionary)
                        labels += sentence_labels.cpu().view(-1).tolist()
                        preds += predictions.view(-1).tolist()
                        running_loss += loss.item()
                # probabilities = sigmoid(outputs)
                # probabilities = softmax(outputs)
                # predictions = probabilities.argmax(dim=-1).clone().detach().cpu()
                # predictions = outputs.argmax(dim=-1).clone().detach().cpu()
                # predictions = non_entity(predictions, original_sentence)
                # predictions = (probabilities[:, 1] > THRESHOLD).long() # cross entropy
                # predictions = (outputs[:] > THRESHOLD).long() # binary cross entropy
                # labels += sentence_labels.cpu().view(-1).tolist()
                # preds += predictions.view(-1).tolist()
                # running_loss += loss.item()

            epoch_loss = running_loss / len(data_sets[phase])
            epoch_f1 = f1_score(labels, preds, average='binary')

            epoch_f1 = round(epoch_f1, 5)
            scores[phase].append(epoch_f1)
            losses[phase].append(epoch_f1)

            if phase == 'train':
                scheduler.step()  # Update scheduler after each epoch

            if phase.title() == "dev":
                print(f'{phase.title()} Loss: {epoch_loss:.4} f1: {epoch_f1}')
            else:
                print(f'{phase.title()} Loss: {epoch_loss:.4} f1: {epoch_f1}')

            if phase == 'dev' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                with open('model.pkl', 'wb') as f:
                    pickle.dump(model, f)
            if phase == 'dev' and epoch_f1 > 0.6:
                with open(f'saved_preds_{epoch_f1}_{epoch}.pkl', 'wb') as f:
                    pickle.dump(preds, f)
        print()

    print(f'Best Development f1-score: {best_f1:4f}')
    if plot:
        plot_results(scores, losses)
    return best_f1


def plot_results(scores, losses):
    for phase in ['train', 'dev']:
        for result, result_name in zip([scores, losses], ['f1-score', 'loss']):
            plt.plot(result[phase], label=result_name)
            plt.xlim([1, EPOCHS])
            plt.title(f'{phase} {result_name} w.r.t epochs')
            plt.xlabel('epochs')
            plt.ylabel(f'{result_name}')
            plt.show()


def calculate_class_weights(dataset):
    label_counts = Counter()
    for _, labels in dataset:
        label_counts.update(labels.flatten().tolist())

    # Calculate weights: the inverse of the frequency, normalized
    total_counts = sum(label_counts.values())
    class_weights = {class_id: total_counts / count for class_id, count in label_counts.items()}
    # Normalize weights so they sum up to the number of classes
    weight_sum = sum(class_weights.values())
    num_classes = len(class_weights)
    class_weights = {class_id: weight * num_classes / weight_sum for class_id, weight in class_weights.items()}

    # Convert class weights to a tensor
    weights_tensor = torch.tensor([class_weights[i] for i in range(len(class_weights))], dtype=torch.float)
    pos_weight = label_counts[0] / label_counts[1]

    return weights_tensor, pos_weight


def main():
    train_path = 'new_train.tagged'
    dev_path = 'new_dev.tagged'

    # Load Word2Vec model
    if os.path.exists('word2vec-google-news-300.model'):
        model_word2vec = KeyedVectors.load('word2vec-google-news-300.model')
    else:
        print('downloading word2vec-google-news-300')
        model_word2vec = downloader.load('word2vec-google-news-300')
        model_word2vec.save('word2vec-google-news-300.model')

    # Load GloVe models
    if os.path.exists(f'glove-twitter-25.model'):
        model_glove_25 = KeyedVectors.load(f'glove-twitter-25.model')
    else:
        print('downloading glove-twitter-25')
        model_glove_25 = downloader.load(f'glove-twitter-25')
        model_glove_25.save(f'glove-twitter-25.model')

    # Load GloVe models
    if os.path.exists(f'glove-twitter-50.model'):
        model_glove_50 = KeyedVectors.load(f'glove-twitter-50.model')
    else:
        print('downloading glove-twitter-25')
        model_glove_50 = downloader.load(f'glove-twitter-50')
        model_glove_50.save(f'glove-twitter-50.model')

    score_list = []
    # Load the pre-trained models
    models = [model_word2vec, model_glove_25]
    for _ in range(10):
        # DataLoader
        train_set = NERDataset(file_path=train_path, models=models, is_train=True)
        dev_set = NERDataset(file_path=dev_path, models=models)

        # Calculate classes weights
        weights, pos_weight = calculate_class_weights(train_set)

        # Initialize the model
        ner_nn = NERLSTM(vec_dim=train_set.get_embedding_dimension() + len(PATTERNS), num_classes=2, weights=weights,
                         pos_weight=pos_weight)

        # Optimizer
        optimizer_AdamW = AdamW(ner_nn.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler_StepLR = StepLR(optimizer_AdamW, step_size=STEP_SIZE, gamma=GAMMA)

        # train and dev the models
        data_sets = {"train": train_set, "dev": dev_set}
        score = train_and_dev(model=ner_nn, data_sets=data_sets, optimizer=optimizer_AdamW,
                              scheduler=scheduler_StepLR, num_epochs=EPOCHS, plot=False)
        score_list.append(score)

    print(score_list)
    print(np.mean(score_list))


if __name__ == "__main__":
    main()