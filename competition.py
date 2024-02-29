import os
import re
from collections import Counter
from torch.optim import Adam, AdamW
import numpy as np
import torch.nn as nn
from gensim import downloader
from gensim.models import KeyedVectors
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
from torch.optim.lr_scheduler import StepLR, CyclicLR, OneCycleLR
from random import shuffle

EPOCHS = 10
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 16
DROPOUT_RATE = 0.5
GAMMA = 0.1
STEP_SIZE = 5

def under_sample_data(sentences, labels):
    # Combine sentences and labels into a single list of tuples for easy shuffling
    sentence_label_pairs = list(zip(sentences, labels))
    shuffle(sentence_label_pairs)  # Shuffle to randomize the order

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
        if sentence_labels.count(min_class) > 0:
            filtered_sentences.append(sentence)
            filtered_labels.append(sentence_labels)
            counts[min_class] += sentence_labels.count(min_class)
            counts[max_class] += sentence_labels.count(max_class)

    return filtered_sentences, filtered_labels


class NERDataset(Dataset):
    def __init__(self, file_path, models, under_sample=False):
        self.sentences, self.labels = read_data(file_path)
        self.balanced_sentences, self.balanced_labels = None, None
        if under_sample:
            self.sentences, self.labels = under_sample_data(self.sentences, self.labels)

        self.models = models
        self.embedding_dimension = sum([model.vector_size for model in models])
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
                    # model1, model2 = self.models[0], self.models[1]
                    # word_vec_1 = self.get_word_vector(model=model1, word=word, vector_size=model1.vector_size)
                    # word_vec_2 = self.get_word_vector(model=model2, word=word, vector_size=model2.vector_size)
                    # embedded_word = torch.cat((word_vec_1, word_vec_2), dim=0)
                    embedded_word = torch.cat(tuple(embeddings), dim=0)
                    word_embedding_history[word] = embedded_word
                    embedded_sentence.append(embedded_word)
                sentence_labels.append(label)

            self.embedded_sentences.append(torch.stack(embedded_sentence))
            mapped_labels.append(torch.LongTensor(sentence_labels))
        self.mapped_labels = mapped_labels

    def get_word_vector(self, model, word, vector_size):
        """Get the word vector from the model or return a zero vector if the word is OOV."""
        if word in model:
            return torch.tensor(model[word], dtype=torch.float)
        else:
            return torch.as_tensor(np.random.randn(vector_size), dtype=torch.float)

    def get_embedding_dimension(self):
        return self.embedding_dimension

    def __len__(self):
        # Return the number of sentences in the dataset
        return len(self.sentences)

    def __getitem__(self, idx):
        # Return the processed sentence and its corresponding labels at the given index
        return self.embedded_sentences[idx], self.mapped_labels[idx]


def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    sentences_list = []
    labels_list = []
    sentence = []
    labels = []

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

    if sentence and labels:  # Add last sentence if file does not end with a newline
        sentences_list.append(sentence)
        labels_list.append(labels)

    return sentences_list, labels_list


class NERLSTM(nn.Module):
    def __init__(self, vec_dim, num_classes, weights=None, hidden_dim=64, dropout_rate=DROPOUT_RATE, num_layers=2):
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
        # Dropout layer applied to the outputs of the LSTM
        self.dropout = nn.Dropout(dropout_rate)
        # Fully connected layer that maps LSTM outputs to class scores
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)  # *2 because of bidirectional
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim * 2)  # *2 because of bidirectional
        self.fc3 = nn.Linear(hidden_dim * 2, num_classes)  # *2 because of bidirectional
        self.activation = nn.Sigmoid()
        self.loss = nn.CrossEntropyLoss(weight=weights)

        # CRF layer
        self.crf = CRF(num_tags=num_classes, batch_first=True)

    # def forward(self, input_ids, labels=None):
    #     x, (hidden, cell) = self.lstm(input_ids) # transform 2d input_ids to 3d input_ids
    #     x = self.dropout(x)
    #     x = self.fc1(x)
    #     x = self.activation(x)
    #     x = self.fc2(x)
    #     x = x.squeeze(1)
    #     if labels is not None:
    #         # loss = self.loss(x.view(-1, num_classes), labels.view(-1))
    #         loss = self.loss(x, labels)
    #         return x, loss
    #     return x, None

    def forward(self, input_ids, labels=None):
        x, (hidden, cell) = self.lstm(input_ids.unsqueeze(1))
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.activation(x)
        emissions = self.fc2(x)
        labels = labels.unsqueeze(1)
        if labels is not None:
            # Compute the log-likelihood of the labels given the emissions using the CRF layer
            loss = -self.crf(emissions, labels)  # CRF returns log-likelihood
            return emissions, loss
        else:
            # Decode the best path, given the emissions using the CRF layer
            decoded_sequence = self.crf.decode(emissions)
            return decoded_sequence, None


def collate_batch(batch):
    sentences, labels = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)  # Adjust padding_value as needed
    return sentences_padded, labels_padded


def train_and_dev(model, data_sets, optimizer, scheduler, num_epochs: int, batch_size=16, plot=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # data_loaders = {"train": DataLoader(data_sets["train"], batch_size=BATCH_SIZE, collate_fn=collate_batch,
    # shuffle=True), "dev": DataLoader(data_sets["dev"], batch_size=BATCH_SIZE, collate_fn=collate_batch,
    # shuffle=False)}
    best_f1 = 0.0
    scores = {'train': [], 'dev': []}
    losses = {'train': [], 'dev': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'dev']:
            data = data_sets[phase]
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            labels, preds = [], []

            for sentence, sentence_labels in zip(data.embedded_sentences, data.mapped_labels):
                if phase == 'train':
                    outputs, loss = model(sentence, sentence_labels)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    with torch.no_grad():
                        outputs, loss = model(sentence, sentence_labels)

                pred = outputs.argmax(dim=-1).clone().detach().cpu()
                labels += sentence_labels.cpu().view(-1).tolist()
                preds += pred.view(-1).tolist()
                running_loss += loss.item()

            if phase == 'train':
                scheduler.step()  # Update scheduler after each epoch

            epoch_loss = running_loss / len(data_sets[phase])
            epoch_f1 = f1_score(labels, preds)

            epoch_f1 = round(epoch_f1, 5)
            scores[phase].append(epoch_f1)
            losses[phase].append(epoch_f1)

            if phase.title() == "dev":
                print(f'{phase.title()} Loss: {epoch_loss:.4} f1: {epoch_f1}')
            else:
                print(f'{phase.title()} Loss: {epoch_loss:.4} f1: {epoch_f1}')

            if phase == 'dev' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                with open('model.pkl', 'wb') as f:
                    torch.save(model, f)
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
    return weights_tensor


def main():
    avg_f1 = 0
    f1_list = []
    if os.path.exists('word2vec-google-news-300.model'):
        model_word2vec = KeyedVectors.load('word2vec-google-news-300.model')
    else:
        model_word2vec = downloader.load('word2vec-google-news-300.model')
        model_word2vec.save('word2vec-google-news-300.model')
    for i in range(10):
        print(f'--------------fold {i+1}\10-------------')
        best_glove = [None, 0.0]
        # Load the pre-trained models
        for vec_len in [50]:
            # # print(f'glove-twitter-{vec_len} and word2vec-google-news-300')
            # # print(f'glove-twitter-{vec_len}')
            # print(f'word2vec-google-news-300')
            # # Load GloVe model
            # if os.path.exists(f'glove-twitter-{vec_len}.model'):
            #     model_glove = KeyedVectors.load(f'glove-twitter-{vec_len}.model')
            # else:
            #     model_glove = downloader.load(f'glove-twitter-{vec_len}.model')
            #     model_glove.save(f'glove-twitter-{vec_len}.model')
            #
            # # Load Word2Vec model
            # if os.path.exists('word2vec-google-news-300.model'):
            #     model_word2vec = KeyedVectors.load('word2vec-google-news-300.model')
            # else:
            #     model_word2vec = downloader.load('word2vec-google-news-300.model')
            #     model_word2vec.save('word2vec-google-news-300.model')

            # DataLoader
            train_set = NERDataset(file_path="train.tagged", models=[model_word2vec], under_sample=True)
            dev_set = NERDataset(file_path="dev.tagged", models=[model_word2vec])

            # Calculate classes weights
            weights = calculate_class_weights(train_set)

            # Initialize the model
            ner_nn = NERLSTM(vec_dim=train_set.get_embedding_dimension(), num_classes=2, weights=weights)

            # Optimizer
            optimizer_AdamW = AdamW(ner_nn.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            scheduler_StepLR = StepLR(optimizer_AdamW, step_size=STEP_SIZE, gamma=GAMMA)

            # train and dev the models
            data_sets = {"train": train_set, "dev": dev_set}
            score = train_and_dev(model=ner_nn, data_sets=data_sets, optimizer=optimizer_AdamW,
                                  scheduler=scheduler_StepLR, num_epochs=EPOCHS, plot=False)

            if score >= best_glove[1]:
                best_glove = [f'glove-twitter-{vec_len}', score]

        print(f'The best glove model is {best_glove[0]}: {best_glove[1]}')  # print the best model (f1 wise)
        f1_list.append(best_glove[1])
        avg_f1 += best_glove[1]
    print(f"Avg f1 = {avg_f1 / 10}")

if __name__ == "__main__":
    main()
