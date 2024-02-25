import re
from torch.optim import Adam
import numpy as np
import torch.nn as nn
from gensim import downloader
from gensim.models import KeyedVectors
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence

EPOCHS = 15 # Change if needed
LEARNING_RATE = 0.001 # Change if needed
BATCH_SIZE = 16 # Change if needed


class NERDataset(Dataset):
    def __init__(self, file_path, models):
        self.sentences, self.labels = read_data(file_path)
        self.models = models
        self.embedding_dim = sum([model.vector_size for model in models])
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
                embedded_word = None
                if word in word_embedding_history.keys():
                    embedded_word = word_embedding_history[word]
                    embedded_sentence.append(embedded_word)
                else:
                    model1, model2 = self.models[0], self.models[1]
                    word_vec_1 = self.get_word_vector(model=model1, word=word, vector_size=model1.vector_size)
                    word_vec_2 = self.get_word_vector(model=model2, word=word, vector_size=model2.vector_size)
                    embedded_word = torch.cat((word_vec_1, word_vec_2), dim=0)
                    word_embedding_history[word] = embedded_word
                    embedded_sentence.append(embedded_word)
                sentence_labels.append(0 if label == "O" else 1)
      
            self.embedded_sentences.append(torch.stack(embedded_sentence))
            mapped_labels.append(torch.LongTensor(sentence_labels))
        self.mapped_labels = mapped_labels
    
    def get_word_vector(self, model, word, vector_size):
        """Get the word vector from the model or return a zero vector if the word is OOV."""
        if word in model.key_to_index:
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
            labels.append(label)
    
    if sentence and labels:  # Add last sentence if file does not end with a newline
        sentences_list.append(sentence)
        labels_list.append(labels)
    
    return sentences_list, labels_list

class NERNN(nn.Module):
    def __init__(self, vec_dim, num_classes, hidden_dim=128): # check other parameters for hidden_dim
        super(NERNN, self).__init__()
        self.first_layer = nn.Linear(vec_dim, hidden_dim)
        self.second_layer = nn.Linear(hidden_dim, hidden_dim)
        self.third_layer = nn.Linear(hidden_dim, num_classes)
        self.activation = nn.Tanh()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        x = self.first_layer(input_ids)
        x = self.activation(x)
        x = self.second_layer(x)
        x = self.activation(x)
        x = self.third_layer(x)
        if labels is None:
            return x, None
        loss = self.loss(x, labels)
        return x, loss
    

def collate_batch(batch):
    sentences, labels = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)  # Adjust padding_value as needed
    return sentences_padded, labels_padded


# def train_and_dev(model, data_sets, optimizer, num_epochs: int, batch_size=16):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     data_loaders = {"train": DataLoader(data_sets["train"], batch_size=BATCH_SIZE, collate_fn=collate_batch, shuffle=True),
#                     "dev": DataLoader(data_sets["dev"], batch_size=BATCH_SIZE, collate_fn=collate_batch, shuffle=False)}
#     best_f1 = 0.0

#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch + 1}/{num_epochs}')
#         print('-' * 10)
#         scores = {'train': [], 'dev':[]}
#         losses = {'train': [], 'dev':[]}

#         for phase in ['train', 'dev']:
#             if phase == 'train':
#                 model.train()
#             else:
#                 model.eval()

#             running_loss = 0.0
#             labels, preds = [], []

#             for batch in data_loaders[phase]:
#                 batch_size = 0
#                 inputs, labels = batch  # Assuming batch is now a tuple of (padded_data, padded_labels)
#                 inputs, labels = inputs.to(device), labels.to(device)

#                 optimizer.zero_grad()
#                 if phase == 'train':
#                     outputs, loss = model(inputs, labels)
#                     loss.backward()
#                     optimizer.step()
#                 else:
#                     with torch.no_grad():
#                         outputs, loss = model(inputs, labels)

#                 pred = outputs.argmax(dim=-1).clone().detach().cpu()
#                 labels += batch['labels'].cpu().view(-1).tolist()
#                 preds += pred.view(-1).tolist()
#                 running_loss += loss.item() * batch_size

#             epoch_loss = running_loss / len(data_sets[phase])
#             epoch_f1 = f1_score(labels, preds)

#             epoch_f1 = round(epoch_f1, 5)
#             scores[phase].append(epoch_f1)
#             losses[phase].append(epoch_f1)

#             if phase.title() == "dev":
#                 print(f'{phase.title()} Loss: {epoch_loss:.4e} Accuracy: {epoch_f1}')
#             else:
#                 print(f'{phase.title()} Loss: {epoch_loss:.4e} Accuracy: {epoch_f1}')

#             if phase == 'dev' and epoch_f1 > best_f1:
#                 best_f1 = epoch_f1
#                 with open('model.pkl', 'wb') as f:
#                     torch.save(model, f)
#         print()

#     print(f'Best Validation f1-score: {best_f1:4f}')
#     plot_results(scores,losses)

def train_and_dev(model, data_sets, optimizer, num_epochs: int, batch_size=16, plot=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # data_loaders = {"train": DataLoader(data_sets["train"], batch_size=BATCH_SIZE, collate_fn=collate_batch, shuffle=True),
    #                 "dev": DataLoader(data_sets["dev"], batch_size=BATCH_SIZE, collate_fn=collate_batch, shuffle=False)}
    best_f1 = 0.0
    scores = {'train': [], 'dev':[]}
    losses = {'train': [], 'dev':[]}

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
                running_loss += loss.item() * batch_size

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

def plot_results(scores, losses):
    for phase in ['train', 'dev']:
        for result, result_name in zip([scores,losses], ['f1-score', 'loss']):
            plt.plot(result[phase], label=result_name)
            plt.xlim([1, EPOCHS])
            plt.title(f'{phase} {result_name} w.r.t epochs')
            plt.xlabel('epochs')
            plt.ylabel(f'{result_name}')
            plt.show()



def main():
    # # Load the pre-trained models
    # model_word2vec = downloader.load('glove-twitter-50')
    # model_glove = downloader.load('word2vec-google-news-300')
    # # Save the models to disk
    # model_word2vec.save('glove-twitter-50.model')
    # model_glove.save('word2vec-google-news-300.model')
    # Load the models directly from disk
    model_word2vec = KeyedVectors.load('glove-twitter-50.model')
    model_glove = KeyedVectors.load('word2vec-google-news-300.model')
    embedded_vector_dimension = model_word2vec.vector_size + model_glove.vector_size
    # DataLoader
    train_set = NERDataset(file_path="train.tagged", models = [model_word2vec, model_glove])
    dev_set = NERDataset(file_path="dev.tagged", models = [model_word2vec, model_glove])
    
    # Initialize the model
    ner_nn = NERNN(vec_dim = embedded_vector_dimension, num_classes=2)
    
    # Optimizer
    optimizer = Adam(ner_nn.parameters(), lr=LEARNING_RATE) 

    # train and dev the models
    data_sets = {"train": train_set,"dev": dev_set}
    train_and_dev(model=ner_nn, data_sets=data_sets, optimizer=optimizer, num_epochs=EPOCHS, plot=False) 


if __name__ == "__main__":
    main()

