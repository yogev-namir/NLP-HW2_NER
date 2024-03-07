from itertools import chain
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score
import torch.nn as nn
import torch.nn.functional as F
import re
import string
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

PATTERNS = {
    "isInitCapitalWord": re.compile(r'^[A-Z][a-z]+'),
    "isAllCapitalWord": re.compile(r'^[A-Z]+$'),
    "isAllSmallCase": re.compile(r'^[a-z]+$'),
    "isWord": re.compile(r'^[a-zA-Z]+$'),
    "isAlphaNumeric": re.compile(r'^[a-zA-Z0-9]+$'),
    "isSingleCapLetter": re.compile(r'^[A-Z]$'),
    "containsDashes": re.compile(r'.*--.*'),
    "containsDash": re.compile(r'.*-.*'),
    "singlePunctuation": re.compile(r'^[' + re.escape(string.punctuation) + r']$'),
    "repeatedPunctuation": re.compile(r'^[.,!?":;_\-]{2,}$'),
    "singleDot": re.compile(r'[.]'),
    "singleComma": re.compile(r'[,]'),
    "singleQuote": re.compile(r'[\'\"]'),
    "isSpecialCharacter": re.compile(r'^[#;:/<>\'"()&]$'),
    "fourDigits": re.compile(r'^\d{4}$'),
    "isDigits": re.compile(r'^\d+$'),
    "isNumber": re.compile(r'^\d+(\.\d+)?$'),  # Simplified version
    "containsDigit": re.compile(r'.*\d+.*'),
    "endsWithDot": re.compile(r'.+\.$'),
    "isURL": re.compile(r'^https?://'),
    "isMention": re.compile(r'^(RT)?@[\w_]+$'),
    "isHashtag": re.compile(r'^#\w+$'),
    "isMoney": re.compile(r'^\$\d+(\.\d+)?$'),  # Simplified version
}


def word_to_features(word):
    features = []
    for pattern in PATTERNS.values():
        features.append(1 if pattern.match(word) else 0)
    return features


class NeuralNetworks:
    def __init__(self, hidden_dim, glove_model, tag_to_ix, device, unique_words, second_model, ix_to_tag, val_sen,
                 val_tags, model_type):
        self.val_sen = val_sen
        self.val_tags = val_tags
        self.device = device
        self.embedding_dim = glove_model.vector_size
        self.hidden_dim = hidden_dim
        self.model = None
        self.word_to_ix = {}  # Start with PAD and UNK
        self.embedding_dim = glove_model.vector_size + second_model.vector_size  # New embedding dimension
        self.model_dict = {"FFNN": NeuralNetworks.FFNN_Residual, "LSTM": NeuralNetworks.LSTMLayerNorm,
                           "COMP": NeuralNetworks.LSTMLayerNormResidual}
        self.model_type = model_type
        # Adjust embedding dimension based on model type
        if self.model_type == "COMP":
            self.embedding_dim = glove_model.vector_size + second_model.vector_size + len(
                PATTERNS)  # Include pattern features
        else:
            self.embedding_dim = glove_model.vector_size + second_model.vector_size  # Do not include pattern features

        combined_embeddings_array = []

        for word in unique_words:
            self.word_to_ix[word] = len(self.word_to_ix)

            # Initialize embedding vector for the word with zeros for the total size
            word_embedding = np.zeros(self.embedding_dim, )

            # If the word exists in glove_model, replace the relevant part of the embedding
            if word in glove_model:
                start = 0  # Starting index for GloVe embeddings
                end = glove_model.vector_size  # Ending index for GloVe embeddings
                word_embedding[start:end] = glove_model[word]
            elif word.lower() in glove_model:
                start = 0
                end = glove_model.vector_size
                word_embedding[start:end] = glove_model[word.lower()]

                # If the word exists in second_model, replace the relevant part of the embedding
            if word in second_model:
                start = glove_model.vector_size  # Starting index for second_model embeddings
                end = glove_model.vector_size + second_model.vector_size  # Adjust ending index based on actual size
                word_embedding[start:end] = second_model[word]
            elif word.lower() in second_model:
                start = glove_model.vector_size
                end = glove_model.vector_size + second_model.vector_size
                word_embedding[start:end] = second_model[word.lower()]

            # Generate features based on regex patterns if model type is COMP
            if self.model_type == "COMP":
                start = glove_model.vector_size + second_model.vector_size
                end = self.embedding_dim
                word_features = np.array(word_to_features(word), dtype=float)
                word_embedding[start:end] = word_features

            # Append the combined embedding to the list
            combined_embeddings_array.append(word_embedding)  # Keep it as a NumPy array for efficiency

        # Convert the list of combined embeddings into a tensor
        self.embeddings = torch.tensor(np.array(combined_embeddings_array), dtype=torch.float).to(self.device)
        self.ix_to_tag = ix_to_tag
        self.vocab_size = len(self.word_to_ix)
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(self.tag_to_ix)

    def train(self, train_sentences, train_tags, model_type, lr=0.008, num_epochs=10):
        self.model = self.model_dict[model_type](self.vocab_size, self.embedding_dim, self.hidden_dim,
                                                 self.embeddings).to(self.device)
        tag_counts = {tag: sum(1 for sublist in train_tags for t in sublist if t == tag) for tag in
                      self.tag_to_ix.keys()}
        pos_weight = torch.tensor([tag_counts['O'] / tag_counts['ENT']]).to(self.device)
        loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode='max',
                                      factor=0.5,  # Reduce lr by half when needed
                                      patience=3,
                                      # Number of epochs with no improvement after which learning rate will be reduced
                                      threshold=0.005,  # Threshold for measuring the new optimum
                                      verbose=True)
        max_score = 0
        for epoch in range(num_epochs):
            self.model.train()  # Set the model to training mode
            total_loss = 0
            for sentence, tags in zip(train_sentences, train_tags):
                # Convert sentence to tensor
                sentence_tensor = self.prepare_sentence(sentence)
                sentence_tensor = sentence_tensor.to(self.device)
                self.model.zero_grad()
                tags_tensor = torch.tensor([self.tag_to_ix[tag] for tag in tags], dtype=torch.float).unsqueeze(-1).to(
                    self.device)
                tag_scores = self.model(sentence_tensor.unsqueeze(0))  # Add batch dimension
                loss = loss_function(tag_scores.squeeze(0), tags_tensor)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            score = self.predict(train_sentences, train_tags, verbose=True)
            print(f'Epoch {epoch}: Total Loss = {total_loss:.4f} train f1 = {score:.5f} lr = {lr}')
            score = self.predict(self.val_sen, self.val_tags, verbose=True)
            if score > 0.66:  ##################################################################################
                break  #######################################################################################
            max_score = max(max_score, score)
            print(f'val f1 = {score:.5f}')
            scheduler.step(score)
        print(f'Max score: {max_score:.5f}')

    def predict(self, test_sentences, test_tags=None, verbose=False):
        # Switch model to evaluation mode
        self.model.eval()
        self.model.to(self.device)  # Ensure the model is on the correct device
        all_predictions = []
        with torch.no_grad():  # No need to track gradients during testing
            for sentence in test_sentences:
                sentence_tensor = self.prepare_sentence(sentence)
                sentence_tensor = sentence_tensor.to(self.device)

                tag_scores = self.model(sentence_tensor.unsqueeze(0))  # Add batch dimension
                tag_scores = tag_scores.squeeze()  # Remove the batch dimension and any unnecessary last dimension
                # Convert logits to probabilities
                probabilities = torch.sigmoid(tag_scores)

                # Handling the edge case where predictions result in a single value
                if probabilities.ndim == 0:
                    # If probabilities is just a single number, convert to a 1-element list
                    predictions = [probabilities.item() >= 0.5]
                else:
                    # Usual case: convert probabilities to binary predictions
                    predictions = (probabilities >= 0.5).long()
                    predictions = predictions.cpu().numpy().tolist()
                # If you need the predictions as a Python list (e.g., for evaluation)
                all_predictions.append(predictions)
                # predictions = tag_scores.argmax(dim=2).squeeze(0)  # Remove batch dimension and get predicted tags)
        y_pred = list(chain.from_iterable(all_predictions))
        y_pred = [self.ix_to_tag[ix] for ix in y_pred]
        if verbose and test_tags is not None:
            y_true = list(chain.from_iterable(test_tags))  # flatten y_true to one list
            score = f1_score(y_true, y_pred, pos_label='ENT', average='binary')
            return score
        else:
            return all_predictions

    class FFNN_Residual(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, embeddings, output_dim=1):
            super(NeuralNetworks.FFNN_Residual, self).__init__()
            self.hidden_dim = hidden_dim
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.word_embeddings.weight.data.copy_(embeddings)
            self.fc1 = nn.Linear(embedding_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Additional layer for forming a residual block
            self.fc3 = nn.Linear(hidden_dim, output_dim)
            self.activation = nn.Tanh()  # Apply ReLU activation function
            self.shortcut = nn.Linear(embedding_dim, hidden_dim)  # Shortcut connection

        def forward(self, x):
            identity = self.activation(self.shortcut(x))  # Shortcut connection
            out = self.activation(self.fc1(x))
            out = out + identity  # Add input directly before the second activation
            out = self.activation(self.fc2(out))
            out = self.fc3(out)
            return out

    class LSTMLayerNorm(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, embeddings, output_dim=1):
            super(NeuralNetworks.LSTMLayerNorm, self).__init__()
            self.hidden_dim = hidden_dim
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.word_embeddings.weight.data.copy_(embeddings)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
            self.layer_norm = nn.LayerNorm(hidden_dim * 2)  # Normalization layer for LSTM output
            # Fully connected layers
            self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)  # Adjust input dimension for bidirectionality
            self.fc2 = nn.Linear(hidden_dim * 2, output_dim)  # Maps to tagset_size
            self.dropout = nn.Dropout(0.5)  # Dropout layer
            self.activation = nn.Tanh()  # Apply ReLU activation function

        def forward(self, sentence):
            # embeds = self.word_embeddings(sentence)
            lstm_out, _ = self.lstm(sentence)
            norm_out = self.layer_norm(lstm_out)  # Apply layer normalization
            # Applying fully connected layers
            fc1_out = self.activation(self.fc1(norm_out))  # Apply ReLU activation function
            dropout = self.dropout(fc1_out)
            tag_space = self.fc2(dropout)  # Output from second fully connected layer
            return tag_space

    class LSTMLayerNormResidual(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, embeddings):
            super(NeuralNetworks.LSTMLayerNormResidual, self).__init__()
            self.hidden_dim = hidden_dim
            # Initialize embeddings and LSTM components...
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.word_embeddings.weight.data.copy_(embeddings)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
            self.layer_norm = nn.LayerNorm(hidden_dim * 2)
            # Initialize reduced dimensional residual FFNN components...
            self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # First reduction: hidden_dim * 2 to hidden_dim
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)  # Second reduction: hidden_dim to hidden_dim / 2
            self.fc3 = nn.Linear(hidden_dim // 2, 1)  # Final layer maps to output dimension
            self.shortcut = nn.Linear(hidden_dim * 2,
                                      hidden_dim // 2)  # Shortcut should match the last layer's input dimension
            self.activation = nn.Tanh()

        # 0.56-0.61
        def forward(self, sentence):
            # Make sure 'sentence' is of long type for embedding layer
            lstm_out, _ = self.lstm(sentence)
            norm_out = self.layer_norm(lstm_out)
            # Process through reduced dimensional residual FFNN
            identity = self.activation(self.shortcut(norm_out))  # Adjusted shortcut to match final FC layer's input
            out = self.activation(self.fc1(norm_out))
            out = self.activation(self.fc2(out))
            out = out + identity  # Add input directly before the final activation
            tag_space = self.fc3(out)
            return tag_space

    def prepare_sentence(self, sentence):
        """
        Converts a sentence into a tensor of embeddings, where each word in the sentence
        is represented by its embedding if present in glove_model or second_model,
        or a random vector otherwise.

        Args:
            sentence (list of str): The sentence to convert, represented as a list of words.

        Returns:
            torch.Tensor: The tensor representation of the sentence.
        """
        sentence_embeddings = []
        for word in sentence:
            # Check if the word is in the vocabulary; if not, use the '<UNK>' token.
            word_index = self.word_to_ix.get(word)

            # Fetch the embedding vector for the word.
            word_embedding = self.embeddings[word_index]

            # Append the embedding vector to the list.
            sentence_embeddings.append(word_embedding)

        # Convert the list of embedding vectors into a 2D tensor.
        sentence_tensor = torch.stack(sentence_embeddings)

        return sentence_tensor

    def convert_to_binary(c_list, ix_to_tag):
        """
        Converts a list of categorical indices into a binary list.

        Args:
        - c_list: List of categorical indices (e.g., [1, 2, 0, 1]).
        - ix_to_tag: Mapping from indices to tags (e.g., {0: 'O', 1: 'B-LOC', 2: 'I-PER'}).

        Returns:
        - A list where each element is 'O' if the corresponding tag in ix_to_tag is 'O',
        otherwise 'ENT'.
        """
        binary_list = ['O' if ix_to_tag[ix] == 'O' else 'ENT' for ix in c_list]
        return binary_list