# models.py

from functools import reduce
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
# was included
import collections

#####################
# MODELS FOR PART 1 #
#####################

class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")

class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
    """
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1


class RNNClassifier(nn.Module):
    def __init__(self, vocab_index, dict_size=27, input_size=50, hidden_size=30, class_size=2):
        super(RNNClassifier, self).__init__()
        self.word_embedding = nn.Embedding(dict_size, input_size)
        self.vocab = vocab_index
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.init_weight()
        self.hidden2tag = nn.Linear(hidden_size, class_size)
        self.soft = nn.LogSoftmax(0)

    def init_weight(self):
        nn.init.xavier_normal_(self.rnn.weight_hh_l0).type('torch.FloatTensor')
        nn.init.xavier_normal_(self.rnn.weight_ih_l0).type('torch.FloatTensor')
        # nn.init.xavier_normal_(self.rnn.bias_hh_l0)
        # nn.init.xavier_normal_(self.rnn.bias_ih_l0)

    def forward(self, input):
        embedded_input = self.word_embedding(input)
        # Note: the hidden state and cell state are 1x1xdim tensor: num layers * num directions x batch_size x dimensionality
        init_state = (torch.from_numpy(np.zeros((2, len(input), self.hidden_size))).type('torch.FloatTensor'),
                      torch.from_numpy(np.zeros((2, len(input), self.hidden_size))).type('torch.FloatTensor'))
        output, (hidden_state, cell_state) = self.rnn(embedded_input, init_state)
        
        # Note: hidden_state is a 1x1xdim tensor: num layers * num directions x batch_size x dimensionality
        return self.hidden2tag(hidden_state[-1])

    def predict(self, context):
        input = preprocess(context, [0], self.vocab)
        y_pred = self.forward(torch.tensor(input).unsqueeze(0))
        y_pred = self.soft(y_pred.squeeze())
        return y_pred.max(0)[1]


def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)

def preprocess(list_of_exs, lables, index):
    data = []
    if lables == [0]:
        for letter in list_of_exs:
            letter_idx = index.index_of(letter) if index.index_of(letter) != -1 else -1
            data.append(letter_idx)  
    else:
        for i in range(len(list_of_exs)):
            for item in list_of_exs[i]:
                letters = []
                for letter in item:
                    letter_idx = index.index_of(letter) if index.index_of(letter) != -1 else -1
                    letters.append(letter_idx)
                data.append((letters, lables[i]))
    return data

def get_batches(data, batch_size):
    count = 0
    batches = []
    count_down = len(data)
    while count_down > batch_size:
        batches.append(data[count:(count + batch_size)])
        count += batch_size
        count_down -= batch_size
    if count_down > 1:
        batches.append(data[count:])
    
    return batches

def get_labels_and_data(batch):
    labels = []
    data = []
    for datem, label in batch:
        labels.append(label)
        data.append(np.array(datem))
    return torch.tensor(data), torch.tensor(labels)

def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """
    
    # Define hyper parmeters and model
    num_epochs = 10 #10 is better to turn in
    initial_learning_rate = 0.001
    batch_size = 32

    # Model specifications
    model = RNNClassifier(vocab_index)
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
    loss_funct = torch.nn.CrossEntropyLoss() # what loss functions should we used NLL is need calcte after softmax but befor loss

    # Preprocess data
    print("Preprocessing the Training data")
    train_data = preprocess([train_cons_exs, train_vowel_exs], [0,1] ,vocab_index)
    dev_data = preprocess([dev_cons_exs, dev_vowel_exs], [0, 1], vocab_index)

    for epoch in range(num_epochs):
        total_loss = 0.0
        accuracys = []

        #Batch the data
        random.shuffle(train_data)
        batches = get_batches(train_data, batch_size)

        for batch in batches:
            batch_data, batch_label = get_labels_and_data(batch)

            optimizer.zero_grad()
            y_pred = model.forward(batch_data)
            
            # calculate loss and accuracy
            loss = loss_funct(y_pred, batch_label)
            total_loss += loss
            for i in range(len(batch)):
                ret = 1 if y_pred[i].max(0)[1] == batch_label[i] else 0
                accuracys.append(ret)
 
            # Computes the gradient and takes the optimizer step
            loss.backward()
            optimizer.step()

        ## Dev Testing
        dev_accuracys = []
        batches = get_batches(dev_data, batch_size)
        for batch in batches:
            batch_data, batch_label = get_labels_and_data(batch)
            y_pred = model.forward(batch_data)
            for i in range(len(batch)):
                ret = 1 if y_pred[i].max(0)[1] == batch_label[i] else 0
                dev_accuracys.append(ret)

        print("Total loss on epoch %i: %f" % (epoch, total_loss))
        print("The traing set accuracy for epoch %i: %f" % (epoch, np.mean(accuracys)))
        print("The dev set accuracy for epoch %i: %f" % (epoch, np.mean(dev_accuracys)))

    return model


#####################
# MODELS FOR PART 2 #
#####################

# predict dirtbition over all possible next characters
# use "run santiy check" -> checks that produces valid probalities 
# need to evalue with perplexity
# normatization test -> makes sure sume to one 
# check that they are probalities that normalize 
# if fail you def have a bug



class LanguageModel(object):

    def get_next_char_log_probs(self, context): # -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e
        :param context: a single character to score
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context): # -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_index, dict_size=27, input_size=50, hidden_size=30, class_size=27, num_layers=2): 
        super(RNNLanguageModel, self).__init__()    
        self.word_embedding = nn.Embedding(dict_size, input_size)
        self.vocab = vocab_index
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.init_weight()
        self.hidden2tag = nn.Linear(hidden_size, class_size)
        self.soft = nn.LogSoftmax(-1)
        self.num_layers = num_layers

    def init_weight(self):
        nn.init.xavier_normal_(self.rnn.weight_hh_l0).type('torch.FloatTensor')
        nn.init.xavier_normal_(self.rnn.weight_ih_l0).type('torch.FloatTensor')
        # nn.init.xavier_normal_(self.rnn.bias_hh_l0).type('torch.FloatTensor')
        # nn.init.xavier_normal_(self.rnn.bias_ih_l0).type('torch.FloatTensor')

    def forward(self, input, batch=False):
        if not batch:
            input = input.unsqueeze(0)
        batch_size = len(input) #if batch else 1

        # print("the input dim ar currently", input.shape)
        embedded_input = self.word_embedding(input)


        # Note: the hidden state and cell state are 1x1xdim tensor: num layers * num directions x batch_size x dimensionality
        init_state = (torch.from_numpy(np.zeros((self.num_layers, batch_size, self.hidden_size))).type('torch.FloatTensor'),
                      torch.from_numpy(np.zeros((self.num_layers, batch_size, self.hidden_size))).type('torch.FloatTensor'))
        # if not batch:
        #     embedded_input = embedded_input.unsqueeze(0)
        # print("the embedd dim ar currently", embedded_input.shape)
        output, (hidden_state, cell_state) = self.rnn(embedded_input, init_state)
        # print("the output dim ar currently", output.shape)
        y = self.hidden2tag(output)
        # print("the classification dim ar currently", y.shape)
        # Note: hidden_state is a 1x1xdim tensor: num layers * num directions x batch_size x dimensionality
        # y = self.soft(y)
        # print("the soft dim ar currently", y.shape)
        return y

    def get_next_char_log_probs(self, context): # insert sapce into input
        context = ' ' + context
        context_vec = self.vectorize(context)
        
        output = self.forward(context_vec)
        output = self.soft(output)
        return output[0][-1].detach().numpy() ## double check this

    def get_log_prob_sequence(self, next_chars, context):
        full_str = " " + context + next_chars
        full_str_vec = self.vectorize(full_str)

        output = self.forward(full_str_vec[:-1])
        output = self.soft(output)
        log_probs = 0
        for i in range(len(next_chars)):
            log_probs += output[0][i + len(context)][full_str_vec[i + len(context)+ 1]]
        return log_probs.item()
   
    def vectorize(self, context):
        vector = []
        for char in context:
            vector.append(self.vocab.index_of(char))
        return torch.tensor(vector)

def lm_preprocess(text, chunk_size, vocab):
    count = 0
    data = []
    indexed_text = []
    for letter in text:
        indexed_text.append(vocab.index_of(letter))
        
    while count + chunk_size + 1 < len(text):
        # word_chunk = text[count:count + chunk_size]
        # word_label = text[count:count + chunk_size + 1]
        chunk = indexed_text[count:count + chunk_size] 
        chunk.insert(0, 26)
        label = indexed_text[count:count + chunk_size + 1]
        # print(chunk, word_chunk)
        # print(label, word_label)
        data.append((chunk, label))
        count += 1

    return data
    


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    # Define hyper parmeters and model
    num_epochs = 8
    initial_learning_rate = 1e-4
    batch_size = 64
    chunk_size = 30

    ## Create Dataset
    train_data = lm_preprocess(train_text, chunk_size, vocab_index)

    ## Model specifications
    model = RNNLanguageModel(vocab_index, dict_size=27, input_size=50, hidden_size=30, class_size=27, num_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
    # loss_funct = torch.nn.NLLLoss(reduction='mean')
    loss_funct = torch.nn.CrossEntropyLoss(reduction='mean')


    for epoch in range(num_epochs):
        ## set epoch level varibles
        total_loss = 0.0

        ## Batch and Shuffle the Data
        random.shuffle(train_data)
        batches = get_batches(train_data, batch_size)

        for batch in batches:
            batch_data, batch_label = get_labels_and_data(batch)

            model.zero_grad()
            # print("the input dim ar currently", batch_data.shape)
            y_pred = model.forward(batch_data, batch=True)
            y_pred = y_pred.view(len(batch_data) * (chunk_size + 1), 27)
            batch_label = batch_label.view(len(batch_data)  * (chunk_size + 1))
            # print("the output dim ar currently", y_pred.shape)
            # y_pred = torch.transpose(y_pred, 1, 2)

            loss = loss_funct(y_pred, batch_label)
            total_loss += loss

            # Computes the gradient and takes the optimizer step
            loss.backward()
            optimizer.step()

        print("Total loss on epoch %i: %f" % (epoch, total_loss))

    return model
