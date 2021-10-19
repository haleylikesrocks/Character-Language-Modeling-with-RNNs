# models.py

# import torch
# import torch.nn as nn
# from torch import optim
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


class RNNClassifier(ConsonantVowelClassifier):
    def __init__(self, embeddings, inp=300, hid=32, out=2):
        super().__init__()

        self.embedding = nn.Embedding(26, 50, padding_idx=0)
        self.V = nn.Linear(inp, hid)
        self.g = nn.ReLU()
        self.mid = nn.Linear(hid,hid)
        self.W = nn.Linear(hid, out)
        self.classify = nn.Softmax(out)

    def predict(self, context):
        y = self.embedding(context)

        raise Exception("Implement me")


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
    for i in range(len(list_of_exs)):
        for item in list_of_exs[i]:
            letters = []
            for letter in item:
                letter_idx = index.index_of(letter) if index.index_of(letter) != -1 else -1
                letters.append(letter_idx)
            data.append((letters, lables[i]))
    return data

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
    num_epochs = 8
    initial_learning_rate = 0.01
    batch_size = 32

    # Model specifications
    # model = RNNClassifier()
    # optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
    # loss_funct = torch.nn.CrossEntropyLoss() # what loss functions should we used NLL is need calcte after softmax but befor loss

    # Preprocess data
    print("Preprocessing the Training data")
    train_data = preprocess([train_cons_exs, train_vowel_exs], [0,1] ,vocab_index)
    dev_data = preprocess([dev_cons_exs, dev_vowel_exs], [0, 1], vocab_index)

    for epoch in range(num_epochs):
        # print("entering epoch %i" % epoch)
        # set epoch level varibles
        total_loss = 0.0
        accuracys = []

        #Batch the data
        random.shuffle(train_data)
        # batches = get_batches(train_data, batch_size)

        for data, label in train_data:
            # batch_data, batch_label = get_labels_and_data(batch)

            model.zero_grad()
            y_pred = model.forward(data)
            
            # calculate loss and accuracy
            loss = loss_funct(y_pred, label)
            total_loss += loss
            for i in range(len(batch)):
                ret = 1 if y_pred[i].max(0)[1] == label else 0
                accuracys.append(ret)
            
            # Computes the gradient and takes the optimizer step
            loss.backward()
            optimizer.step()

        # Dev Testing
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

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e
        :param context: a single character to score
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
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


class RNNLanguageModel(LanguageModel):
    def __init__(self):
        raise Exception("Implement me")

    def get_next_char_log_probs(self, context):
        raise Exception("Implement me")

    def get_log_prob_sequence(self, next_chars, context):
        raise Exception("Implement me")


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
    initial_learning_rate = 0.01
    batch_size = 32

    # Model specifications
    model = DANClassifier(word_embeddings)
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
    loss_funct = torch.nn.CrossEntropyLoss()

    # Preprocess data
    print("Preprocessing the Training data")
    train_data = []
    for item in train_exs: #for testing
        train_data.append((model.preprocess(item.words), item.label))
    
    dev_data = []
    for item in dev_exs:
        dev_data.append((model.preprocess(item.words), item.label))

    for epoch in range(num_epochs):
        # print("entering epoch %i" % epoch)
        # set epoch level varibles
        total_loss = 0.0
        accuracys = []

        #Batch the data
        random.shuffle(train_data)
        batches = get_batches(train_data, batch_size)

        for batch in batches:
            batch_data, batch_label = get_labels_and_data(batch)

            model.zero_grad()
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

        # Dev Testing
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
