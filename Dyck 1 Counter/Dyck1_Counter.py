import torch
import torch.nn as nn
import torch.optim as optim
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns


counter_input_size = 3
output_size = 1
counter_output_size = 3
hidden_size = 1


X = ['()','((', '()','))', '()',')(']
y = ['valid', 'invalid', 'valid', 'invalid', 'valid', 'invalid']
labels = ['valid', 'invalid']
vocab = ['(',')']

max_length =2
n_letters = len(vocab)


def classFromOutput(output):
    if output.item() > 0.5:
        category_i = 1
    else:
        category_i = 0
    return labels[category_i], category_i

def encode_sentence(sentence):
    rep = torch.zeros(max_length, 1, n_letters)
    if len(sentence) < max_length:
        for index, char in enumerate(sentence):
            pos = vocab.index(char)
            rep[index + 2][0][pos] = 1
    else:
        for index, char in enumerate(sentence):
            pos = vocab.index(char)
            rep[index][0][pos] = 1
    rep.requires_grad_(True)
    return rep

def encode_labels(label):
    return torch.tensor(labels.index(label), dtype=torch.float32)

def encode_dataset(sentences, labels):
    encoded_sentences = []
    encoded_labels = []
    for sentence in sentences:
        encoded_sentences.append(encode_sentence(sentence))
    for label in labels:
        encoded_labels.append(encode_labels(label))
    return encoded_sentences, encoded_labels





class Net(nn.Module):
    def __init__(self, counter_input_size, hidden_size, counter_output_size, output_size):
        super(Net, self).__init__()
        self.hidden_size = hidden_size

        #counter input is size 3. one input for (, one for ) and one for the count recurrent connection
        self.counter = nn.Linear(counter_input_size,counter_output_size)

        #hidden layer has 2 inputs, one from the counter, one recurrent,
        #and two outputs, one that goes to the ReLU activation and one recurrent.
        #this applies to hidden1 and hidden2

        self.hidden1 = nn.Linear(2*hidden_size,2*hidden_size)
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(2*hidden_size,2*hidden_size)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(counter_output_size,output_size)
        self.sig = nn.Sigmoid()

    def forward(self,x, counter_rec_input, hidden1_rec_input, hidden2_rec_input):
        combined = torch.cat((x,counter_rec_input))
        x = self.counter(combined)




