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
    for index, char in enumerate(sentence):
        pos = vocab.index(char)
        if pos == 0:
            rep[index][0][pos] = 1
        elif pos == 1:
            rep[index][0][pos] = -1
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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
print('length of training set = ', len(X_train))
print('length of test set = ', len(X_test))

X_notencoded = X
y_notencoded = y
X_train_notencoded = X_train
y_train_notencoded = y_train
X_test_notencoded = X_test
y_test_notencoded = y_test
X_train, y_train = encode_dataset(X_train, y_train)
X_test, y_test = encode_dataset(X_test, y_test)
X_encoded, y_encoded = encode_dataset(X, y)

print(X_train[0])
print(X_train[0][0][0])
print(X_train[0][0][0][0])


class Net(nn.Module):
    def __init__(self, counter_input_size, hidden_size, counter_output_size, output_size):
        super(Net, self).__init__()
        self.hidden_size = hidden_size

        #counter input is size 3. one input for (, one for ) and one for the count recurrent connection
        self.counter = nn.Linear(counter_input_size,counter_output_size)
        self.counter.weight = nn.Parameter(torch.eye(3))
        self.counter.bias = nn.Parameter(torch.tensor([0,0,0],dtype=torch.float32))

        #hidden layer has 2 inputs, one from the counter, one recurrent,
        #and two outputs, one that goes to the ReLU activation and one recurrent.
        #this applies to hidden1 and hidden2

        self.hidden1 = nn.Linear(2*hidden_size,2*hidden_size)
        self.hidden1.weight = nn.Parameter(torch.tensor([[1,0],[0,0]],dtype=torch.float32))
        self.hidden1.bias = nn.Parameter(torch.tensor([0,0],dtype=torch.float32))
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(2*hidden_size,2*hidden_size)
        self.hidden2.weight = nn.Parameter(torch.tensor([[1,0],[0,-1]],dtype=torch.float32))
        self.hidden2.bias = nn.Parameter(torch.tensor([0,0],dtype=torch.float32))
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(2*hidden_size,output_size)
        self.out.weight = nn.Parameter(torch.tensor([1,1],dtype=torch.float32))
        self.out.bias = nn.Parameter(torch.tensor(0,dtype=torch.float32))
        self.sig = nn.Sigmoid()

    def forward(self,x, counter_rec_input, hidden1_rec_input, hidden2_rec_input):
        print('x = ',x)
        print('counter_rec_input = ',counter_rec_input)
        print('hidden1_rec_input = ',hidden1_rec_input)
        print('hidden2_rec_input = ',hidden2_rec_input)
        counter_combined = torch.cat((x,counter_rec_input))
        print('counter_combined = ',counter_combined)
        counter_combined = self.counter(counter_combined)
        print('x after counter = ',counter_combined)
        # counter_rec_output = x[2]
        hidden1_input, hidden2_input, counter_rec_output = counter_combined.split(1)
        print('hidden1_input = ',hidden1_input)
        print('hidden2_input = ',hidden2_input)
        print('counter_rec_output = ',counter_rec_output)
        hidden1_combined = torch.cat((hidden1_input,hidden1_rec_input))
        print('hidden1_combined = ',hidden1_combined)
        hidden2_combined = torch.cat((hidden2_input,hidden2_rec_input))
        print('hidden2_combined = ',hidden2_combined)
        hidden1_output = self.hidden1(hidden1_combined)
        print('hidden1_output = ',hidden1_output)
        relu1_input, hidden1_rec_output = hidden1_output.split(1)
        print('relu1_input = ',relu1_input)
        print('hidden1_rec_output = ',hidden1_rec_output)
        relu1_output = self.relu1(relu1_input)
        print('relu1_output = ',relu1_output)
        hidden2_output = self.hidden2(hidden2_combined)
        print('hidden2_output = ',hidden2_output)
        relu2_input, hidden2_rec_output = hidden2_output.split(1)
        print('relu2_input = ',relu2_input)
        print('hidden2_rec_output = ',hidden2_rec_output)
        relu2_output = self.relu2(relu2_input)
        print('relu2_output = ',relu2_output)
        output = torch.cat((relu1_output,relu2_output))
        print('input to the output layer = ',output)
        output = self.out(output)
        print('output before sigmoid = ',output)
        output = self.sig(output)
        print('output after sigmoid = ',output)
        return output, counter_rec_output,hidden1_rec_output,hidden2_rec_output


model = Net(counter_input_size,hidden_size,counter_output_size,output_size)
print('counter weight = ',model.counter.weight)
print('counter bias = ',model.counter.bias)
print('hidden1_weight = ',model.hidden1.weight)
print('hidden1_bias = ',model.hidden1.bias)
print('output weight = ',model.out.weight)
print('output bias = ',model.out.bias)
rec_counter_input = torch.tensor([0],dtype=torch.float32)
hidden1_rec_input = torch.tensor([0],dtype=torch.float32)
hidden2_rec_input = torch.tensor([0],dtype=torch.float32)
print(model(X_train[0][0][0],rec_counter_input,hidden1_rec_input,hidden2_rec_input))







