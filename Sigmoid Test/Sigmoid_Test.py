import torch
import torch.nn as nn
import torch.optim as optim
import sklearn
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

num_epochs = 10000
num_classes = 2
output_size = 1
input_size=1

labels = ['<= 0', '> 0']

X = [1, -1, 0.0001, -10000000, 2, -3]
y = ['> 0', '<= 0', '> 0', '<= 0', '> 0', '<= 0']



def encode_sentence(sentence):
    # rep = torch.zeros(max_length, 1, n_letters)
    # if len(sentence) < max_length:
    #     for index, char in enumerate(sentence):
    #         pos = vocab.index(char)
    #         rep[index + 2][0][pos] = 1
    # else:
    #     for index, char in enumerate(sentence):
    #         pos = vocab.index(char)
    #         rep[index][0][pos] = 1
    # rep.requires_grad_(True)
    rep = torch.tensor(sentence,dtype=torch.float32,requires_grad=True)
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

# print(encode_sentence(-5))

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


def classFromOutput(output):
    if output.item() > 0.5:
        category_i = 1
    else:
        category_i = 0
    return labels[category_i], category_i

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size,output_size)
        self.fc1.weight = nn.Parameter(torch.tensor([1],dtype=torch.float32))
        self.fc1.bias = nn.Parameter(torch.tensor([0],dtype=torch.float32))
        self.sig = nn.Sigmoid()

    def forward(self,x):
        x = self.sig(x)
        return x

model = Net(input_size,output_size)
print(model.fc1.weight)
print(model.fc1.bias)
print(model.fc1.weight.grad)
print(model(torch.tensor(1,dtype=torch.float32)))
print(model(torch.tensor(-50000,dtype=torch.float32)))
print(model(torch.tensor(-1,dtype=torch.float32)))
print(model(torch.tensor(0,dtype=torch.float32)))
print(model(torch.tensor(100000000,dtype=torch.float32)))
print(model(torch.tensor(0.001,dtype=torch.float32)))
print(model(torch.tensor(0.01,dtype=torch.float32)))



learning_rate = 0.005
criterion = nn.MSELoss()
optimiser = optim.SGD(model.parameters(),lr=learning_rate)

all_losses = []
epoch_accuracies = []
initial_weights = []
final_weights = []
initial_gradients = []
final_gradients = []
initial_biases = []
final_biases = []
epochs = []


for epoch in range(num_epochs):
    #these first few lines are for plotting later on

    initial_biases.append(model.fc1.bias.clone())
    initial_weights.append(model.fc1.weight.clone())
    # initial_gradients.append(model.fc1.weight.grad.clone())
    num_correct = 0
    current_loss = 0
    epochs.append(epoch)
    confusion = torch.zeros(num_classes,num_classes)
    expected_classes = []
    predicted_classes = []


    # going through the training set and performing online training.
    for i in range(len(X_train)):
        optimiser.zero_grad()
        correct = False
        target_tensor = y_train[i]
        target_label = y_train_notencoded[i]
        input_sentence = X_train_notencoded[i]
        input_tensor = X_train[i]
        output_tensor = model(input_tensor)

        loss = criterion(output_tensor,target_tensor)
        loss.backward()
        optimiser.step()
        output_label, output_label_index = classFromOutput(output_tensor)
        if output_label==target_label:
            num_correct+=1
            correct=True
        current_loss+=loss.item()
        expected_classes.append(target_label)
        predicted_classes.append(output_label)


        if epoch == (num_epochs-1):
            # print('input number = ',input_sentence)
            print('////////////////////////////////////')
            print('input tensor = ',input_tensor)
            print('predicted class = ',output_label)
            print('actual class = ',target_label)
            if correct==True:
                print('Correct Prediction')
            else:
                print('Incorrect Prediction')


    accuracy = num_correct/len(X_train)*100
    epoch_accuracies.append(accuracy)
    all_losses.append(current_loss/len(X_train))
    final_weights.append(model.fc1.weight.clone())
    final_biases.append(model.fc1.bias.clone())
    # final_gradients.append(model.fc1.weight.grad.clone())
    print('Accuracy for epoch',epoch,'=',accuracy,'%')

    if (epoch+1)%20==0:
        print('weight = ', model.fc1.weight)
        print('bias = ', model.fc1.bias)
        print('gradient = ', model.fc1.weight.grad)

    if epoch==(num_epochs-1):
        print('weight = ',model.fc1.weight)
        print('bias = ',model.fc1.bias)
        print('gradient = ',model.fc1.weight.grad)
        print('confusion matrix for training set\n', confusion)
        print('Final training accuracy = ', num_correct / len(X_train) * 100, '%')
        conf_matrix = sklearn.metrics.confusion_matrix(expected_classes, predicted_classes)
        heat = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
        bottom1, top1 = heat.get_ylim()
        heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
        print('confusion matrix for training set = \n', conf_matrix)
        plt.savefig('Test_Sigmoid_Confusion_Matrix_Training.png')
        plt.show()

