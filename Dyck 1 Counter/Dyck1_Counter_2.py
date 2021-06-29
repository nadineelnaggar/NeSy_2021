import torch
import torch.nn as nn
import torch.optim as optim
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import random

"""
same as code Dyck1_Counter but with ( = 1 and ) = -1 encodings instead of binary vectors/one-hot
"""

counter_input_size = 3
output_size = 2
counter_output_size = 1
hidden_size = 1
input_size = 1

num_classes = 2
num_epochs = 10000


X = ['()','((', '()','))', '()',')(']
y = ['valid', 'invalid', 'valid', 'invalid', 'valid', 'invalid']
labels = ['valid', 'invalid']
vocab = ['(',')']

max_length =2
n_letters = len(vocab)


def classFromOutput(output):
    # if output.item() > 0.5:
    #     category_i = 1
    # else:
    #     category_i = 0
    # return labels[category_i], category_i
    top_n, top_i = output.data.topk(1)  # Tensor out of Variable with .data
    category_i = top_i[0]
    return labels[category_i], category_i

def encode_sentence(sentence):
    rep = torch.zeros(max_length, 1, input_size)
    for index, char in enumerate(sentence):
        pos = vocab.index(char)
        if pos == 0:
            rep[index][0] = 1
        elif pos == 1:
            rep[index][0] = -1
    rep.requires_grad_(True)
    return rep

def encode_labels(label):
    # return torch.tensor(labels.index(label), dtype=torch.float32)
    if label=='valid':
        return torch.tensor([1,0],dtype=torch.float32)
    elif label =='invalid':
        return torch.tensor([0,1],dtype=torch.float32)

def encode_dataset(sentences, labels):
    encoded_sentences = []
    encoded_labels = []
    for sentence in sentences:
        encoded_sentences.append(encode_sentence(sentence))
    for label in labels:
        encoded_labels.append(encode_labels(label))
    return encoded_sentences, encoded_labels


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
# print('length of training set = ', len(X_train))
# print('length of test set = ', len(X_test))

X_notencoded = X
y_notencoded = y
X_train_notencoded = X_train
y_train_notencoded = y_train
X_test_notencoded = X_test
y_test_notencoded = y_test
X_train, y_train = encode_dataset(X_train, y_train)
X_test, y_test = encode_dataset(X_test, y_test)
X_encoded, y_encoded = encode_dataset(X, y)

# print(X_train[0])
# print(X_train[0][0][0])
# print(X_train[0][0][0][0])


class Net(nn.Module):
    def __init__(self, counter_input_size, hidden_size, counter_output_size, output_size):
        super(Net, self).__init__()
        self.hidden_size = hidden_size

        #counter input is size 3. one input for (, one for ) and one for the count recurrent connection
        self.counter = nn.Linear(counter_input_size,counter_output_size)
        # self.counter.weight = nn.Parameter(torch.eye(3))
        # self.counter.weight=nn.Parameter(torch.tensor([[1,0,0],[0,1,0],[1,1,1]],dtype=torch.float32))
        self.counter.weight = nn.Parameter(torch.tensor([1,1],dtype=torch.float32))
        self.counter.bias = nn.Parameter(torch.tensor(0,dtype=torch.float32))

        #hidden layer has 2 inputs, one from the counter, one recurrent,
        #and two outputs, one that goes to the ReLU activation and one recurrent.
        #this applies to hidden1 and hidden2

        self.hidden1 = nn.Linear(2*hidden_size,hidden_size)
        self.hidden1.weight = nn.Parameter(torch.tensor([1,0],dtype=torch.float32))
        self.hidden1.bias = nn.Parameter(torch.tensor(0,dtype=torch.float32))
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(2*hidden_size,hidden_size)
        self.hidden2.weight = nn.Parameter(torch.tensor([-1,1],dtype=torch.float32))
        self.hidden2.bias = nn.Parameter(torch.tensor(0,dtype=torch.float32))
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(2*hidden_size,output_size)
        # self.out.weight = nn.Parameter(torch.tensor([[-1,0],[1,1]],dtype=torch.float32))
        self.out.weight = nn.Parameter(torch.tensor([[0,0],[1,1]],dtype=torch.float32))
        self.out.bias = nn.Parameter(torch.tensor([0,0],dtype=torch.float32))
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def forward(self,x, counter_rec_input, hidden1_rec_input, hidden2_rec_input, print_flag=False):
        if print_flag==True:
            print('input x = ',x)
            print('input counter_rec_input = ',counter_rec_input)
            print('input hidden1_rec_input = ',hidden1_rec_input)
            print('input hidden2_rec_input = ',hidden2_rec_input)
        counter_combined = torch.cat((x,counter_rec_input))
        if print_flag==True:
            print('counter_combined = ',counter_combined)
        counter_combined = self.counter(counter_combined)
        counter_rec_output = counter_combined.unsqueeze(dim=0)
        if print_flag==True:
            print('x after counter = ',counter_combined)
        # hidden1_input, hidden2_input, counter_rec_output = counter_combined.split(1)
        # print('hidden1_input = ',hidden1_input)
        # print('hidden2_input = ',hidden2_input)
        # print('counter_rec_output = ',counter_rec_output)
        # hidden1_combined = torch.cat((hidden1_input,hidden1_rec_input))
        hidden1_combined = torch.cat((counter_combined.unsqueeze(dim=0),hidden1_rec_input))
        if print_flag==True:
            print('hidden1_combined = ',hidden1_combined)
        # hidden2_combined = torch.cat((hidden2_input,hidden2_rec_input))
        hidden2_combined = torch.cat((counter_combined.unsqueeze(dim=0),hidden2_rec_input))
        if print_flag==True:
            print('hidden2_combined = ',hidden2_combined)
        hidden1_output = self.hidden1(hidden1_combined)
        if print_flag==True:
            print('hidden1_output = ',hidden1_output)
        # relu1_input, hidden1_rec_output = hidden1_output.split(1)
        hidden1_rec_output = hidden1_output.unsqueeze(dim=0)
        if print_flag==True:
            print('hidden1_rec_output = ',hidden1_rec_output)
            print('relu1_input = ',hidden1_output)
        # print('relu1_input = ',relu1_input)
        # print('hidden1_rec_output = ',hidden1_rec_output)
        # relu1_output = self.relu1(relu1_input)
        relu1_output = self.relu1(hidden1_output)
        if print_flag==True:
            print('relu1_output = ',relu1_output)
        hidden2_output = self.hidden2(hidden2_combined)
        if print_flag==True:
            print('hidden2_output = ',hidden2_output)
        # relu2_input, hidden2_rec_output = hidden2_output.split(1)
        hidden2_rec_output = hidden2_output.unsqueeze(dim=0)
        if print_flag==True:
            print('hidden2_rec_output = ',hidden2_rec_input)
        # print('relu2_input = ',relu2_input)
        # print('hidden2_rec_output = ',hidden2_rec_output)
        # relu2_output = self.relu2(relu2_input)
        if print_flag==True:
            print('relu2_input = ',hidden2_output)
        relu2_output = self.relu2(hidden2_output)
        if print_flag==True:
            print('relu2_output = ',relu2_output)
        output = torch.cat((relu1_output.unsqueeze(dim=0),relu2_output.unsqueeze(dim=0)))
        if print_flag == True:
            print('input to the output layer = ',output)
        output = self.out(output)
        if print_flag==True:
            print('output before sigmoid = ',output)
        # output = self.sig(output)
        output = self.softmax(output)
        if print_flag==True:
            print('output after sigmoid = ',output)
        return output, counter_rec_output,hidden1_rec_output,hidden2_rec_output


# class Net(nn.Module):
#     def __init__(self, counter_input_size, hidden_size, counter_output_size, output_size):
#         super(Net, self).__init__()
#         self.hidden_size = hidden_size
#         self.counter = nn.Linear(counter_input_size,counter_output_size)
#         self.counter.weight = nn.Parameter(torch.tensor([1,1,1],dtype=torch.float32))
#         self.counter.bias = nn.Parameter(torch.tensor(0,dtype=torch.float32))
#         self.hidden1 = nn.Linear()

model = Net(counter_input_size,hidden_size,counter_output_size,output_size)
# print('counter weight = ',model.counter.weight)
print('counter bias = ',model.counter.bias)
# print('hidden1_weight = ',model.hidden1.weight)
# print('hidden1_bias = ',model.hidden1.bias)
# print('output weight = ',model.out.weight)
# print('output bias = ',model.out.bias)
# rec_counter_input = torch.tensor([0],dtype=torch.float32)
# hidden1_rec_input = torch.tensor([0],dtype=torch.float32)
# hidden2_rec_input = torch.tensor([0],dtype=torch.float32)
# print(model(X_train[0][0][0],rec_counter_input,hidden1_rec_input,hidden2_rec_input))


def test_whole_dataset():
    model.eval()
    num_correct = 0
    num_samples = len(X_encoded)
    confusion = torch.zeros(num_classes, num_classes)
    expected_classes = []
    predicted_classes = []
    correct_guesses = []
    incorrect_guesses = []
    print('////////////////////////////////////////')
    print('TEST WHOLE DATASET')
    with torch.no_grad():
        for i in range(num_samples):
            class_category = y_notencoded[i]
            class_tensor = y_encoded[i]
            input_sentence = X_notencoded[i]
            input_tensor = X_encoded[i]

            counter_rec = torch.tensor([0], dtype=torch.float32)
            hidden1_rec = torch.tensor([0], dtype=torch.float32)
            hidden2_rec = torch.tensor([0], dtype=torch.float32)

            print('////////////////////////////////////////////')
            print('Test sample = ', input_sentence)

            for j in range(input_tensor.size()[0]):
                print('input tensor[j][0] = ', input_tensor[j][0])

                output, counter_rec,hidden1_rec,hidden2_rec = model(input_tensor[j][0],counter_rec,hidden1_rec,hidden2_rec,print_flag=True)
                print('counter_rec = ',counter_rec)
                print('hidden1_rec = ',hidden1_rec)
                print('hidden2_rec = ',hidden2_rec)
                print('output = ',output)

            guess, guess_i = classFromOutput(output)
            class_i = labels.index(class_category)
            print('predicted class = ', guess)
            print('actual class = ', class_category)
            confusion[class_i][guess_i] += 1
            predicted_classes.append(guess_i)
            expected_classes.append(class_i)

            if guess == class_category:
                num_correct += 1
                correct_guesses.append(input_sentence)
            else:
                incorrect_guesses.append(input_sentence)

    accuracy = num_correct / num_samples * 100
    print('confusion matrix for test set \n', confusion)
    conf_matrix = sklearn.metrics.confusion_matrix(expected_classes, predicted_classes)
    heat = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
    bottom1, top1 = heat.get_ylim()
    heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
    # plt.savefig('Counter_Sigmoid_Confusion_Matrix_Testing.png')
    # plt.show()
    print('correct guesses in testing = ', correct_guesses)
    print('incorrect guesses in testing = ', incorrect_guesses)
    return accuracy


print('test accuracy = ', test_whole_dataset())




all_losses = []
epoch_accuracies = []
all_epoch_incorrect_guesses = []
# initial_weights = []
# final_weights = []
# initial_gradients = []
# final_gradients = []
# initial_biases = []
# final_biases = []
epochs = []
initial_weights_counter = []
initial_weights_hidden1 = []
initial_weights_hidden2 = []
initial_weights_output = []
initial_biases_counter = []
initial_biases_hidden1 = []
initial_biases_hidden2 = []
initial_biases_output = []
# initial_gradients_counter = []
# initial_gradients_hidden1 = []
# initial_gradients_hidden2 = []
# initial_gradients_output = []

final_weights_counter = []
final_weights_hidden1 = []
final_weights_hidden2 = []
final_weights_output = []
final_biases_counter = []
final_biases_hidden1 = []
final_biases_hidden2 = []
final_biases_output = []
final_gradients_counter = []
final_gradients_hidden1 = []
final_gradients_hidden2 = []
final_gradients_output = []

learning_rate = 0.005
# criterion = nn.MSELoss()
# criterion=nn.CrossEntropyLoss()
criterion = nn.BCELoss()
# criterion=nn.BCEWithLogitsLoss()
optimiser = optim.SGD(model.parameters(), lr=learning_rate)
# optimiser=optim.Adam(model.parameters(),lr=learning_rate)


def train():
    for epoch in range(num_epochs):
        shuffle = True
        epoch_incorrect_guesses = []
        num_correct = 0
        current_loss = 0
        epochs.append(epoch)
        expected_classes = []
        predicted_classes = []

        print_flag=False
        initial_weights_counter.append(model.counter.weight.clone())
        # initial_gradients_counter.append(model.counter.weight.grad.clone())
        initial_biases_counter.append(model.counter.bias.clone())
        initial_weights_hidden1.append(model.hidden1.weight.clone())
        initial_biases_hidden1.append(model.hidden1.bias.clone())
        initial_weights_hidden2.append(model.hidden2.weight.clone())
        initial_biases_hidden2.append(model.hidden2.bias.clone())
        initial_weights_output.append(model.out.weight.clone())
        initial_biases_output.append(model.out.bias.clone())
        if epoch == (num_epochs-1):
            print_flag=True
        num_samples = len(X_train)
        order = []
        for x in range(num_samples):
            order.append(x)
        random.shuffle(order)

        if (epoch+1)%20==0 or epoch==0:
            print('initial counter weight = ', model.counter.weight)
            print('initial counter bias = ', model.counter.bias)
            print('initial counter gradient = ', model.counter.weight.grad)
            print('initial hidden1 weight = ', model.hidden1.weight)
            print('initial hidden1 bias = ', model.hidden1.bias)
            print('initial hidden1 gradient = ', model.hidden1.weight.grad)
            print('initial hidden2 weight = ', model.hidden2.weight)
            print('initial hidden2 bias = ', model.hidden2.bias)
            print('initial hidden2 gradient = ', model.hidden2.weight.grad)
            print('initial output weight = ',model.out.weight)
            print('initial output bias = ',model.out.bias)

        for i in range(len(X_train)):

            optimiser.zero_grad()
            correct = False
            if shuffle==False:
                input_tensor = X_train[i]
                target_tensor = y_train[i]
                input_sentence = X_train_notencoded[i]
                target_label = y_train_notencoded[i]
            elif shuffle==True:
                input_tensor = X_train[order[i]]
                target_tensor = y_train[order[i]]
                input_sentence = X_train_notencoded[order[i]]
                target_label = y_train_notencoded[order[i]]
            if print_flag==True:
                print('/////////////////////////////////////////////////')
                print('input_sentence = ',input_sentence)
                print('input_tensor = ',input_tensor)
            #initialise the values for the recurrent connections

            counter_rec = torch.tensor([0],dtype=torch.float32)
            hidden1_rec = torch.tensor([0],dtype=torch.float32)
            hidden2_rec = torch.tensor([0],dtype=torch.float32)

            for j in range(input_tensor.size()[0]):

                output_tensor, counter_rec, hidden1_rec,hidden2_rec = model(input_tensor[j][0],counter_rec,hidden1_rec,hidden2_rec)
                if print_flag==True:
                    print('counter rec = ',counter_rec)
                    print('hidden1_rec = ',hidden1_rec)
                    print('hidden2_rec = ',hidden2_rec)

            loss = criterion(output_tensor, target_tensor)
            loss.backward()
            optimiser.step()
            output_label, output_label_index = classFromOutput(output_tensor)
            if output_label==target_label:
                num_correct+=1
                correct=True

            if correct==False:
                epoch_incorrect_guesses.append(input_sentence)


            if print_flag==True:
                print('predicted label = ',output_label)
                print('actual label = ',target_label)
                if correct==True:
                    print('correct prediction')
                else:
                    print('incorrect prediction')
            current_loss+=loss.item()
            expected_classes.append(target_label)
            predicted_classes.append(output_label)

        accuracy = num_correct/num_samples*100
        epoch_accuracies.append(accuracy)
        print('Accuracy for epoch', epoch, '=', accuracy, '%')

        all_losses.append(current_loss/len(X_train))
        all_epoch_incorrect_guesses.append(epoch_incorrect_guesses)

        final_weights_counter.append(model.counter.weight.clone())
        final_gradients_counter.append(model.counter.weight.grad.clone())
        final_biases_counter.append(model.counter.bias.clone())
        final_weights_hidden1.append(model.hidden1.weight.clone())
        final_gradients_hidden1.append(model.hidden1.weight.grad.clone())
        final_biases_hidden1.append(model.hidden1.bias.clone())
        final_weights_hidden2.append(model.hidden2.weight.clone())
        final_gradients_hidden2.append(model.hidden2.weight.grad.clone())
        final_biases_hidden2.append(model.hidden2.bias.clone())
        final_weights_output.append(model.out.weight.clone())
        final_gradients_output.append(model.out.weight.grad.clone())
        final_biases_output.append(model.out.bias.clone())

        if (epoch+1)%20==0 or epoch==0:
            print('counter weight = ', model.counter.weight)
            print('counter bias = ', model.counter.bias)
            print('counter gradient = ', model.counter.weight.grad)
            print('hidden1 weight = ', model.hidden1.weight)
            print('hidden1 bias = ', model.hidden1.bias)
            print('hidden1 gradient = ', model.hidden1.weight.grad)
            print('hidden2 weight = ', model.hidden2.weight)
            print('hidden2 bias = ', model.hidden2.bias)
            print('hidden2 gradient = ', model.hidden2.weight.grad)
            print('output weight = ',model.out.weight)
            print('output bias = ',model.out.bias)
            print('output gradient = ',model.out.weight.grad)
            print('loss = ',loss.item())

        if epoch==(num_epochs-1):
            print('Final training accuracy = ', num_correct / len(X_train) * 100, '%')
            conf_matrix = sklearn.metrics.confusion_matrix(expected_classes, predicted_classes)
            heat = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
            bottom1, top1 = heat.get_ylim()
            heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
            print('confusion matrix for training set = \n', conf_matrix)
            # plt.show()
            print(all_epoch_incorrect_guesses)
    plt.plot(epochs,all_losses)
    # plt.show()

train()

def test():
    model.eval()
    num_correct = 0
    num_samples = len(X_test)
    confusion = torch.zeros(num_classes, num_classes)
    expected_classes = []
    predicted_classes = []
    correct_guesses = []
    incorrect_guesses = []
    print('////////////////////////////////////////')
    print('TEST WHOLE DATASET')
    with torch.no_grad():
        for i in range(num_samples):
            class_category = y_test_notencoded[i]
            class_tensor = y_test[i]
            input_sentence = X_test_notencoded[i]
            input_tensor = X_test[i]

            counter_rec = torch.tensor([0], dtype=torch.float32)
            hidden1_rec = torch.tensor([0], dtype=torch.float32)
            hidden2_rec = torch.tensor([0], dtype=torch.float32)

            print('////////////////////////////////////////////')
            print('Test sample = ', input_sentence)

            for j in range(input_tensor.size()[0]):
                print('input tensor[j][0] = ', input_tensor[j][0])

                output, counter_rec,hidden1_rec,hidden2_rec = model(input_tensor[j][0],counter_rec,hidden1_rec,hidden2_rec,print_flag=True)
                print('counter_rec = ',counter_rec)
                print('hidden1_rec = ',hidden1_rec)
                print('hidden2_rec = ',hidden2_rec)
                print('output = ',output)

            guess, guess_i = classFromOutput(output)
            class_i = labels.index(class_category)
            print('predicted class = ', guess)
            print('actual class = ', class_category)
            confusion[class_i][guess_i] += 1
            predicted_classes.append(guess_i)
            expected_classes.append(class_i)

            if guess == class_category:
                num_correct += 1
                correct_guesses.append(input_sentence)
            else:
                incorrect_guesses.append(input_sentence)

    accuracy = num_correct / num_samples * 100
    print('confusion matrix for test set \n', confusion)
    conf_matrix = sklearn.metrics.confusion_matrix(expected_classes, predicted_classes)
    heat = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
    bottom1, top1 = heat.get_ylim()
    heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
    # plt.savefig('Counter_Sigmoid_Confusion_Matrix_Testing.png')
    # plt.show()
    print('correct guesses in testing = ', correct_guesses)
    print('incorrect guesses in testing = ', incorrect_guesses)
    return accuracy


print('test accuracy = ', test())
