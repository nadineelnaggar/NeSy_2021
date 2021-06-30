import torch
import torch.nn as nn
import torch.optim as optim
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import seaborn as sns
import pandas


"""
this is an implementation of the image Dyck1_Counter_6.png but with a sigmoid activation instead of softmax
"""

max_length=4
num_epochs = 1000

vocab = [')','(']
labels = ['invalid','valid']

n_letters = len(vocab)
num_classes = len(labels)
input_size = 2
output_size = 1
hidden_1_size = 1
hidden_2_size = 2

# import the data

data = []
X = []
y = []


with open('Dyck1_Dataset_2pairs_balanced.txt','r') as f:
    for line in f:
        line = line.split(",")
        sentence = line[0].strip()
        label = line[1].strip()
        X.append(sentence)
        y.append(label)
        data.append((sentence,label))

# start the encoding of the dataset and labels

def encode_sentence(sentence):
    rep = torch.zeros(max_length,1,input_size)
    for index, char in enumerate(sentence):
        pos = vocab.index(char)
        rep[index][0][pos] = 1
        # if pos == 0:
        #     rep[index][0][pos] = 1
        # elif pos == 1:
        #     rep[index][0] = -1
    rep.requires_grad_(True)
    return rep

print(encode_sentence('()()'))
print(encode_sentence('()()')[0][0].size())


def encode_labels(label):
    # return torch.tensor(labels.index(label), dtype=torch.float32)
    if label=='valid':
        return torch.tensor(0,dtype=torch.float32)
    elif label =='invalid':
        return torch.tensor(1,dtype=torch.float32)

def encode_dataset(sentences, labels):
    encoded_sentences = []
    encoded_labels = []
    for sentence in sentences:
        encoded_sentences.append(encode_sentence(sentence))
    for label in labels:
        encoded_labels.append(encode_labels(label))
    return encoded_sentences, encoded_labels


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, stratify=y)
# print(X_train)
# print(y_train)

X_notencoded = X
y_notencoded = y
X_train_notencoded = X_train
X_test_notencoded = X_test
y_train_notencoded = y_train
y_test_notencoded = y_test

X_train, y_train = encode_dataset(X_train, y_train)
X_test, y_test = encode_dataset(X_test, y_test)
X_encoded, y_encoded = encode_dataset(X, y)


def classFromOutput(output):
    if output.item() > 0.5:
        category_i = 0
    else:
        category_i = 1
    return labels[category_i], category_i
    # top_n, top_i = output.data.topk(1)  # Tensor out of Variable with .data
    # category_i = top_i[0]
    # return labels[category_i], category_i


class Net(nn.Module):
    def __init__(self, input_size,output_size, hidden_size_1, hidden_size_2):
        super(Net, self).__init__()
        self.closing_filter = nn.Linear(input_size,hidden_size_1)
        self.closing_filter.weight = nn.Parameter(torch.tensor([1,0],dtype=torch.float32))
        self.closing_filter.bias = nn.Parameter(torch.tensor(0,dtype=torch.float32))
        self.closing_filter_relu = nn.ReLU()
        self.opening_filter = nn.Linear(input_size,hidden_size_1)
        self.opening_filter.weight = nn.Parameter(torch.tensor([0,1],dtype=torch.float32))
        self.opening_filter.bias = nn.Parameter(torch.tensor(0,dtype=torch.float32))
        self.opening_filter_relu = nn.ReLU()
        self.closing_bracket_counter = nn.Linear(hidden_size_2,hidden_size_1)
        self.closing_bracket_counter.weight = nn.Parameter(torch.tensor([1,1],dtype=torch.float32))
        self.closing_bracket_counter.bias = nn.Parameter(torch.tensor(0,dtype=torch.float32))
        self.closing_bracket_counter_relu = nn.ReLU()
        self.opening_bracket_counter = nn.Linear(hidden_size_2,hidden_size_1)
        self.opening_bracket_counter.weight=nn.Parameter(torch.tensor([1,1],dtype=torch.float32))
        self.opening_bracket_counter.bias = nn.Parameter(torch.tensor(0,dtype=torch.float32))
        self.opening_bracket_counter_relu = nn.ReLU()
        self.closing_minus_opening = nn.Linear(hidden_size_2,hidden_size_1)
        self.closing_minus_opening.weight = nn.Parameter(torch.tensor([1,-1],dtype=torch.float32))
        self.closing_minus_opening.bias = nn.Parameter(torch.tensor(0,dtype=torch.float32))
        self.closing_minus_opening_relu = nn.ReLU()
        self.opening_minus_closing = nn.Linear(hidden_size_2,hidden_size_1)
        self.opening_minus_closing.weight = nn.Parameter(torch.tensor([-1,1],dtype=torch.float32))
        self.opening_minus_closing.bias = nn.Parameter(torch.tensor(0,dtype=torch.float32))
        self.opening_minus_closing_relu = nn.ReLU()
        self.closing_bracket_surplus = nn.Linear(hidden_size_2,hidden_size_1)
        self.closing_bracket_surplus.weight = nn.Parameter(torch.tensor([1,1],dtype=torch.float32))
        self.closing_bracket_surplus.bias = nn.Parameter(torch.tensor(0,dtype=torch.float32))
        self.closing_bracket_surplus_relu = nn.ReLU()
        self.opening_minus_closing_copy = nn.Linear(hidden_size_1,hidden_size_1)
        self.opening_minus_closing_copy.weight = nn.Parameter(torch.tensor([1],dtype=torch.float32))
        self.opening_minus_closing_copy.bias = nn.Parameter(torch.tensor(0,dtype=torch.float32))
        self.opening_minus_closing_copy_relu = nn.ReLU()
        self.out = nn.Linear(hidden_size_2,output_size)
        self.out.weight = nn.Parameter(torch.tensor([1,1],dtype=torch.float32))
        self.out.bias = nn.Parameter(torch.tensor(0,dtype=torch.float32))
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, opening_brackets, closing_brackets, excess_closing_brackets):
        closing = self.closing_filter(x)
        closing = self.closing_filter_relu(closing)

        closing = torch.cat((closing.unsqueeze(dim=0),closing_brackets.unsqueeze(dim=0)))
        closing = self.closing_bracket_counter(closing)
        closing = self.closing_bracket_counter_relu(closing)
        closing_brackets = closing

        opening = self.opening_filter(x)
        opening = self.opening_filter_relu(opening)

        opening = torch.cat((opening.unsqueeze(dim=0),opening_brackets.unsqueeze(dim=0)))
        opening = self.opening_bracket_counter(opening)
        opening = self.opening_bracket_counter_relu(opening)
        opening_brackets=opening

        closing_minus_opening = torch.cat((closing.unsqueeze(dim=0),opening.unsqueeze(dim=0)))
        opening_minus_closing = torch.cat((closing.unsqueeze(dim=0),opening.unsqueeze(dim=0)))
        closing_minus_opening = self.closing_minus_opening(closing_minus_opening)
        closing_minus_opening = self.closing_minus_opening_relu(closing_minus_opening)
        opening_minus_closing = self.opening_minus_closing(opening_minus_closing)
        opening_minus_closing = self.opening_minus_closing_relu(opening_minus_closing)

        opening_minus_closing = self.opening_minus_closing_copy(opening_minus_closing.unsqueeze(dim=0))
        opening_minus_closing = self.opening_minus_closing_copy_relu(opening_minus_closing)
        surplus_closing_brackets = torch.cat((closing_minus_opening.unsqueeze(dim=0),excess_closing_brackets.unsqueeze(dim=0)))
        surplus_closing_brackets = self.closing_bracket_surplus(surplus_closing_brackets)
        surplus_closing_brackets = self.closing_bracket_surplus_relu(surplus_closing_brackets)

        output = torch.cat((surplus_closing_brackets.unsqueeze(dim=0),opening_minus_closing.unsqueeze(dim=0)))
        output = self.out(output)
        # output = self.softmax(output)
        output = self.sigmoid(output)
        return output, opening_brackets, closing_brackets, surplus_closing_brackets

model = Net(input_size,output_size,hidden_1_size,hidden_2_size)

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

            opening_bracket_count = torch.tensor(0, dtype=torch.float32)
            closing_bracket_count = torch.tensor(0, dtype=torch.float32)
            surplus_closing_bracket_count = torch.tensor(0, dtype=torch.float32)

            print('////////////////////////////////////////////')
            print('Test sample = ', input_sentence)

            for j in range(input_tensor.size()[0]):
                print('input tensor[j][0] = ', input_tensor[j][0])

                output, opening_bracket_count,closing_bracket_count,surplus_closing_bracket_count = model(input_tensor[j][0],opening_bracket_count,closing_bracket_count,surplus_closing_bracket_count)
                print('opening bracket count = ',opening_bracket_count)
                print('closing bracket count = ',closing_bracket_count)
                print('surplus closing bracket count = ',surplus_closing_bracket_count)
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
# initial_weights_counter = []
# initial_weights_hidden1 = []
# initial_weights_hidden2 = []
# initial_weights_output = []
# initial_biases_counter = []
# initial_biases_hidden1 = []
# initial_biases_hidden2 = []
# initial_biases_output = []
# # initial_gradients_counter = []
# # initial_gradients_hidden1 = []
# # initial_gradients_hidden2 = []
# # initial_gradients_output = []
#
# final_weights_counter = []
# final_weights_hidden1 = []
# final_weights_hidden2 = []
# final_weights_output = []
# final_biases_counter = []
# final_biases_hidden1 = []
# final_biases_hidden2 = []
# final_biases_output = []
# final_gradients_counter = []
# final_gradients_hidden1 = []
# final_gradients_hidden2 = []
# final_gradients_output = []


weight_opening_bracket_filter = []
bias_opening_bracket_filter = []
weight_closing_bracket_filter = []
bias_closing_bracket_filter = []
weight_opening_bracket_counter = []
bias_opening_bracket_counter = []
weight_closing_bracket_counter = []
bias_closing_bracket_counter = []
weight_opening_minus_closing = []
bias_opening_minus_closing = []
weight_opening_minus_closing_copy = []
bias_opening_minus_closing_copy = []
weight_closing_minus_opening = []
bias_closing_minus_opening = []
weight_surplus_closing_bracket_count = []
bias_surplus_closing_bracket_count = []
weight_output_layer = []
bias_output_layer = []



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
        # shuffle=False
        epoch_incorrect_guesses = []
        num_correct = 0
        current_loss = 0
        epochs.append(epoch)
        expected_classes = []
        predicted_classes = []

        print_flag=False
        # initial_weights_counter.append(model.counter.weight.clone())
        # # initial_gradients_counter.append(model.counter.weight.grad.clone())
        # initial_biases_counter.append(model.counter.bias.clone())
        # initial_weights_hidden1.append(model.hidden1.weight.clone())
        # initial_biases_hidden1.append(model.hidden1.bias.clone())
        # initial_weights_hidden2.append(model.hidden2.weight.clone())
        # initial_biases_hidden2.append(model.hidden2.bias.clone())
        # initial_weights_output.append(model.out.weight.clone())
        # initial_biases_output.append(model.out.bias.clone())
        if epoch == (num_epochs-1):
            print_flag=True
        num_samples = len(X_train)
        order = []
        for x in range(num_samples):
            order.append(x)
        random.shuffle(order)

        # if (epoch+1)%20==0 or epoch==0:
        #     print('initial counter weight = ', model.counter.weight)
        #     print('initial counter bias = ', model.counter.bias)
        #     print('initial counter gradient = ', model.counter.weight.grad)
        #     print('initial hidden1 weight = ', model.l2n1.weight)
        #     print('initial hidden1 bias = ', model.l2n1.bias)
        #     print('initial hidden1 gradient = ', model.l2n1.weight.grad)
        #     print('initial hidden2 weight = ', model.l2n2.weight)
        #     print('initial hidden2 bias = ', model.l2n2.bias)
        #     print('initial hidden2 gradient = ', model.l2n2.weight.grad)
        #     print('initial output weight = ',model.l3n1.weight)
        #     print('initial output bias = ',model.l3n1.bias)

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

            opening_bracket_count = torch.tensor(0,dtype=torch.float32)
            closing_bracket_count = torch.tensor(0,dtype=torch.float32)
            surplus_closing_bracket_count = torch.tensor(0,dtype=torch.float32)

            for j in range(input_tensor.size()[0]):

                output_tensor, opening_bracket_count, closing_bracket_count,surplus_closing_bracket_count = model(input_tensor[j][0],opening_bracket_count,closing_bracket_count,surplus_closing_bracket_count)
                if print_flag==True:
                    print('opening bracket count = ',opening_bracket_count)
                    print('closing bracket count = ',closing_bracket_count)
                    print('surplus closing bracket count = ',surplus_closing_bracket_count)

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

        # final_weights_counter.append(model.counter.weight.clone())
        # final_gradients_counter.append(model.counter.weight.grad.clone())
        # final_biases_counter.append(model.counter.bias.clone())
        # final_weights_hidden1.append(model.hidden1.weight.clone())
        # final_gradients_hidden1.append(model.hidden1.weight.grad.clone())
        # final_biases_hidden1.append(model.hidden1.bias.clone())
        # final_weights_hidden2.append(model.hidden2.weight.clone())
        # final_gradients_hidden2.append(model.hidden2.weight.grad.clone())
        # final_biases_hidden2.append(model.hidden2.bias.clone())
        # final_weights_output.append(model.out.weight.clone())
        # final_gradients_output.append(model.out.weight.grad.clone())
        # final_biases_output.append(model.out.bias.clone())

        if (epoch+1)%20==0 or epoch==0:
        #     print('initial counter weight = ', model.counter.weight)
        #     print('initial counter bias = ', model.counter.bias)
        #     print('initial counter gradient = ', model.counter.weight.grad)
        #     print('initial hidden1 weight = ', model.l2n1.weight)
        #     print('initial hidden1 bias = ', model.l2n1.bias)
        #     print('initial hidden1 gradient = ', model.l2n1.weight.grad)
        #     print('initial hidden2 weight = ', model.l2n2.weight)
        #     print('initial hidden2 bias = ', model.l2n2.bias)
        #     print('initial hidden2 gradient = ', model.l2n2.weight.grad)
        #     print('initial output weight = ',model.l3n1.weight)
        #     print('initial output bias = ',model.l3n1.bias)
            print('loss = ',loss.item())


        # print('loss = ', loss.item())

        weight_opening_bracket_filter.append(model.opening_filter.weight.clone())
        bias_opening_bracket_filter.append(model.opening_filter.bias.clone())
        weight_closing_bracket_filter.append(model.closing_filter.weight.clone())
        bias_closing_bracket_filter.append(model.closing_filter.bias.clone())
        weight_opening_bracket_counter.append(model.opening_bracket_counter.weight.clone())
        bias_opening_bracket_counter.append(model.opening_bracket_counter.bias.clone())
        weight_closing_bracket_counter.append(model.closing_bracket_counter.weight.clone())
        bias_closing_bracket_counter.append(model.closing_bracket_counter.bias.clone())
        weight_opening_minus_closing.append(model.opening_minus_closing.weight.clone())
        bias_opening_minus_closing.append(model.opening_minus_closing.bias.clone())
        weight_opening_minus_closing_copy.append(model.opening_minus_closing_copy.weight.clone())
        bias_opening_minus_closing_copy.append(model.opening_minus_closing_copy.bias.clone())
        weight_closing_minus_opening.append(model.closing_minus_opening.weight.clone())
        bias_closing_minus_opening.append(model.closing_minus_opening.bias.clone())
        weight_surplus_closing_bracket_count.append(model.closing_bracket_surplus.weight.clone())
        bias_surplus_closing_bracket_count.append(model.closing_bracket_surplus.bias.clone())
        weight_output_layer.append(model.out.weight.clone())
        bias_output_layer.append(model.out.bias.clone())


        if epoch==(num_epochs-1):
            print('Final training accuracy = ', num_correct / len(X_train) * 100, '%')
            conf_matrix = sklearn.metrics.confusion_matrix(expected_classes, predicted_classes)
            heat = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
            bottom1, top1 = heat.get_ylim()
            heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
            print('confusion matrix for training set = \n', conf_matrix)
            # plt.show()
            print(all_epoch_incorrect_guesses)

        if accuracy==100:
            print('Final training accuracy = ', num_correct / len(X_train) * 100, '%')
            conf_matrix = sklearn.metrics.confusion_matrix(expected_classes, predicted_classes)
            heat = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
            bottom1, top1 = heat.get_ylim()
            heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
            print('confusion matrix for training set = \n', conf_matrix)
            # plt.show()
            print(all_epoch_incorrect_guesses)
            break
    # plt.plot(epochs,all_losses)
    # plt.show()
    df1 = pandas.DataFrame()
    for i in range(len(epochs)):
        weight_opening_bracket_filter[i] = weight_opening_bracket_filter[i].detach().numpy()
        bias_opening_bracket_filter[i] = bias_opening_bracket_filter[i].detach().numpy()
        weight_closing_bracket_filter[i] = weight_closing_bracket_filter[i].detach().numpy()
        bias_closing_bracket_filter[i] = bias_closing_bracket_filter[i].detach().numpy()
        weight_opening_bracket_counter[i] = weight_opening_bracket_counter[i].detach().numpy()
        bias_opening_bracket_counter[i] = bias_opening_bracket_counter[i].detach().numpy()
        weight_closing_bracket_counter[i] = weight_closing_bracket_counter[i].detach().numpy()
        bias_closing_bracket_counter[i] = bias_closing_bracket_counter[i].detach().numpy()
        weight_opening_minus_closing[i] = weight_opening_minus_closing[i].detach().numpy()
        bias_opening_minus_closing[i] = bias_opening_minus_closing[i].detach().numpy()
        weight_opening_minus_closing_copy[i] = weight_opening_minus_closing_copy[i].detach().numpy()
        bias_opening_minus_closing_copy[i] = bias_opening_minus_closing_copy[i].detach().numpy()
        weight_closing_minus_opening[i] = weight_closing_minus_opening[i].detach().numpy()
        bias_closing_minus_opening[i] = bias_closing_minus_opening[i].detach().numpy()
        weight_surplus_closing_bracket_count[i] = weight_surplus_closing_bracket_count[i].detach().numpy()
        bias_surplus_closing_bracket_count[i] = bias_surplus_closing_bracket_count[i].detach().numpy()
        weight_output_layer[i] = weight_output_layer[i].detach().numpy()
        bias_output_layer[i] = bias_output_layer[i].detach().numpy()

    df1['epochs'] = epochs
    df1['weight_opening_bracket_filter'] = weight_opening_bracket_filter
    df1['bias_opening_bracket_filter'] = bias_opening_bracket_filter
    df1['weight_closing_bracket_filter'] = weight_closing_bracket_filter
    df1['bias_closing_bracket_filter'] = bias_closing_bracket_filter
    df1['weight_opening_bracket_counter'] = weight_opening_bracket_counter
    df1['bias_opening_bracket_counter'] = bias_opening_bracket_counter
    df1['weight_closing_bracket_counter'] = weight_closing_bracket_counter
    df1['bias_closing_bracket_counter'] = bias_closing_bracket_counter
    df1['weight_opening_minus_closing'] = weight_opening_minus_closing
    df1['bias_opening_minus_closing'] = bias_opening_minus_closing
    df1['weight_opening_minus_closing_copy'] = weight_opening_minus_closing_copy
    df1['bias_opening_minus_closing_copy'] = bias_opening_minus_closing_copy
    df1['weight_closing_minus_opening'] = weight_closing_minus_opening
    df1['bias_closing_minus_opening'] = bias_closing_minus_opening
    df1['weight_surplus_closing_bracket_count'] = weight_surplus_closing_bracket_count
    df1['bias_surplus_closing_bracket_count'] = bias_surplus_closing_bracket_count
    df1['weight_output_layer'] = weight_output_layer
    df1['bias_output_layer'] = bias_output_layer
    df1['all losses'] = all_losses
    df1['epoch accuracies'] = epoch_accuracies
    df1['epoch incorrect guesses'] = all_epoch_incorrect_guesses

    df1.to_excel('Dyck1_Counter_7_early_stopping_Sigmoid_BCE.xlsx')

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

            opening_bracket_count = torch.tensor(0, dtype=torch.float32)
            closing_bracket_count = torch.tensor(0, dtype=torch.float32)
            surplus_closing_bracket_count = torch.tensor(0, dtype=torch.float32)

            print('////////////////////////////////////////////')
            print('Test sample = ', input_sentence)

            for j in range(input_tensor.size()[0]):
                print('input tensor[j][0] = ', input_tensor[j][0])

                output_tensor, opening_bracket_count, closing_bracket_count, surplus_closing_bracket_count = model(
                    input_tensor[j][0], opening_bracket_count, closing_bracket_count, surplus_closing_bracket_count)

                print('opening bracket count = ', opening_bracket_count)
                print('closing bracket count = ', closing_bracket_count)
                print('surplus closing bracket count = ', surplus_closing_bracket_count)
                print('output = ',output_tensor)

            guess, guess_i = classFromOutput(output_tensor)
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



# data_length = []
# X_length = []
# y_length = []
#
#
# with open('Dyck1_Dataset_6pairs_balanced.txt','r') as f:
#     for line in f:
#         line = line.split(",")
#         sentence = line[0].strip()
#         label = line[1].strip()
#         X_length.append(sentence)
#         y_length.append(label)
#         data_length.append((sentence,label))
#
# # start the encoding of the dataset and labels
#
# def encode_sentence_length(sentence):
#     rep = torch.zeros(12,1,input_size)
#     for index, char in enumerate(sentence):
#         pos = vocab.index(char)
#         rep[index][0][pos] = 1
#         # if pos == 0:
#         #     rep[index][0][pos] = 1
#         # elif pos == 1:
#         #     rep[index][0] = -1
#     rep.requires_grad_(True)
#     return rep
#
#
# def encode_labels(label):
#     # return torch.tensor(labels.index(label), dtype=torch.float32)
#     if label=='valid':
#         return torch.tensor([0,1],dtype=torch.float32)
#     elif label =='invalid':
#         return torch.tensor([1,0],dtype=torch.float32)
#
# def encode_dataset_length(sentences, labels):
#     encoded_sentences = []
#     encoded_labels = []
#     for sentence in sentences:
#         encoded_sentences.append(encode_sentence_length(sentence))
#     for label in labels:
#         encoded_labels.append(encode_labels(label))
#     return encoded_sentences, encoded_labels
#
#
# X_length_encoded, y_length_encoded = encode_dataset_length(X_length, y_length)
#
# def test_length():
#     model.eval()
#     num_correct = 0
#     num_samples = len(X_length)
#     confusion = torch.zeros(num_classes, num_classes)
#     expected_classes = []
#     predicted_classes = []
#     correct_guesses = []
#     incorrect_guesses = []
#     print('////////////////////////////////////////')
#     print('TEST LENGTH DATASET')
#     with torch.no_grad():
#         for i in range(num_samples):
#             class_category = y_length[i]
#             class_tensor = y_length_encoded[i]
#             input_sentence = X_length[i]
#             input_tensor = X_length_encoded[i]
#
#             opening_bracket_count = torch.tensor(0, dtype=torch.float32)
#             closing_bracket_count = torch.tensor(0, dtype=torch.float32)
#             surplus_closing_bracket_count = torch.tensor(0, dtype=torch.float32)
#
#             # print('////////////////////////////////////////////')
#             # print('Test sample = ', input_sentence)
#
#             for j in range(input_tensor.size()[0]):
#                 # print('input tensor[j][0] = ', input_tensor[j][0])
#
#                 output_tensor, opening_bracket_count, closing_bracket_count, surplus_closing_bracket_count = model(
#                     input_tensor[j][0], opening_bracket_count, closing_bracket_count, surplus_closing_bracket_count)
#
#                 # print('opening bracket count = ', opening_bracket_count)
#                 # print('closing bracket count = ', closing_bracket_count)
#                 # print('surplus closing bracket count = ', surplus_closing_bracket_count)
#                 # print('output = ',output_tensor)
#
#             guess, guess_i = classFromOutput(output_tensor)
#             class_i = labels.index(class_category)
#             # print('predicted class = ', guess)
#             # print('actual class = ', class_category)
#             confusion[class_i][guess_i] += 1
#             predicted_classes.append(guess_i)
#             expected_classes.append(class_i)
#
#             if guess == class_category:
#                 num_correct += 1
#                 correct_guesses.append(input_sentence)
#             else:
#                 incorrect_guesses.append(input_sentence)
#
#     accuracy = num_correct / num_samples * 100
#     print('confusion matrix for test set \n', confusion)
#     conf_matrix = sklearn.metrics.confusion_matrix(expected_classes, predicted_classes)
#     heat = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
#     bottom1, top1 = heat.get_ylim()
#     heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
#     # plt.savefig('Counter_Sigmoid_Confusion_Matrix_Testing.png')
#     # plt.show()
#     print('correct guesses in testing = ', correct_guesses)
#     print('incorrect guesses in testing = ', incorrect_guesses)
#     return accuracy
#
#
# print('test length accuracy = ', test_length())


data_length = []
X_length = []
y_length = []


with open('Dyck1_Dataset_6pairs_balanced.txt','r') as f:
    for line in f:
        line = line.split(",")
        sentence = line[0].strip()
        label = line[1].strip()
        X_length.append(sentence)
        y_length.append(label)
        data_length.append((sentence,label))

# start the encoding of the dataset and labels

def encode_sentence_length(sentence):
    rep = torch.zeros(12,1,input_size)
    for index, char in enumerate(sentence):
        pos = vocab.index(char)
        rep[index][0][pos] = 1
        # if pos == 0:
        #     rep[index][0][pos] = 1
        # elif pos == 1:
        #     rep[index][0] = -1
    rep.requires_grad_(True)
    return rep


def encode_labels(label):
    # return torch.tensor(labels.index(label), dtype=torch.float32)
    if label=='valid':
        return torch.tensor(0,dtype=torch.float32)
    elif label =='invalid':
        return torch.tensor(1,dtype=torch.float32)

def encode_dataset_length(sentences, labels):
    encoded_sentences = []
    encoded_labels = []
    for sentence in sentences:
        encoded_sentences.append(encode_sentence_length(sentence))
    for label in labels:
        encoded_labels.append(encode_labels(label))
    return encoded_sentences, encoded_labels


X_length_encoded, y_length_encoded = encode_dataset_length(X_length, y_length)

def test_length():
    model.eval()
    num_correct = 0
    num_samples = len(X_length)
    confusion = torch.zeros(num_classes, num_classes)
    expected_classes = []
    predicted_classes = []
    correct_guesses = []
    incorrect_guesses = []
    print('////////////////////////////////////////')
    print('TEST LENGTH DATASET')
    with torch.no_grad():
        for i in range(num_samples):
            class_category = y_length[i]
            class_tensor = y_length_encoded[i]
            input_sentence = X_length[i]
            input_tensor = X_length_encoded[i]

            opening_bracket_count = torch.tensor(0, dtype=torch.float32)
            closing_bracket_count = torch.tensor(0, dtype=torch.float32)
            surplus_closing_bracket_count = torch.tensor(0, dtype=torch.float32)

            # print('////////////////////////////////////////////')
            # print('Test sample = ', input_sentence)

            for j in range(input_tensor.size()[0]):
                # print('input tensor[j][0] = ', input_tensor[j][0])

                output_tensor, opening_bracket_count, closing_bracket_count, surplus_closing_bracket_count = model(
                    input_tensor[j][0], opening_bracket_count, closing_bracket_count, surplus_closing_bracket_count)

                # print('opening bracket count = ', opening_bracket_count)
                # print('closing bracket count = ', closing_bracket_count)
                # print('surplus closing bracket count = ', surplus_closing_bracket_count)
                # print('output = ',output_tensor)

            guess, guess_i = classFromOutput(output_tensor)
            class_i = labels.index(class_category)
            # print('predicted class = ', guess)
            # print('actual class = ', class_category)
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


print('test length accuracy = ', test_length())