import torch
import torch.nn as nn
import torch.optim as optim
import sklearn
from sklearn.model_selection import train_test_split
import pandas
import seaborn as sns
import random
import matplotlib.pyplot as plt

document_name = 'Dyck1_Counter_10_L1_Regularisation_Random_Weights_Sigmoid_BCE_run_3.txt'
excel_name = 'Dyck1_Counter_10_L1_Regularisation_Random_Weights_Sigmoid_BCE_run_3.xlsx'

num_epochs = 1000
max_length = 2

input_size = 2
output_size = 1
hidden_size = 2
counter_input_size = 3
counter_output_size = 1
vocab = ['(', ')']
labels = ['empty','not empty']
n_letters = len(vocab)
num_classes = 2





def classFromOutput(output):
    if output.item() > 0.5:
        category_i = 1
    else:
        category_i = 0
    return labels[category_i], category_i

def encode_sentence(sentence):
    rep = torch.zeros(max_length, 1, n_letters)
    # if len(sentence) < max_length:
    #     for index, char in enumerate(sentence):
    #         pos = vocab.index(char)
    #         rep[index][0][pos] = 1
    # else:
    for index, char in enumerate(sentence):
        pos = vocab.index(char)
        rep[index][0][pos] = 1
    rep.requires_grad_(True)
    return rep

def encode_labels(label):
    return torch.tensor([labels.index(label)], dtype=torch.float32)

def encode_dataset(sentences, labels):
    encoded_sentences = []
    encoded_labels = []
    for sentence in sentences:
        encoded_sentences.append(encode_sentence(sentence))
    for label in labels:
        encoded_labels.append(encode_labels(label))
    return encoded_sentences, encoded_labels


X = ['()','((',')(', '((','))','((']
y = ['empty','not empty','empty', 'not empty', 'empty', 'not empty']

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


class Counter(nn.Module):
    def __init__(self, input_size, hidden_size, counter_input_size, counter_output_size, output_size):
        super(Counter, self).__init__()
        self.hidden_size = hidden_size
        # self.fc1 = nn.Linear(input_size,hidden_size)
        # self.fc1.weight = nn.Parameter(torch.eye(2))
        # self.fc1.bias = nn.Parameter(torch.tensor([0],dtype=torch.float32))
        self.counter = nn.Linear(counter_input_size,counter_output_size)
        # self.counter.weight = nn.Parameter(torch.tensor([[1,0,1],[0,-1,1]],dtype=torch.float32))
        # self.counter.weight = nn.Parameter(torch.tensor([[1, -1, 1]], dtype=torch.float32))
        # self.counter.bias = nn.Parameter(torch.tensor([0],dtype=torch.float32))
        self.fc2 = nn.Linear(counter_output_size,output_size)
        # self.fc2.weight = nn.Parameter(torch.tensor([[1]],dtype=torch.float32))
        # self.fc2.bias = nn.Parameter(torch.tensor([0],dtype=torch.float32))
        self.out = nn.Sigmoid()

    def forward(self,x,previous_count):
        # x = self.fc1(x)
        combined = torch.cat((x,previous_count))
        x = self.counter(combined)
        previous_count=x
        x = self.fc2(x)
        x = self.out(x)
        return x, previous_count

model = Counter(input_size,hidden_size, counter_input_size, counter_output_size,output_size)
# print(model.counter.weight)
# test_input = torch.tensor([1,0],dtype=torch.float32)
# test_input_2 = torch.tensor([0,1],dtype=torch.float32)
# count = torch.tensor([0],dtype=torch.float32)
# op, count = model(test_input,count)
# print(op)
# print(count)
# op, count = model(test_input_2,count)
# print(op)
# print(count)
# op, count = model(test_input_2,count)
# print(op)
# print(count)


def test_whole_dataset():
    model.eval()
    num_correct = 0
    num_samples = len(X)
    confusion = torch.zeros(num_classes, num_classes)
    expected_classes = []
    predicted_classes = []
    correct_guesses = []
    incorrect_guesses = []
    with open(document_name,'w') as f:
        f.write('////////////////////////////////////////\n')
        f.write('TEST WHOLE DATASET\n')
    print('////////////////////////////////////////')
    print('TEST WHOLE DATASET')
    with torch.no_grad():
        for i in range(num_samples):
            class_category = y[i]
            class_tensor = y_encoded[i]
            input_sentence = X[i]
            input_tensor = X_encoded[i]

            print('////////////////////////////////////////////')
            print('Test sample = ', input_sentence)
            with open(document_name,'a') as f:
                f.write('////////////////////////////////////////////\n')
                f.write('Test sample = '+input_sentence+'\n')
            count = torch.tensor([0],dtype=torch.float32)
            for j in range(input_tensor.size()[0]):
                # print('input tensor[j][0] = ', input_tensor[j][0])
                with open(document_name,'a') as f:
                    f.write('input tensor[j][0] = '+input_sentence[j][0]+'\n')

                output, count = model(input_tensor[j][0],count)
                # print('count = ',count)
                # print('output = ',output)
                with open(document_name,'a') as f:
                    f.write('count = '+str(count)+'\n')
                    f.write('output = '+str(output)+'\n')

            guess, guess_i = classFromOutput(output)
            class_i = labels.index(class_category)
            print('predicted class = ', guess)
            print('actual class = ', class_category)
            with open(document_name,'a') as f:
                f.write('predicted class = '+guess+'\n')
                f.write('actual class = '+class_category+'\n')
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
    with open(document_name, 'a') as f:
        f.write('confusion matrix for test set \n' + str(confusion) + '\n')
        f.write('correct guesses in testing = ' + str(correct_guesses) + '\n')
        f.write('incorrect guesses in testing = ' + str(incorrect_guesses) + '\n')
        f.write('accuracy = ' + str(accuracy) + '\n')
    return accuracy


print('test accuracy = ', test_whole_dataset())


# initial_input_layer_weights = model.fc1.weight.clone().detach().numpy()
# initial_counter_weights = model.counter.weight.clone().detach().numpy()

learning_rate = 0.005
# criterion = nn.MSELoss()
criterion = nn.BCELoss()
optimiser = optim.SGD(model.parameters(),lr=learning_rate)

def train(input_tensor, class_tensor, input_sentence='', print_flag=False):
    optimiser.zero_grad()
    if print_flag == True:
        print('////////////////////////////////////////')
        print('input sentence = ', input_sentence)
    count = torch.tensor([0],dtype=torch.float32)
    for i in range(input_tensor.size()[0]):
        output, count = model(input_tensor[i].squeeze(), count)
        if print_flag == True:
            print('count = ',count)

    loss = criterion(output, class_tensor)
    l1_regularisation = torch.tensor([0], dtype=torch.float32)
    for param in model.parameters():
        l1_regularisation += torch.norm(param, 1)
    loss = loss + l1_regularisation
    if print_flag == True:
        print('Loss = ', loss)

    # model.zero_grad()
    loss.backward()
    optimiser.step()
    if print_flag == True:
        print('output in train function = ', output)

    return output, loss.item()

all_losses = []
current_loss = 0
all_epoch_incorrect_guesses = []
df1 = pandas.DataFrame()
epochs = []
confusion_matrices = []

weights_input_layer = []
weights_counter = []
biases_counter = []
weights_output_layer = []
biases_output_layer = []
gradients_input_layer = []
gradients_counter = []
gradients_output_layer = []
weight_11_input_layer = []
weight_12_input_layer = []
weight_21_input_layer = []
weight_22_input_layer = []
weight_1_counter = []
weight_2_counter = []
weight_3_counter = []
# weight_output_layer = []
gradient_11_input_layer = []
gradient_12_input_layer = []
gradient_21_input_layer = []
gradient_22_input_layer = []
# gradient_output_layer = []
gradient_1_counter = []
gradient_2_counter = []
gradient_3_counter = []
accuracies = []
gradients = []



for epoch in range(num_epochs):
    confusion = torch.zeros(num_classes, num_classes)
    num_correct = 0
    num_samples = len(X_train)
    current_loss = 0
    epoch_incorrect_guesses = []
    predicted_classes = []
    expected_classes = []
    epochs.append(epoch)

    for i in range(len(X_train)):
        input_tensor = X_train[i]
        class_tensor = y_train[i]
        input_sentence = X_train_notencoded[i]
        class_category = y_train_notencoded[i]
        if epoch == num_epochs - 1:
            output, loss = train(input_tensor, class_tensor, input_sentence, True)
        else:
            output, loss = train(input_tensor, class_tensor, input_sentence, False)
        guess, guess_i = classFromOutput(output)
        class_i = labels.index(class_category)
        confusion[class_i][guess_i] += 1
        current_loss += loss
        expected_classes.append(class_i)
        predicted_classes.append(guess_i)
        if guess == class_category:
            num_correct += 1
        else:
            epoch_incorrect_guesses.append(input_sentence)

        # weights_input_layer.append(model.fc1.weight.clone())
        # weights_output_layer.append(model.fc2.weight.clone())
        # gradients_input_layer.append(model.fc1.weight.grad.clone())
        # gradients_output_layer.append(model.fc2.weight.grad.clone())
        # weights_counter.append(model.counter.weight.clone())
        # gradients_counter.append(model.counter.weight.grad.clone())
        # gradients.append(input_tensor.grad.clone())
    accuracy = num_correct / len(X_train) * 100
    if (epoch + 1) % 10 == 0:
        # print('input layer weights = ', model.fc1.weight)
        # print('input layer weight gradient = ', model.fc1.weight.grad)
        # print('input layer bias = ',model.fc1.bias)
        print('counter weights = ',model.counter.weight)
        print('counter gradients = ',model.counter.weight.grad)
        print('counter bias = ',model.counter.bias)
        print('output layer weights  = ', model.fc2.weight)
        print('output layer weight gradient = ', model.fc2.weight.grad)
        print('output layer bias = ',model.fc2.bias)

        # print('counter weights = ',model.counter.weight)
        # print('counter weight gradients = ',model.counter.weight.grad)

    print('Accuracy for epoch ', epoch, '=', accuracy, '%')
    all_losses.append(current_loss / len(X_train))
    all_epoch_incorrect_guesses.append(epoch_incorrect_guesses)
    conf_matrix = sklearn.metrics.confusion_matrix(expected_classes, predicted_classes)
    confusion_matrices.append(conf_matrix)

    accuracies.append(accuracy)
    # weights_input_layer.append(model.fc1.weight.clone())
    weights_output_layer.append(model.fc2.weight.clone())
    # gradients_input_layer.append(model.fc1.weight.grad.clone())
    gradients_output_layer.append(model.fc2.weight.grad.clone())
    weights_counter.append(model.counter.weight.clone())
    gradients_counter.append(model.counter.weight.grad.clone())
    gradients.append(input_tensor.grad.clone())
    biases_counter.append(model.counter.bias.clone())
    biases_output_layer.append(model.fc2.bias.clone())

    if epoch == num_epochs - 1:
        print('\n////////////////////////////////////////////////////////////////////////////////////////\n')
        print('confusion matrix for training set\n', confusion)
        print('Final training accuracy = ', num_correct / len(X_train) * 100, '%')
        conf_matrix = sklearn.metrics.confusion_matrix(expected_classes, predicted_classes)
        heat = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
        bottom1, top1 = heat.get_ylim()
        heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
        print('confusion matrix for training set = \n', conf_matrix)
        # plt.savefig('Counter_Sigmoid_Confusion_Matrix_Training.png')
        # plt.show()
        if i == len(X_train) - 1:
            print('input tensor = ', input_tensor)
            print('input tensor gradient = ', input_tensor.grad)
            # print('input layer weight = ', model.fc1.weight)
            # print('input layer weight gradient = ', model.fc1.weight.grad)
            print('counter weights = ', model.counter.weight)
            print('counter weight gradient = ',model.counter.weight.grad)
            print('output layer weight = ', model.fc2.weight)
            print('output layer weight gradient = ', model.fc2.weight.grad)


print('all incorrect guesses in training across all epochs = \n', all_epoch_incorrect_guesses)

for i in range(len(epochs)):
    weights_counter[i] = weights_counter[i].detach().numpy()
    weights_output_layer[i] = weights_output_layer[i].detach().numpy()
    biases_counter[i] = biases_counter[i].detach().numpy()
    biases_output_layer[i] = biases_output_layer[i].detach().numpy()
    # # weights_input_layer[i] = weights_input_layer[i].detach().numpy()
    # weights_output_layer[i] = weights_output_layer[i].detach().numpy()
    # weights_counter[i] = weights_counter[i].detach().numpy()
    # # gradients_input_layer[i] = gradients_input_layer[i].detach().numpy()
    # gradients_output_layer[i] = gradients_output_layer[i].detach().numpy()
    # gradients_counter[i] = gradients_counter[i].detach().numpy()
    # # weight_11_input_layer.append(weights_input_layer[i][0][0])
    # # weight_12_input_layer.append(weights_input_layer[i][0][1])
    # # weight_21_input_layer.append(weights_input_layer[i][1][0])
    # # weight_22_input_layer.append(weights_input_layer[i][1][1])
    # # # gradient_11_input_layer.append(gradients_input_layer[i][0][0])
    # # # gradient_12_input_layer.append(gradients_input_layer[i][0][1])
    # # # gradient_21_input_layer.append(gradients_input_layer[i][1][0])
    # # # gradient_22_input_layer.append(gradients_input_layer[i][1][1])
    # # weight_1_counter.append(weights_counter[i][0][0])
    # # weight_2_counter.append(weights_counter[i][0][1])
    # # weight_3_counter.append(weights_counter[i][0][2])
    # # gradient_1_counter.append(gradients_counter[i][0][0])
    # # gradient_2_counter.append(gradients_counter[i][0][1])
    # # gradient_3_counter.append(gradients_counter[i][0][2])


# df1 = pandas.DataFrame()
df1['epochs'] = epochs
df1['counter weights'] = weights_counter
df1['counter biases'] = biases_counter
df1['output layer weights'] = weights_output_layer
df1['output layer biases'] = biases_output_layer
df1['losses'] = all_losses
df1['epoch accuracies'] = accuracies
df1['epoch incorrect guesses'] = all_epoch_incorrect_guesses
df1['confusion matrices'] = confusion_matrices

df1.to_excel(excel_name)

# df1.to_excel('Dyck1_Counter_10_Sigmoid_BCE.xlsx')

#
# df1['weight_11_input_layer'] = weight_11_input_layer
# df1['weight_12_input_layer'] = weight_12_input_layer
# df1['weight_21_input_layer'] = weight_21_input_layer
# df1['weight_22_input_layer'] = weight_22_input_layera
# # df1['gradient_11_input_layer'] = gradient_11_input_layer
# # df1['gradient_12_input_layer'] = gradient_12_input_layer
# # df1['gradient_21_input_layer'] = gradient_21_input_layer
# # df1['gradient_22_input_layer'] = gradient_22_input_layer
# df1['weight_1_counter'] = weight_1_counter
# df1['weight_2_counter'] = weight_2_counter
# df1['weight_3_counter'] = weight_3_counter
# df1['gradient_1_counter'] = gradient_1_counter
# df1['gradient_2_counter'] = gradient_2_counter
# df1['gradient_3_counter'] = gradient_3_counter
# df1['weight_output_layer'] = weights_output_layer
# df1['gradient_output_layer'] = gradients_output_layer
# df1['accuracies'] = accuracies
# df1['losses'] = all_losses
#
# epochs = []
# for i in range(num_epochs):
#     epochs.append(i)
#
# fig = plt.figure()
# plt.plot(epochs, weight_11_input_layer, label='Input Layer Increment Weight 1 (W11)')
# plt.plot(epochs, weight_12_input_layer, label='Input Layer Increment Weight 2 (W12)')
# plt.plot(epochs, weight_21_input_layer, label='Input Layer Decrement Weight 1 (W21)')
# plt.plot(epochs, weight_22_input_layer, label='Input Layer Decrement Weight 2 (W22)')
# plt.xlabel('Epoch')
# plt.ylabel('Weight Value')
# plt.legend(loc='lower center')
# # plt.savefig('Counter_Sigmoid_Plot_Input_Layer_Weights.png')
# plt.show()
#
# fig2 = plt.figure()
# plt.plot(epochs, weight_1_counter, label='Counter Weight 1')
# plt.plot(epochs, weight_2_counter, label='Counter Weight 2')
# plt.plot(epochs, weight_3_counter, label='Counter Weight 3')
# plt.plot(epochs, weights_output_layer, label='Output Layer Weight ')
# plt.xlabel('Epoch')
# plt.ylabel('Weight Value')
# plt.legend(loc='lower center')
# fig.subplots_adjust(bottom=0.5)
# # plt.savefig('Counter_Sigmoid_Plot_Counter_and_Output_Layer_Weights.png')
# plt.show()
#
# fig3 = plt.figure()
# plt.plot(epochs, gradient_11_input_layer, label='Gradient for Increment Input 1')
# plt.plot(epochs, gradient_12_input_layer, label='Gradient for Increment Input 2')
# plt.plot(epochs, gradient_21_input_layer, label='Gradient for Decrement Input 1')
# plt.plot(epochs, gradient_22_input_layer, label='Gradient for Decrement Input 2')
# plt.xlabel('Epoch')
# plt.ylabel('Weight Gradient Value')
# plt.legend(loc='lower center')
# fig.subplots_adjust(bottom=0.5)
# # plt.savefig('Counter_Sigmoid_Plot_Input_Layer_Gradients.png')
# plt.show()
#
#
# fig3 = plt.figure()
# plt.plot(epochs, gradient_1_counter, label='Counter Weight Gradient 1')
# plt.plot(epochs, gradient_2_counter, label='Counter Weight Gradient 2')
# plt.plot(epochs, gradient_3_counter, label='Counter Weight Gradient 3')
# plt.xlabel('Epoch')
# plt.ylabel('Weight Gradient Value')
# plt.legend(loc='lower center')
# fig.subplots_adjust(bottom=0.5)
# # plt.savefig('Counter_Sigmoid_Plot_Counter_and_Output_Layer_Gradients.png')
# plt.show()
#
# fig4 = plt.figure()
# plt.plot(epochs, accuracies, label='Training Accuracies')
# plt.plot(epochs, all_losses, label='Losses')
# plt.xlabel('Epoch')
# plt.ylabel('Value')
# plt.legend(loc='lower center')
# fig.subplots_adjust(bottom=0.5)
# # plt.savefig('Counter_Sigmoid_Plot_Accuracies_and_Losses.png')
# plt.show()
#
#
# df1.to_excel('Counter_Sigmoid.xlsx')
#
# weight_1 = []
# weight_2 = []
# u = []
# v = []
#
# for i in range(len(gradient_11_input_layer)):
#     if gradient_11_input_layer[i]!=0 or gradient_12_input_layer!=0:
#         weight_1.append(weight_11_input_layer[i])
#         weight_2.append(weight_12_input_layer[i])
#         u.append(gradient_11_input_layer[i])
#         v.append(gradient_12_input_layer[i])
#
# xaxis = np.arange(-2,2,0.2)
# yaxis=np.arange(-2,2,0.2)
#
# A, B = np.meshgrid(weight_1, weight_2)
# fig5, ax = plt.subplots()
# ax.quiver(A, B, u, v)
# ax.xaxis.set_ticks(xaxis)
# ax.yaxis.set_ticks(yaxis)
# plt.xticks(rotation=90)
# plt.xlabel('Input Layer Weight 11 (Increment Weight 1)')
# plt.ylabel('Input Layer Weight 12 (Increment Weight 2)')
# # plt.savefig('Counter_Sigmoid_Vector_Field_Input_Layer_W11_W12.png')
# plt.show()
#
#
# weight_1 = []
# weight_2 = []
# u = []
# v = []
#
# for i in range(len(gradient_21_input_layer)):
#     if gradient_21_input_layer[i]!=0 or gradient_22_input_layer!=0:
#         weight_1.append(weight_21_input_layer[i])
#         weight_2.append(weight_22_input_layer[i])
#         u.append(gradient_21_input_layer[i])
#         v.append(gradient_22_input_layer[i])
#
# xaxis = np.arange(-2,2,0.2)
# yaxis=np.arange(-2,2,0.2)
#
# A, B = np.meshgrid(weight_1, weight_2)
# fig5, ax = plt.subplots()
# ax.quiver(A, B, u, v)
# ax.xaxis.set_ticks(xaxis)
# ax.yaxis.set_ticks(yaxis)
# plt.xticks(rotation=90)
# plt.xlabel('Input Layer Weight 21 (Decrement Weight 1)')
# plt.ylabel('Input Layer Weight 22 (Decrement Weight 2)')
# # plt.savefig('Counter_Sigmoid_Vector_Field_Input_Layer_W21_W22.png')
# plt.show()



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
    print('TEST')
    with open(document_name,'a') as f:
        f.write('////////////////////////////////////////\n')
        f.write('TEST\n')

    with torch.no_grad():
        for i in range(num_samples):
            class_category = y_test_notencoded[i]
            class_tensor = y_test[i]
            input_sentence = X_test_notencoded[i]
            input_tensor = X_test[i]

            print('////////////////////////////////////////////')
            print('Test sample = ', input_sentence)
            with open(document_name,'a') as f:
                f.write('////////////////////////////////////////////\n')
                f.write('Test sample = '+input_sentence+'\n')
            count = torch.tensor([0],dtype=torch.float32)
            for j in range(input_tensor.size()[0]):
                print('input tensor[j][0] = ', input_tensor[j][0])
                with open(document_name,'a') as f:
                    f.write('input tensor[j][0] = '+str(input_tensor[j][0])+'\n')

                output, count = model(input_tensor[j][0],count)
                print('count = ',count)
                print('output = ',output)
                with open(document_name,'a') as f:
                    f.write('count = '+str(count)+'\n')
                    f.write('output = '+str(output)+'\n')

            guess, guess_i = classFromOutput(output)
            class_i = labels.index(class_category)
            print('predicted class = ', guess)
            print('actual class = ', class_category)
            confusion[class_i][guess_i] += 1
            predicted_classes.append(guess_i)
            expected_classes.append(class_i)

            with open(document_name,'a') as f:
                f.write('predicted class = '+guess+'\n')
                f.write('actual class = '+class_category+'\n')

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
    with open(document_name,'a') as f:
        f.write('confusion matrix for test set \n'+str(confusion)+'\n')
        f.write('correct guesses in testing = '+str(correct_guesses)+'\n')
        f.write('incorrect guesses in testing = '+str(incorrect_guesses)+'\n')
        f.write('accuracy = ' + str(accuracy) + '\n')
    return accuracy


print('test accuracy = ', test())


# X_length = ['()()','(()(',')()(', '((()',')))(','((((', '((()))','(())((','(())(())', '((((()))']
# y_length = ['empty','not empty','empty', 'not empty', 'empty', 'not empty', 'empty', 'not empty', 'empty', 'not empty']
#
#
#
# def encode_sentence_length(sentence):
#     rep = torch.zeros(8, 1, n_letters)
#     # if len(sentence) < max_length:
#     #     for index, char in enumerate(sentence):
#     #         pos = vocab.index(char)
#     #         rep[index][0][pos] = 1
#     # else:
#     for index, char in enumerate(sentence):
#         pos = vocab.index(char)
#         rep[index][0][pos] = 1
#     rep.requires_grad_(True)
#     return rep

X_length = ['()()','(()(',')()(', '((()',')))(','((((', '((()))','(())((','(())(())','((((()))','()()()()(())',')))(((()((', '((((()))))(())(())()','((()((()()((())(())(','(((((((((((((((((((())))))))))','(())(())(())((()))((()))((()))','))(()(((()(())()((())()(())()(','(((((((())))((()(()()((()()))(()((()()(()((())(()())()(()(()','))(())()((())(())()((()))((()))(())(())(())(())((']
y_length = ['empty','not empty','empty', 'not empty', 'empty', 'not empty', 'empty', 'not empty', 'empty', 'not empty', 'empty','not empty', 'empty', 'not empty','not empty','empty', 'not empty', 'not empty', 'not empty']

print(len(X_length))
print(len(y_length))

def encode_sentence_length(sentence):
    rep = torch.zeros(60, 1, n_letters)
    # if len(sentence) < max_length:
    #     for index, char in enumerate(sentence):
    #         pos = vocab.index(char)
    #         rep[index][0][pos] = 1
    # else:
    for index, char in enumerate(sentence):
        pos = vocab.index(char)
        rep[index][0][pos] = 1
    rep.requires_grad_(True)
    return rep

def encode_labels(label):
    return torch.tensor([labels.index(label)], dtype=torch.float32)

def encode_dataset_length(sentences, labels):
    encoded_sentences = []
    encoded_labels = []
    for sentence in sentences:
        encoded_sentences.append(encode_sentence_length(sentence))
    for label in labels:
        encoded_labels.append(encode_labels(label))
    return encoded_sentences, encoded_labels

X_length_encoded, y_length_encoded = encode_dataset_length(X_length,y_length)

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
    print('TEST LENGTH')
    with open(document_name,'a') as f:
        f.write('////////////////////////////////////////\n')
        f.write('TEST LENGTH\n')
    with torch.no_grad():
        for i in range(num_samples):
            class_category = y_length[i]
            class_tensor = y_length_encoded[i]
            input_sentence = X_length[i]
            input_tensor = X_length_encoded[i]

            print('////////////////////////////////////////////')
            print('Test sample = ', input_sentence)

            with open(document_name,'a') as f:
                f.write('////////////////////////////////////////////\n')
                f.write('Test sample = '+input_sentence+'\n')
            count = torch.tensor([0],dtype=torch.float32)
            for j in range(input_tensor.size()[0]):
                # print('input tensor[j][0] = ', input_tensor[j][0])

                output, count = model(input_tensor[j][0],count)
                # print('count = ',count)
                # print('output = ',output)

            guess, guess_i = classFromOutput(output)
            class_i = labels.index(class_category)
            print('predicted class = ', guess)
            print('actual class = ', class_category)
            confusion[class_i][guess_i] += 1
            predicted_classes.append(guess_i)
            expected_classes.append(class_i)
            print('output = ',output)
            print('count = ',count)

            with open(document_name,'a') as f:
                f.write('predicted class = '+guess+'\n')
                f.write('actual class = '+class_category+'\n')

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
    with open(document_name,'a') as f:
        f.write('confusion matrix for test set \n'+str(confusion)+'\n')
        f.write('correct guesses in testing = '+str(correct_guesses)+'\n')
        f.write('incorrect guesses in testing = '+str(incorrect_guesses)+'\n')
        f.write('accuracy = ' + str(accuracy) + '\n')
    return accuracy


print('test accuracy = ', test_length())





# import torch
# import torch.nn as nn
# import torch.optim as optim
# import sklearn
# from sklearn.model_selection import train_test_split
# import pandas
# import seaborn as sns
# import random
# import matplotlib.pyplot as plt
#
# # input_size = 2
# # counter_rec_input_size = 1
# # counter_output_size = 1
# # output_size = 1
# #
# # class Net(nn.Module):
# #     def __init__(self, input_size, counter_rec_input_size, counter_output_size, output_size):
# #         super(Net, self).__init__()
# #         self.counter = nn.Linear(input_size+counter_rec_input_size,counter_output_size)
# #         self.out = nn.Linear(counter_output_size,output_size)
# #         self.sig = nn.Sigmoid()
# #
# #     def forward(self,x,counter_rec_input):
# #         counter_combined = torch.cat((x,counter_rec_input))
# #         counter_combined = self.counter(counter_combined)
# #         counter_rec_output = counter_combined
# #         out = self.out(counter_combined)
# #         out = self.sig(out)
# #         return out, counter_rec_output
# #
# #
# #
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # import sklearn
# # from sklearn.model_selection import train_test_split
# # import matplotlib.pyplot as plt
# # import numpy as np
# # import pandas
# # import seaborn as sns
# #
# #
# #
#
# num_epochs = 1000
# max_length = 2
#
# input_size = 2
# output_size = 1
# hidden_size = 2
# counter_input_size = 3
# counter_output_size = 1
# vocab = ['(', ')']
# labels = ['empty','not empty']
# n_letters = len(vocab)
# num_classes = 2
#
#
#
#
#
# def classFromOutput(output):
#     if output.item() > 0.5:
#         category_i = 1
#     else:
#         category_i = 0
#     return labels[category_i], category_i
#
# def encode_sentence(sentence):
#     rep = torch.zeros(max_length, 1, n_letters)
#     # if len(sentence) < max_length:
#     #     for index, char in enumerate(sentence):
#     #         pos = vocab.index(char)
#     #         rep[index][0][pos] = 1
#     # else:
#     for index, char in enumerate(sentence):
#         pos = vocab.index(char)
#         rep[index][0][pos] = 1
#     rep.requires_grad_(True)
#     return rep
#
# def encode_labels(label):
#     return torch.tensor([labels.index(label)], dtype=torch.float32)
#
# def encode_dataset(sentences, labels):
#     encoded_sentences = []
#     encoded_labels = []
#     for sentence in sentences:
#         encoded_sentences.append(encode_sentence(sentence))
#     for label in labels:
#         encoded_labels.append(encode_labels(label))
#     return encoded_sentences, encoded_labels
#
#
# X = ['()','((',')(', '((','))','((']
# y = ['empty','not empty','empty', 'not empty', 'empty', 'not empty']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
# print('length of training set = ', len(X_train))
# print('length of test set = ', len(X_test))
#
# X_notencoded = X
# y_notencoded = y
# X_train_notencoded = X_train
# y_train_notencoded = y_train
# X_test_notencoded = X_test
# y_test_notencoded = y_test
# X_train, y_train = encode_dataset(X_train, y_train)
# X_test, y_test = encode_dataset(X_test, y_test)
# X_encoded, y_encoded = encode_dataset(X, y)
#
#
# class Counter(nn.Module):
#     def __init__(self, input_size, hidden_size, counter_input_size, counter_output_size, output_size):
#         super(Counter, self).__init__()
#         self.hidden_size = hidden_size
#         # self.fc1 = nn.Linear(input_size,hidden_size)
#         # self.fc1.weight = nn.Parameter(torch.eye(2))
#         # self.fc1.bias = nn.Parameter(torch.tensor([0],dtype=torch.float32))
#         self.counter = nn.Linear(counter_input_size,counter_output_size)
#         # self.counter.weight = nn.Parameter(torch.tensor([[1,0,1],[0,-1,1]],dtype=torch.float32))
#         # self.counter.weight = nn.Parameter(torch.tensor([[1, -1, 1]], dtype=torch.float32))
#         # self.counter.bias = nn.Parameter(torch.tensor([0],dtype=torch.float32))
#         self.fc2 = nn.Linear(counter_output_size,output_size)
#         # self.fc2.weight = nn.Parameter(torch.tensor([[1]],dtype=torch.float32))
#         # self.fc2.bias = nn.Parameter(torch.tensor([0],dtype=torch.float32))
#         self.out = nn.Sigmoid()
#
#     def forward(self,x,previous_count):
#         # x = self.fc1(x)
#         combined = torch.cat((x,previous_count))
#         x = self.counter(combined)
#         previous_count=x
#         x = self.fc2(x)
#         x = self.out(x)
#         return x, previous_count
#
# model = Counter(input_size,hidden_size, counter_input_size, counter_output_size,output_size)
# print(model.counter.weight)
# test_input = torch.tensor([1,0],dtype=torch.float32)
# test_input_2 = torch.tensor([0,1],dtype=torch.float32)
# count = torch.tensor([0],dtype=torch.float32)
# op, count = model(test_input,count)
# print(op)
# print(count)
# op, count = model(test_input_2,count)
# print(op)
# print(count)
# op, count = model(test_input_2,count)
# print(op)
# print(count)
#
# # initial_input_layer_weights = model.fc1.weight.clone().detach().numpy()
# # initial_counter_weights = model.counter.weight.clone().detach().numpy()
#
# learning_rate = 0.005
# # criterion = nn.MSELoss()
# criterion = nn.BCELoss()
# optimiser = optim.SGD(model.parameters(),lr=learning_rate)
#
# def train(input_tensor, class_tensor, input_sentence='', print_flag=False):
#     optimiser.zero_grad()
#     if print_flag == True:
#         print('////////////////////////////////////////')
#         print('input sentence = ', input_sentence)
#     count = torch.tensor([0],dtype=torch.float32)
#     for i in range(input_tensor.size()[0]):
#         output, count = model(input_tensor[i].squeeze(), count)
#         if print_flag == True:
#             print('count = ',count)
#
#     loss = criterion(output, class_tensor)
#     if print_flag == True:
#         print('Loss = ', loss)
#
#     # model.zero_grad()
#     loss.backward()
#     optimiser.step()
#     if print_flag == True:
#         print('output in train function = ', output)
#
#     return output, loss.item()
#
# all_losses = []
# current_loss = 0
# all_epoch_incorrect_guesses = []
# df1 = pandas.DataFrame()
# epochs = []
#
# weights_input_layer = []
# weights_counter = []
# biases_counter = []
# weights_output_layer = []
# biases_output_layer = []
# gradients_input_layer = []
# gradients_counter = []
# gradients_output_layer = []
# weight_11_input_layer = []
# weight_12_input_layer = []
# weight_21_input_layer = []
# weight_22_input_layer = []
# weight_1_counter = []
# weight_2_counter = []
# weight_3_counter = []
# # weight_output_layer = []
# gradient_11_input_layer = []
# gradient_12_input_layer = []
# gradient_21_input_layer = []
# gradient_22_input_layer = []
# # gradient_output_layer = []
# gradient_1_counter = []
# gradient_2_counter = []
# gradient_3_counter = []
# accuracies = []
# gradients = []
#
#
#
# for epoch in range(num_epochs):
#     confusion = torch.zeros(num_classes, num_classes)
#     num_correct = 0
#     num_samples = len(X_train)
#     current_loss = 0
#     epoch_incorrect_guesses = []
#     predicted_classes = []
#     expected_classes = []
#     epochs.append(epoch)
#
#     for i in range(len(X_train)):
#         input_tensor = X_train[i]
#         class_tensor = y_train[i]
#         input_sentence = X_train_notencoded[i]
#         class_category = y_train_notencoded[i]
#         if epoch == num_epochs - 1:
#             output, loss = train(input_tensor, class_tensor, input_sentence, True)
#         else:
#             output, loss = train(input_tensor, class_tensor, input_sentence, False)
#         guess, guess_i = classFromOutput(output)
#         class_i = labels.index(class_category)
#         confusion[class_i][guess_i] += 1
#         current_loss += loss
#         expected_classes.append(class_i)
#         predicted_classes.append(guess_i)
#         if guess == class_category:
#             num_correct += 1
#         else:
#             epoch_incorrect_guesses.append(input_sentence)
#
#         # weights_input_layer.append(model.fc1.weight.clone())
#         # weights_output_layer.append(model.fc2.weight.clone())
#         # gradients_input_layer.append(model.fc1.weight.grad.clone())
#         # gradients_output_layer.append(model.fc2.weight.grad.clone())
#         # weights_counter.append(model.counter.weight.clone())
#         # gradients_counter.append(model.counter.weight.grad.clone())
#         # gradients.append(input_tensor.grad.clone())
#     accuracy = num_correct / len(X_train) * 100
#     if (epoch + 1) % 10 == 0:
#         # print('input layer weights = ', model.fc1.weight)
#         # print('input layer weight gradient = ', model.fc1.weight.grad)
#         # print('input layer bias = ',model.fc1.bias)
#         print('counter weights = ',model.counter.weight)
#         print('counter gradients = ',model.counter.weight.grad)
#         print('counter bias = ',model.counter.bias)
#         print('output layer weights  = ', model.fc2.weight)
#         print('output layer weight gradient = ', model.fc2.weight.grad)
#         print('output layer bias = ',model.fc2.bias)
#
#         # print('counter weights = ',model.counter.weight)
#         # print('counter weight gradients = ',model.counter.weight.grad)
#
#     print('Accuracy for epoch ', epoch, '=', accuracy, '%')
#     all_losses.append(current_loss / len(X_train))
#     all_epoch_incorrect_guesses.append(epoch_incorrect_guesses)
#
#     accuracies.append(accuracy)
#     # weights_input_layer.append(model.fc1.weight.clone())
#     weights_output_layer.append(model.fc2.weight.clone())
#     # gradients_input_layer.append(model.fc1.weight.grad.clone())
#     gradients_output_layer.append(model.fc2.weight.grad.clone())
#     weights_counter.append(model.counter.weight.clone())
#     gradients_counter.append(model.counter.weight.grad.clone())
#     gradients.append(input_tensor.grad.clone())
#     biases_counter.append(model.counter.bias.clone())
#     biases_output_layer.append(model.fc2.bias.clone())
#
#     if epoch == num_epochs - 1:
#         print('\n////////////////////////////////////////////////////////////////////////////////////////\n')
#         print('confusion matrix for training set\n', confusion)
#         print('Final training accuracy = ', num_correct / len(X_train) * 100, '%')
#         conf_matrix = sklearn.metrics.confusion_matrix(expected_classes, predicted_classes)
#         heat = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d")
#         bottom1, top1 = heat.get_ylim()
#         heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
#         print('confusion matrix for training set = \n', conf_matrix)
#         # plt.savefig('Counter_Sigmoid_Confusion_Matrix_Training.png')
#         # plt.show()
#         if i == len(X_train) - 1:
#             print('input tensor = ', input_tensor)
#             print('input tensor gradient = ', input_tensor.grad)
#             # print('input layer weight = ', model.fc1.weight)
#             # print('input layer weight gradient = ', model.fc1.weight.grad)
#             print('counter weights = ', model.counter.weight)
#             print('counter weight gradient = ',model.counter.weight.grad)
#             print('output layer weight = ', model.fc2.weight)
#             print('output layer weight gradient = ', model.fc2.weight.grad)
#
#
# print('all incorrect guesses in training across all epochs = \n', all_epoch_incorrect_guesses)
#
# for i in range(len(weights_input_layer)):
#     # weights_input_layer[i] = weights_input_layer[i].detach().numpy()
#     weights_output_layer[i] = weights_output_layer[i].detach().numpy()
#     weights_counter[i] = weights_counter[i].detach().numpy()
#     # gradients_input_layer[i] = gradients_input_layer[i].detach().numpy()
#     gradients_output_layer[i] = gradients_output_layer[i].detach().numpy()
#     gradients_counter[i] = gradients_counter[i].detach().numpy()
#     # weight_11_input_layer.append(weights_input_layer[i][0][0])
#     # weight_12_input_layer.append(weights_input_layer[i][0][1])
#     # weight_21_input_layer.append(weights_input_layer[i][1][0])
#     # weight_22_input_layer.append(weights_input_layer[i][1][1])
#     # # gradient_11_input_layer.append(gradients_input_layer[i][0][0])
#     # # gradient_12_input_layer.append(gradients_input_layer[i][0][1])
#     # # gradient_21_input_layer.append(gradients_input_layer[i][1][0])
#     # # gradient_22_input_layer.append(gradients_input_layer[i][1][1])
#     # weight_1_counter.append(weights_counter[i][0][0])
#     # weight_2_counter.append(weights_counter[i][0][1])
#     # weight_3_counter.append(weights_counter[i][0][2])
#     # gradient_1_counter.append(gradients_counter[i][0][0])
#     # gradient_2_counter.append(gradients_counter[i][0][1])
#     # gradient_3_counter.append(gradients_counter[i][0][2])
#
#
# # df1 = pandas.DataFrame()
# df1['epochs'] = epochs
# df1['counter weights'] = weights_counter
# df1['counter biases'] = biases_counter
# df1['output layer weights'] = weights_output_layer
# df1['output layer biases'] = biases_output_layer
# df1['losses'] = all_losses
# df1['epoch accuracies'] = accuracies
#
# df1.to_excel('Dyck1_Counter_10_no_initialisation_Sigmoid_BCE.xlsx')
#
# #
# # df1['weight_11_input_layer'] = weight_11_input_layer
# # df1['weight_12_input_layer'] = weight_12_input_layer
# # df1['weight_21_input_layer'] = weight_21_input_layer
# # df1['weight_22_input_layer'] = weight_22_input_layera
# # # df1['gradient_11_input_layer'] = gradient_11_input_layer
# # # df1['gradient_12_input_layer'] = gradient_12_input_layer
# # # df1['gradient_21_input_layer'] = gradient_21_input_layer
# # # df1['gradient_22_input_layer'] = gradient_22_input_layer
# # df1['weight_1_counter'] = weight_1_counter
# # df1['weight_2_counter'] = weight_2_counter
# # df1['weight_3_counter'] = weight_3_counter
# # df1['gradient_1_counter'] = gradient_1_counter
# # df1['gradient_2_counter'] = gradient_2_counter
# # df1['gradient_3_counter'] = gradient_3_counter
# # df1['weight_output_layer'] = weights_output_layer
# # df1['gradient_output_layer'] = gradients_output_layer
# # df1['accuracies'] = accuracies
# # df1['losses'] = all_losses
# #
# # epochs = []
# # for i in range(num_epochs):
# #     epochs.append(i)
# #
# # fig = plt.figure()
# # plt.plot(epochs, weight_11_input_layer, label='Input Layer Increment Weight 1 (W11)')
# # plt.plot(epochs, weight_12_input_layer, label='Input Layer Increment Weight 2 (W12)')
# # plt.plot(epochs, weight_21_input_layer, label='Input Layer Decrement Weight 1 (W21)')
# # plt.plot(epochs, weight_22_input_layer, label='Input Layer Decrement Weight 2 (W22)')
# # plt.xlabel('Epoch')
# # plt.ylabel('Weight Value')
# # plt.legend(loc='lower center')
# # # plt.savefig('Counter_Sigmoid_Plot_Input_Layer_Weights.png')
# # plt.show()
# #
# # fig2 = plt.figure()
# # plt.plot(epochs, weight_1_counter, label='Counter Weight 1')
# # plt.plot(epochs, weight_2_counter, label='Counter Weight 2')
# # plt.plot(epochs, weight_3_counter, label='Counter Weight 3')
# # plt.plot(epochs, weights_output_layer, label='Output Layer Weight ')
# # plt.xlabel('Epoch')
# # plt.ylabel('Weight Value')
# # plt.legend(loc='lower center')
# # fig.subplots_adjust(bottom=0.5)
# # # plt.savefig('Counter_Sigmoid_Plot_Counter_and_Output_Layer_Weights.png')
# # plt.show()
# #
# # fig3 = plt.figure()
# # plt.plot(epochs, gradient_11_input_layer, label='Gradient for Increment Input 1')
# # plt.plot(epochs, gradient_12_input_layer, label='Gradient for Increment Input 2')
# # plt.plot(epochs, gradient_21_input_layer, label='Gradient for Decrement Input 1')
# # plt.plot(epochs, gradient_22_input_layer, label='Gradient for Decrement Input 2')
# # plt.xlabel('Epoch')
# # plt.ylabel('Weight Gradient Value')
# # plt.legend(loc='lower center')
# # fig.subplots_adjust(bottom=0.5)
# # # plt.savefig('Counter_Sigmoid_Plot_Input_Layer_Gradients.png')
# # plt.show()
# #
# #
# # fig3 = plt.figure()
# # plt.plot(epochs, gradient_1_counter, label='Counter Weight Gradient 1')
# # plt.plot(epochs, gradient_2_counter, label='Counter Weight Gradient 2')
# # plt.plot(epochs, gradient_3_counter, label='Counter Weight Gradient 3')
# # plt.xlabel('Epoch')
# # plt.ylabel('Weight Gradient Value')
# # plt.legend(loc='lower center')
# # fig.subplots_adjust(bottom=0.5)
# # # plt.savefig('Counter_Sigmoid_Plot_Counter_and_Output_Layer_Gradients.png')
# # plt.show()
# #
# # fig4 = plt.figure()
# # plt.plot(epochs, accuracies, label='Training Accuracies')
# # plt.plot(epochs, all_losses, label='Losses')
# # plt.xlabel('Epoch')
# # plt.ylabel('Value')
# # plt.legend(loc='lower center')
# # fig.subplots_adjust(bottom=0.5)
# # # plt.savefig('Counter_Sigmoid_Plot_Accuracies_and_Losses.png')
# # plt.show()
# #
# #
# # df1.to_excel('Counter_Sigmoid.xlsx')
# #
# # weight_1 = []
# # weight_2 = []
# # u = []
# # v = []
# #
# # for i in range(len(gradient_11_input_layer)):
# #     if gradient_11_input_layer[i]!=0 or gradient_12_input_layer!=0:
# #         weight_1.append(weight_11_input_layer[i])
# #         weight_2.append(weight_12_input_layer[i])
# #         u.append(gradient_11_input_layer[i])
# #         v.append(gradient_12_input_layer[i])
# #
# # xaxis = np.arange(-2,2,0.2)
# # yaxis=np.arange(-2,2,0.2)
# #
# # A, B = np.meshgrid(weight_1, weight_2)
# # fig5, ax = plt.subplots()
# # ax.quiver(A, B, u, v)
# # ax.xaxis.set_ticks(xaxis)
# # ax.yaxis.set_ticks(yaxis)
# # plt.xticks(rotation=90)
# # plt.xlabel('Input Layer Weight 11 (Increment Weight 1)')
# # plt.ylabel('Input Layer Weight 12 (Increment Weight 2)')
# # # plt.savefig('Counter_Sigmoid_Vector_Field_Input_Layer_W11_W12.png')
# # plt.show()
# #
# #
# # weight_1 = []
# # weight_2 = []
# # u = []
# # v = []
# #
# # for i in range(len(gradient_21_input_layer)):
# #     if gradient_21_input_layer[i]!=0 or gradient_22_input_layer!=0:
# #         weight_1.append(weight_21_input_layer[i])
# #         weight_2.append(weight_22_input_layer[i])
# #         u.append(gradient_21_input_layer[i])
# #         v.append(gradient_22_input_layer[i])
# #
# # xaxis = np.arange(-2,2,0.2)
# # yaxis=np.arange(-2,2,0.2)
# #
# # A, B = np.meshgrid(weight_1, weight_2)
# # fig5, ax = plt.subplots()
# # ax.quiver(A, B, u, v)
# # ax.xaxis.set_ticks(xaxis)
# # ax.yaxis.set_ticks(yaxis)
# # plt.xticks(rotation=90)
# # plt.xlabel('Input Layer Weight 21 (Decrement Weight 1)')
# # plt.ylabel('Input Layer Weight 22 (Decrement Weight 2)')
# # # plt.savefig('Counter_Sigmoid_Vector_Field_Input_Layer_W21_W22.png')
# # plt.show()
#
#
#
# def test():
#     model.eval()
#     num_correct = 0
#     num_samples = len(X_test)
#     confusion = torch.zeros(num_classes, num_classes)
#     expected_classes = []
#     predicted_classes = []
#     correct_guesses = []
#     incorrect_guesses = []
#     print('////////////////////////////////////////')
#     print('TEST')
#     with torch.no_grad():
#         for i in range(num_samples):
#             class_category = y_test_notencoded[i]
#             class_tensor = y_test[i]
#             input_sentence = X_test_notencoded[i]
#             input_tensor = X_test[i]
#
#             print('////////////////////////////////////////////')
#             print('Test sample = ', input_sentence)
#             count = torch.tensor([0],dtype=torch.float32)
#             for j in range(input_tensor.size()[0]):
#                 print('input tensor[j][0] = ', input_tensor[j][0])
#
#                 output, count = model(input_tensor[j][0],count)
#                 print('count = ',count)
#                 print('output = ',output)
#
#             guess, guess_i = classFromOutput(output)
#             class_i = labels.index(class_category)
#             print('predicted class = ', guess)
#             print('actual class = ', class_category)
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
# print('test accuracy = ', test())
#
#
# # X_length = ['()()','(()(',')()(', '((()',')))(','((((', '((()))','(())((','(())(())', '((((()))']
# # y_length = ['empty','not empty','empty', 'not empty', 'empty', 'not empty', 'empty', 'not empty', 'empty', 'not empty']
# #
# #
# #
# # def encode_sentence_length(sentence):
# #     rep = torch.zeros(8, 1, n_letters)
# #     # if len(sentence) < max_length:
# #     #     for index, char in enumerate(sentence):
# #     #         pos = vocab.index(char)
# #     #         rep[index][0][pos] = 1
# #     # else:
# #     for index, char in enumerate(sentence):
# #         pos = vocab.index(char)
# #         rep[index][0][pos] = 1
# #     rep.requires_grad_(True)
# #     return rep
#
# X_length = ['()()','(()(',')()(', '((()',')))(','((((', '((()))','(())((','(())(())','((((()))','()()()()(())',')))(((()((', '((((()))))(())(())()','((()((()()((())(())(','(((((((((((((((((((())))))))))','(())(())(())((()))((()))((()))','))(()(((()(())()((())()(())()(','(((((((())))((()(()()((()()))(()((()()(()((())(()())()(()(()','))(())()((())(())()((()))((()))(())(())(())(())((']
# y_length = ['empty','not empty','empty', 'not empty', 'empty', 'not empty', 'empty', 'not empty', 'empty', 'not empty', 'empty','not empty', 'empty', 'not empty','not empty','empty', 'not empty', 'not empty', 'not empty']
#
# print(len(X_length))
# print(len(y_length))
#
# def encode_sentence_length(sentence):
#     rep = torch.zeros(60, 1, n_letters)
#     # if len(sentence) < max_length:
#     #     for index, char in enumerate(sentence):
#     #         pos = vocab.index(char)
#     #         rep[index][0][pos] = 1
#     # else:
#     for index, char in enumerate(sentence):
#         pos = vocab.index(char)
#         rep[index][0][pos] = 1
#     rep.requires_grad_(True)
#     return rep
#
# def encode_labels(label):
#     return torch.tensor([labels.index(label)], dtype=torch.float32)
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
# X_length_encoded, y_length_encoded = encode_dataset_length(X_length,y_length)
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
#     print('TEST LENGTH')
#     with torch.no_grad():
#         for i in range(num_samples):
#             class_category = y_length[i]
#             class_tensor = y_length_encoded[i]
#             input_sentence = X_length[i]
#             input_tensor = X_length_encoded[i]
#
#             print('////////////////////////////////////////////')
#             print('Test sample = ', input_sentence)
#             count = torch.tensor([0],dtype=torch.float32)
#             for j in range(input_tensor.size()[0]):
#                 print('input tensor[j][0] = ', input_tensor[j][0])
#
#                 output, count = model(input_tensor[j][0],count)
#                 print('count = ',count)
#                 print('output = ',output)
#
#             guess, guess_i = classFromOutput(output)
#             class_i = labels.index(class_category)
#             print('predicted class = ', guess)
#             print('actual class = ', class_category)
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
# print('test accuracy = ', test_length())
