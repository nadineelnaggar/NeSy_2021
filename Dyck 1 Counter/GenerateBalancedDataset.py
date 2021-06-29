import random
import pandas as pd
import sklearn


data = []
X = []
y = []

with open("Dyck1_Dataset_6pairs.txt",'r') as f:
    for line in f:
        line = line.split(",")
        sentence = line[0].strip()
        label = line[1].strip()
        X.append(sentence)
        y.append(label)
        data.append((sentence,label))

print(data)

count_valid = 0
count_invalid = 0

valid_data = []
invalid_data = []

for item in data:
    if item[1]=='valid':
        count_valid+=1
        valid_data.append(item)
    else:
        count_invalid+=1
        invalid_data.append(item)
print('number of valid elements = ',count_valid)
print(len(valid_data))
print('number of invalid elements = ',count_invalid)
print(len(invalid_data))
print(len(data))
print(count_invalid+count_valid)

while count_invalid>count_valid:
    idx = random.randint(0,count_valid-1)
    valid_data.append(valid_data[idx])
    count_valid+=1

print(len(valid_data))
print(count_valid)

data_balanced = []
for elem in valid_data:
    data_balanced.append(elem)
for elem in invalid_data:
    data_balanced.append(elem)
# data_balanced.append(valid_data)
# data_balanced.append(invalid_data)
print(data_balanced)
print(len(data_balanced))
print(count_invalid*2)

random.shuffle(data_balanced)
print(data_balanced)

dataset_ = pd.DataFrame(data_balanced,columns=['sentence','label'])
print(dataset_.head())

dataset_ = sklearn.utils.shuffle(dataset_).reset_index(drop=True)
print(dataset_.head())
print(len(dataset_))
# dataset_=sklearn.utils.shuffle(dataset_)
# print(dataset_.head())

# dataset3=generateLabelledDataset(3)
# print(dataset3)
# print(len(dataset3))

dataset_.to_csv('Dyck1_Dataset_6pairs_balanced.csv',index=False)