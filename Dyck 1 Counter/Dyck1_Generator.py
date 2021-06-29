import pandas as pd
import sklearn
from sklearn.utils import shuffle

class Dyck1_Generator(object):
    def generateParenthesis(self, n):
        def generate(A = []):
            if len(A) == 2*n:
                if valid(A):
                    val.append("".join(A))
                else:
                    inval.append("".join(A))
            else:
                A.append('(')
                generate(A)
                A.pop()
                A.append(')')
                generate(A)
                A.pop()

        def valid(A):
            bal = 0
            for c in A:
                if c == '(': bal += 1
                else: bal -= 1
                if bal < 0: return False
            return bal == 0

        val = []
        inval = []
        generate()
        return val, inval

def generateDataset(n_bracket_pairs):
    gen = Dyck1_Generator()
    # d1_valid, d1_invalid = gen.generateParenthesis(3)
    d1_valid = []
    d1_invalid = []
    for i in range(1,n_bracket_pairs+1):
        x,y = gen.generateParenthesis(i)
        for elem in x:
            d1_valid.append(elem)
        for elem in y:
            d1_invalid.append(elem)
    return d1_valid,d1_invalid

# d1_valid, d1_invalid = generateDataset(20)


# print(d1_valid)
# print('///////////////////////')
# print(d1_invalid)

# add the labels and then create the csv file to complete the dataset

def generateLabelledDataset(n_bracket_pairs):
    d1_valid, d1_invalid = generateDataset(n_bracket_pairs)
    dataset = []
    for elem in d1_valid:
        entry = (elem, 'valid')
        dataset.append(entry)
    for elem in d1_invalid:
        entry = (elem,'invalid')
        dataset.append(entry)
    return dataset

# # dataset = generateLabelledDataset(6)
# dataset = generateLabelledDataset(6)
# print(dataset)
#
# dataset_ = pd.DataFrame(dataset,columns=['sentence','label'])
# print(dataset_.head())
#
# dataset_ = sklearn.utils.shuffle(dataset_).reset_index(drop=True)
# print(dataset_.head())
# print(len(dataset_))
#
#
# # dataset_=sklearn.utils.shuffle(dataset_)
# # print(dataset_.head())
#
# dataset3=generateLabelledDataset(3)
# print(dataset3)
# print(len(dataset3))
#
# dataset_.to_csv('Dyck1_Dataset_20pairs.csv',index=False)

gen = Dyck1_Generator()
print(gen.generateParenthesis(2))