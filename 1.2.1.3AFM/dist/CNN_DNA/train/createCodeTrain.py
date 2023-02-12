import sys 
import tensorflow as tf
import os
import random

number = ['0', '1', '2']

count_0 = 160
count_1 = 80
count_2 = 80
count = {'0':0, '1':0, '2':0}

def random_number_text(char_set=number, code_size=1):
    code_text = []
    for each in range(code_size):
        while True:
            c = random.choice(char_set)
            if c == '0' and count[c] >= count_0:
                continue
            if c == '1' and count[c] >= count_1:
                continue
            if c == '2' and count[c] >= count_2:
                continue
            break

        count[c] = count[c] + 1
        code_text.append(c)
    return code_text

def write_dna_labels(size, name):
    code_list = []
    for each in range(size):
        number_list = random_number_text()
        #number_list.append(str(i))
        code = ''.join(number_list)
        code_list.append(code)
    code_text = '#'.join(code_list)
    f = open(name, 'w')
    f.write(code_text)
    f.close()

def main():
    train_size = 320
    train_label_dna = "code_train_true_dna_text.txt"
    write_dna_labels(train_size, train_label_dna)

if __name__ == '__main__':
    main()
