import cv2
import os
import numpy as np
import math
import time
from PIL import Image

dirs = ["parallel-anti", "and", "cross", "others"]
#dirs = ["parallel-anti", "and", "cross"]
dir_name = "random_true_test"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
count = 0
code_list = []
for dir_single in dirs:
    for filename in os.listdir(dir_single):
        file_true_name = dir_single + '/' + filename
        if not os.path.isdir(file_true_name):
            if dir_single == "cross" :
                code = '2'
            elif dir_single == 'parallel-anti':
                code = '0'
            #elif dir_single == 'and' :
                #code = '1'
            else:
                code = '1'
            code_list.append(code)
            image = cv2.imread(file_true_name, 0)
            file_out_name = dir_name + '/' + str(count) + '.png'
            cv2.imwrite(file_out_name, image)
            count = count + 1

code_text = '#'.join(code_list)
f = open("code_test_true_dna_text.txt", 'w')
f.write(code_text)
f.close()
