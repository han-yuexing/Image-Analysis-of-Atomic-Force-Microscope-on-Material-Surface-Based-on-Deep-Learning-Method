import os
import numpy as np
import cv2
import math
import io
import random

def getStrContent(path):
    string = io.open(path, 'r', encoding="utf-8").read()
    labels = string.split("#")
    return labels;

def getImages(dir_name):
    images = []
    for filename in os.listdir(dir_name):
        file_true_name = dir_name + '/' + filename
        image = cv2.imread(file_true_name, 0)
        images.append(image)
    return images

def main():
    labels = getStrContent("code_train_true_dna_text.txt")
    #dirs = ["parallel-anti_train", "and_train", "cross_train", "others"]
    dirs = ["parallel-anti_train", "and_train", "cross_train"]
    #images = [getImages(dirs[0]), getImages(dirs[1]), getImages(dirs[2]), getImages(dirs[3])]
    images = [getImages(dirs[0]), getImages(dirs[1]), getImages(dirs[2])]
    #count = [0, 0, 0, 0]
    count = [0, 0, 0]
    dir_out = "random_true_train"
    for i in range(320):
        dir_name = labels[i]
        image = images[int(dir_name)][count[int(dir_name)]]
        count[int(dir_name)] = count[int(dir_name)] + 1
        file_random_name = dir_out + '/' + str(i) + '.png'
        cv2.imwrite(file_random_name, image)
    
if __name__ == '__main__':
    main()

