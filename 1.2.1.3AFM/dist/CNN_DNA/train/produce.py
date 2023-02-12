import numpy as np
from PIL import Image
import random
import io
import cv2
import os

num_step_and = 20
hash_map_0 = {}
hash_map_1 = {}
hash_map_2 = {}
hash_map_3 = {}

images_train_0 = []
images_train_1 = []
images_train_2 = []
images_train_3 = []

def getImages(dir_name):
    images = []
    for filename in os.listdir(dir_name):
        file_true_name = dir_name + '/' + filename
        image = cv2.imread(file_true_name)
        images.append(image)
    return images

dirs = ['parallel-anti_source', 'and_source', 'cross_source', 'others_source'] 
images = [getImages(dirs[0]), getImages(dirs[1]), getImages(dirs[2]), getImages(dirs[3])]

num_step_parallel = len(images[0]) / len(images[1]) * num_step_and
num_step_cross = len(images[2]) / len(images[1]) * num_step_and
num_step_cross = num_step_and

count = 0
while len(hash_map_0) != num_step_parallel:
    index = random.randint(0, len(images[0]) - 1)
    if not hash_map_0.has_key(index):
        hash_map_0[index] = 1
        images_train_0.append(images[0][index])
        file_out_name = 'parallel-anti_train/' + str(count) + '.png'
        cv2.imwrite(file_out_name, images[0][index])
        count = count + 1

count = 0
for key in range(len(images[0])):
    if not hash_map_0.has_key(key):
        file_out_name = 'parallel-anti/' + str(count) + '.png'
        cv2.imwrite(file_out_name, images[0][key])
        count = count + 1

count = 0
while len(hash_map_1) != num_step_and:
    index = random.randint(0, len(images[1]) - 1)
    if not hash_map_1.has_key(index):
        hash_map_1[index] = 1
        images_train_1.append(images[1][index])
        file_out_name = 'and_train/' + str(count) + '.png'
        cv2.imwrite(file_out_name, images[1][index])
        count = count + 1

count = 0
for key in range(len(images[1])):
    if not hash_map_1.has_key(key):
        file_out_name = 'and/' + str(count) + '.png'
        cv2.imwrite(file_out_name, images[1][key])
        count = count + 1

count = 0
while len(hash_map_2) != num_step_cross:
    index = random.randint(0, len(images[2]) - 1)
    if not hash_map_2.has_key(index):
        hash_map_2[index] = 1
        images_train_2.append(images[2][index]) 
        file_out_name = 'cross_train/' + str(count) + '.png'
        cv2.imwrite(file_out_name, images[2][index])
        count = count + 1

count = 0
for key in range(len(images[2])):
    if not hash_map_1.has_key(key):
        file_out_name = 'cross/' + str(count) + '.png'
        cv2.imwrite(file_out_name, images[2][key])
        count = count + 1

#count = 0
#while len(hash_map_3) != num_step:
    #index = random.randint(0, len(images[3]) - 1)
    #if not hash_map_3.has_key(index):
        #hash_map_3[index] = 1
        #images_train_3.append(images[3][index]) 
        #file_out_name = 'others_train/' + str(count) + '.png'
        #cv2.imwrite(file_out_name, images[3][index])
        #count = count + 1

#count = 0
#for key in range(len(images[3])):
    #if not hash_map_1.has_key(key):
        #file_out_name = 'others/' + str(count) + '.png'
        #cv2.imwrite(file_out_name, images[3][key])
        #count = count + 1
