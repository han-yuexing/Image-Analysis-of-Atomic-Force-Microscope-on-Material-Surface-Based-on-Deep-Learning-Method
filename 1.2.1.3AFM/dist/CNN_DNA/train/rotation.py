import tensorflow as tf
import cv2
import os
from PIL import Image
from math import pi

dirs = ['and_train', 'parallel-anti_train', 'cross_train', 'others_train']
count = 300
for dir_name in dirs:
    for filename in os.listdir(dir_name):
        file_true_name = dir_name + '/' + filename
        img = Image.open(file_true_name)
        rotated_imgs = img.transpose(Image.FLIP_TOP_BOTTOM)
        file_out_name = dir_name + '/' + str(count) + '.png'
        rotated_imgs.save(file_out_name)
        count = count + 1

count = 500
for dir_name in dirs:
    for filename in os.listdir(dir_name):
        file_true_name = dir_name + '/' + filename
        img = Image.open(file_true_name)
        rotated_imgs = img.transpose(Image.FLIP_LEFT_RIGHT)
        file_out_name = dir_name + '/' + str(count) + '.png'
        rotated_imgs.save(file_out_name)
        count = count + 1
