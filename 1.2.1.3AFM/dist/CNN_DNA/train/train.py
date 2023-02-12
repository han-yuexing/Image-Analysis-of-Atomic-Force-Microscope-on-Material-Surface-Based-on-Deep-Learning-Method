import tensorflow as tf
import numpy as np
from PIL import Image
import random
import io
import cv2
import os

IMAGE_NUMBER = 6000
EPOCH = 5
BATCH_SIZE = 100

IMAGE_PATH = "../random_train/"
LABEL_PATH = "../code_train_dna_text.txt"

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def max_pool_4x4(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
global CHAR_SET_LEN 
CHAR_SET_LEN = 3
xs = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
ys = tf.placeholder(tf.float32, [None, 3])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

def code_cnn():
    # conv layer-1
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = weight_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # conv layer-2
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = weight_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_4x4(h_conv2)
    h_pool2 = tf.nn.dropout(h_pool2, keep_prob)

    # full connection
    W_fc1 = weight_variable([8 * 8 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # dropout
    W_fc2 = weight_variable([1024, 3])
    b_fc2 = bias_variable([3])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    return  prediction

def convert2gray(img):
    if len(img.shape) > 2:
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img
        
def text2vec(text):
    global CHAR_SET_LEN
    text_len = len(text)
    vector = np.zeros(1 * CHAR_SET_LEN)

    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k
    
    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector

def vec2text(vec):
    global CHAR_SET_LEN
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)

# create a batch
def get_next_batch(batch_size, each, images, labels):
    global CHAR_SET_LEN
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, CHAR_SET_LEN])

    def get_text_and_image(i, each):
        image_num = each * batch_size + i
        label = labels[image_num]
        image_path = images[image_num]
        captcha_image = Image.open(image_path)
        captcha_image = np.array(captcha_image)
        return label, captcha_image
    for i in range(batch_size):
        text, image = get_text_and_image(i, each)
        image = convert2gray(image)
        batch_x[i, :] = image.flatten() / 255
        batch_y[i, :] = text2vec(text)
    return batch_x, batch_y

def get_batch_dna(batch_size, images, labels):
    global CHAR_SET_LEN
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, 1 * CHAR_SET_LEN])

    def get_captcha_text_and_image(i):
        image_num = i
        label = labels[image_num]
        image_path = images[image_num]
        captcha_image = Image.open(image_path)
        captcha_image = np.array(captcha_image)
        return label, captcha_image

    for i in range(batch_size):
        text, image = get_captcha_text_and_image(i)
        image = convert2gray(image)
        batch_x[i, :] = image.flatten() / 255
        batch_y[i, :] = text2vec(text)
    return batch_x, batch_y

# compute the accuracy
def compute_accuracy(v_xs, v_ys, sess):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    #print y_pre
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    #print correct_prediction
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #print v_ys
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

prediction = code_cnn()

##
# @brief train_code_cnn 
#
# @param image_paths
# @param labels
#
# @return 
def train_code_cnn(image_paths, labels):
    global prediction
    saver = tf.train.Saver()
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
    train_step1 = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    train_step2 = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    for epoch in range(EPOCH):
        for each in range(int(IMAGE_NUMBER / BATCH_SIZE)):
            batch_x, batch_y = get_next_batch(BATCH_SIZE, each, image_paths, labels)
            _, loss_ = sess.run([train_step1, cross_entropy], feed_dict={xs: batch_x, ys: batch_y, keep_prob:0.5})
            print("epoch: %d iter: %d/%d loss: %f" %(epoch + 1, BATCH_SIZE * each, IMAGE_NUMBER, loss_))

        test_image_path = "random_true_test/"
        test_labels_path = "code_test_true_dna_text.txt"
        test_image_paths, test_labels = get_image_path_labels(test_image_path, test_labels_path, 175)
        batch_x_test, batch_y_test = get_batch_dna(175, test_image_paths, test_labels)
        accuracy_test = compute_accuracy(batch_x_test, batch_y_test, sess)
        print("test sample epoch: %d  acc: %f" % (epoch + 1, accuracy_test))

    #global CHAR_SET_LEN
    #CHAR_SET_LEN = 3
    #ys = tf.placeholder(tf.float32, [None, 3])
    true_train_image_paths, true_train_labels = get_image_path_labels("random_true_train/", "code_train_true_dna_text.txt", 320)
    #print true_train_labels
    true_train_labels[319] = true_train_labels[1]
    for epoch in range(10):
        for each in range(8):
            batch_x, batch_y = get_next_batch(40, each, true_train_image_paths, true_train_labels)
            _, loss_ = sess.run([train_step2, cross_entropy], feed_dict={xs: batch_x, ys: batch_y, keep_prob:0.5})

        test_image_path = "random_true_test/"
        test_labels_path = "code_test_true_dna_text.txt"
        test_image_paths, test_labels = get_image_path_labels(test_image_path, test_labels_path, 175)
        batch_x_test, batch_y_test = get_batch_dna(175, test_image_paths, test_labels)
        accuracy_test = compute_accuracy(batch_x_test, batch_y_test, sess)
        print("test sample epoch: %d  acc: %f" % (epoch + 1, accuracy_test))
    
    saver.save(sess, "save/model.ckpt")

def getStrContent(path):
    return io.open(path, 'r', encoding="utf-8").read()

def get_image_path_labels(IMAGE_PATH=IMAGE_PATH, LABEL_PATH=LABEL_PATH, IMAGE_NUMBER=IMAGE_NUMBER):
    image_path = IMAGE_PATH
    label_path = LABEL_PATH
    image_paths = []
    for each in range(IMAGE_NUMBER):
        image_paths.append(image_path + str(each) + ".png")
    string = getStrContent(label_path)
    labels = string.split("#")
    return image_paths, labels

def main():
    image_paths, labels = get_image_path_labels()
    train_code_cnn(image_paths, labels)

if __name__ == '__main__':
    main()
