import tensorflow as tf
import numpy as np

def prelu_func(x,alpha):
    pos = tf.nn.relu(x)
    neg = alpha*(x-tf.abs(x))*0.5
    return pos+neg

def pre_emph(x, coeff=0.95):
    x0 = tf.reshape(x[0], [1,])
    diff = x[1:] - coeff * x[:-1]
    concat = tf.concat([x0, diff], 0)
    return concat

def pre_emph_test(coeff, test_input):
    test_input = np.array(test_input, dtype = np.float32)
    test_input = (2./65535.) * (test_input - 32767.) + 1
    if coeff>0:
        for k in range(test_input.shape[0]):
            test_input0 = np.reshape(test_input[k][0], [1,])
            test_input_diff = test_input[k][1:] - coeff * test_input[k][:-1]
            test_input[k] = np.concatenate((test_input0, test_input_diff), axis=0)
    return test_input

def de_emph_test(coeff, de_emph_input):
    if coeff>0:
        de_emph_input_x = np.zeros(de_emph_input.shape, dtype = np.float32)
        for k in range(de_emph_input.shape[0]):
            de_emph_input_x[k][0] = de_emph_input[k][0]
            for l in range(1, de_emph_input.shape[1], 1):
                de_emph_input_x[k][l] = coeff * de_emph_input_x[k][l-1] + de_emph_input[k][l]
    return de_emph_input_x
