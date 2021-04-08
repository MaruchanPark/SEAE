from __future__ import print_function
import tensorflow as tf
import numpy as np
from func import prelu_func, pre_emph
import time
import os

class AE:
    def __init__(self, args, sess):
        self.c_name = args.c_name
        self.batch_size = args.batch_size
        self.window_size = args.window_size
        self.c_te_L_F = args.c_te_L_F
        self.d_te_L_F = args.d_te_L_F
        self.c_te_N_F = args.c_te_N_F
        self.c_te_str = args.c_te_str
        self.d_name = args.d_name
        self.do_skip = args.do_skip
        self.c_std = args.c_std
        self.d_std = args.d_std
        self.eta = args.eta
        self.data_path = args.data_path
        self.preemph = args.preemph
        self.sess = sess
        self.tr_iter = args.tr_iter
        self.save_path = args.save_path
        self.load_path = args.load_path
        self.max_to_keep = args.max_to_keep

    def conv_1d(self,c_IN, c_L_F,c_N_F, c_str, c_idx, name='conv', reuse = False, c_std= 0.01):
        ## INPUT : x(1d), W(Filter length, Width, Input channel, Output channel),
        ##         b(Output channel),
        ## INPUT 1d SHAPE : [BATCH, LENGTH, CHANNEL]
        ## INPUT 2d SHAPE : [BATCH, LENGTH, WIDTH ,CHANNEL]
        
        # Reshape to 2d
        c_IN = tf.expand_dims(c_IN,2)
        # Convolution
        with tf.compat.v1.variable_scope(name+ '_' + str(c_idx), reuse = reuse):
            init = tf.truncated_normal_initializer(stddev = c_std)
            W = tf.compat.v1.get_variable('W', [c_L_F, c_IN.get_shape()[2].value,
                                      c_IN.get_shape()[-1].value, c_N_F],
                                initializer = init)
            b = tf.compat.v1.get_variable('b', [c_N_F],initializer = init)
            alpha = tf.compat.v1.get_variable('alpha', [c_N_F], initializer = init)

            c_logits = tf.nn.conv2d(c_IN, W, strides = [1,c_str,1,1], padding='SAME')
            c_logits = tf.nn.bias_add(c_logits,b)
            c_logits = tf.squeeze(c_logits, axis = 2)
            c_act = prelu_func(c_logits,alpha)
            return c_act, c_logits
        
    def deconv_1d(self,d_IN, d_L_F, d_N_F, d_str, d_idx, name='deconv', reuse = False, d_std = 0.01):
        # x-dim = [batch, windowsize, channel]
        # 2d
        d_IN = tf.expand_dims(d_IN,2)
        o2 = d_IN.get_shape().as_list()

        o2[1] = o2[1] * d_str
        o2[-1] = d_N_F
        o2[0] = -1
        o2[0] = self.batch_size
        with tf.compat.v1.variable_scope(name+ '_' + str(d_idx), reuse = reuse):
            init = tf.truncated_normal_initializer(stddev = d_std)
            W = tf.compat.v1.get_variable('W', [d_L_F, d_IN.get_shape()[2].value,
                                      d_N_F, d_IN.get_shape()[-1].value],
                                initializer = init)
            b = tf.compat.v1.get_variable('b', [d_N_F],initializer = init)
            alpha = tf.compat.v1.get_variable('alpha', [d_N_F], initializer = init)
            
            d_logits = tf.nn.conv2d_transpose(d_IN, W, output_shape = o2,
                                            strides = [1,d_str,1,1])
            d_logits = tf.nn.bias_add(d_logits,b)
            d_logits = tf.squeeze(d_logits, axis = 2)
            d_act = prelu_func(d_logits,alpha)
            return d_act, d_logits

    def propagate_AE(self, noisy_input,reuse):
        skip = []
        index = 0
        c_te_IN = noisy_input

        for i in range(len(self.c_te_L_F)):
            c_te_IN, c_te_logits = self.conv_1d(c_te_IN, self.c_te_L_F[i],
                                                     self.c_te_N_F[i], self.c_te_str[i],
                                                     index,name = self.c_name,
                                                     reuse=reuse, c_std = self.c_std)
            if self.do_skip == True:
                if i != len(self.c_te_L_F)-1:
                    skip.append(c_te_logits)
            index = index + 1
            if reuse == False:
                print (c_te_IN)
            
        index = 0
        d_te_IN = c_te_IN
        d_te_L_F = self.d_te_L_F
        d_te_N_F = self.c_te_N_F[:-1][::-1]+[1]
        d_te_str = self.c_te_str[::-1]
        if self.do_skip == True:
            skip = skip[::-1]
        for i in range(len(d_te_L_F)):
            d_te_IN, d_te_logits = self.deconv_1d(d_te_IN, d_te_L_F[i], d_te_N_F[i],
                                             d_te_str[i], index, name=self.d_name,
                                             reuse = reuse, d_std = self.d_std)
            if self.do_skip == True:
                if i != len(d_te_L_F)-1:
                    d_te_IN = tf.concat([d_te_IN, skip[i]],2)
            index = index + 1
            if reuse == False:
                print (d_te_IN)
        if reuse == False:
            var = tf.compat.v1.trainable_variables()
            for i in range(len(var)):
                if 'W' in var[i].name:
                    print (var[i].name,'\n' ,'FILTER SHAPE :',var[i].get_shape().as_list())
        return d_te_IN, d_te_logits

    def build_model(self):
        def from_TFR(serialized):
            features = tf.io.parse_single_example(serialized=serialized,
                    features = {'clean_data' : tf.io.FixedLenFeature([],tf.string),
                               'noisy_data' : tf.io.FixedLenFeature([],tf.string)})
            clean = features["clean_data"]
            noisy = features["noisy_data"]
            clean = tf.decode_raw(clean,tf.float32)
            noisy = tf.decode_raw(noisy,tf.float32)
            clean = tf.reshape(clean,[16384])
            noisy = tf.reshape(noisy,[16384])
            clean = tf.cast(pre_emph(clean, self.preemph), tf.float32)
            noisy = tf.cast(pre_emph(noisy, self.preemph), tf.float32)
            return clean, noisy

        dataset = tf.data.TFRecordDataset(self.data_path).map(from_TFR)

        dataset = dataset.repeat()
        dataset = dataset.shuffle(self.batch_size*100)
        dataset = dataset.batch(self.batch_size)
        self.iterator = tf.compat.v1.data.make_initializable_iterator(dataset)

        clean_batch, self.noisy_batch = self.iterator.get_next()

        clean_batch = tf.expand_dims(clean_batch, -1)
        noisy_batch = tf.expand_dims(self.noisy_batch, -1)

        G_dummy, G_dummy_logits = self.propagate_AE(noisy_batch,False)
        non, G_logits = self.propagate_AE(noisy_batch,True)
        self.G = tf.tanh(G_logits)
        self.LOSS = tf.reduce_mean(tf.abs(tf.subtract(self.G, clean_batch)))
        self.train = tf.compat.v1.train.RMSPropOptimizer(self.eta).minimize(self.LOSS)

    def train_AE(self):
        ### tfDATA
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(self.iterator.initializer)
        print ('\nInitialize variables')
        
        if self.load_path != None:
            saver = tf.train.Saver()
            saver.restore(self.sess, self.load_path)
            print ('Read ', self.load_path)
        else:
            print ('Training from random parameters\n')

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        if not os.path.exists(self.save_path.split('/')[0]):
            os.makedirs(self.save_path.split('/')[0])
        
        for i in range(1,self.tr_iter+1):
            start = time.time()
            _, check_loss = self.sess.run([self.train, self.LOSS])
            end = time.time()

            log_step = int(self.tr_iter / 10000)
            if self.tr_iter < 10000:
                log_step = 5

            if i% log_step == 0:
                with open(self.save_path + '/loss.txt', 'ab') as file:
                    np.savetxt(file, [check_loss])
                with open(self.save_path + '/time.txt', 'at') as file:
                    file.write("%i/%i , %f \n" % (i, self.tr_iter, end-start))

            if i % (self.tr_iter/self.max_to_keep) == 0 or i == self.tr_iter:
                saver = tf.compat.v1.train.Saver(max_to_keep = self.max_to_keep)
                saver.save(self.sess, self.save_path+'/params',global_step=i)
                print ('Save model at', self.save_path)
            
        print ('Training done')
    def inference_AE(self, inf_in):
        idx_pad = False
        if inf_in.shape[0] < self.batch_size:
            n_pad = self.batch_size - inf_in.shape[0]
            pad = np.zeros([n_pad, self.window_size], dtype=np.float32)
            inf_in = np.append(inf_in, pad, axis = 0)
            idx_pad = True

        fdict = {self.noisy_batch : inf_in}
        cleaned = self.sess.run(self.G,
                                feed_dict = fdict)
        if idx_pad == True:
            cleaned = cleaned[:self.batch_size-n_pad]
        return cleaned

