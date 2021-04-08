from __future__ import print_function
import tensorflow as tf
import numpy as np
from func import prelu_func, pre_emph, pre_emph_test, de_emph_test
import os
import scipy.io.wavfile as wavfile
from model import AE


#### PARAMETERS
flags = tf.app.flags

flags.DEFINE_string("c_name", 'encoder', "name for encoder variables")
flags.DEFINE_integer("batch_size", 100, "Batch size")
flags.DEFINE_integer("window_size", 2**14, "Input length")
flags.DEFINE_string("c_te_L_F", None, "List of filters length")
flags.DEFINE_string("d_te_L_F", None, "List of filters length")
flags.DEFINE_string("c_te_N_F", None, "List of filters Number")
flags.DEFINE_string("c_te_str", None, "List of filters stride")
flags.DEFINE_string("d_name", 'decoder', "name for decoder variables")
flags.DEFINE_boolean("do_skip", True, "Do skip connection or not")
flags.DEFINE_float("c_std", 0.01, "Standard deviation for initial encoder")
flags.DEFINE_float("d_std", 0.01, "Standard deviation for initial decoder")
flags.DEFINE_float("eta", 0.0002, "Learning rate")
flags.DEFINE_string("data_path", None, "TFRecord path")
flags.DEFINE_float("preemph", 0.95, "Preemphasis coefficient")
flags.DEFINE_integer("tr_iter", 602, "Training iteration, original full data = 60200")
flags.DEFINE_string("save_path", None, "Save model path")
flags.DEFINE_string("load_path", None, "Load model path")
flags.DEFINE_integer("max_to_keep", 2, "Maximum number of save model")
flags.DEFINE_boolean("Inference", False, "Training or Inference")
flags.DEFINE_string("inf_path", None, "Inference wav directory")
flags.DEFINE_string("inf_save_path", None, "Inference save directory")

FLAGS = flags.FLAGS

FLAGS.c_te_L_F = list(np.int0(np.array(FLAGS.c_te_L_F.split(','))))
FLAGS.d_te_L_F = list(np.int0(np.array(FLAGS.d_te_L_F.split(','))))
FLAGS.c_te_N_F = list(np.int0(np.array(FLAGS.c_te_N_F.split(','))))
FLAGS.c_te_str = list(np.int0(np.array(FLAGS.c_te_str.split(','))))


#### PARAMETERS

with tf.compat.v1.Session() as sess:
###### BUILD MODEL ####
  G_instance = AE(FLAGS, sess)
  G_instance.build_model()
  if FLAGS.Inference == False:
###### TRAIN ######
    G_instance.train_AE()
  else:
    if not os.path.exists(FLAGS.inf_save_path):
        os.makedirs(FLAGS.inf_save_path)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, FLAGS.load_path)
    print ('Read ', FLAGS.load_path)
    # get into test directory and open for loop
    inf_files = [os.path.join(FLAGS.inf_path, wav) for wav in
                   os.listdir(FLAGS.inf_path) if wav.endswith('.wav')]

    for i in range(len(inf_files)):
        fm, inf_wav = wavfile.read(inf_files[i])       
        inf_name = inf_files[i].split('/')[-1]
        print (inf_name,'WAVE LENGTH', len(inf_wav)/float(FLAGS.window_size))
########## SLICE DATA ##############
# SLICE FILES INTO (total chunks, window size)

        # DATA LENGTH IS EVEN FOR WINDOWSIZE
        if int(len(inf_wav)/FLAGS.window_size) == len(inf_wav)/float(FLAGS.window_size):
            print ('EVEN WINDOW')
            inf_slice = int(len(inf_wav)/FLAGS.window_size)
            for j in range(inf_slice):
                start_slice = j*FLAGS.window_size
                end_slice = FLAGS.window_size + j*FLAGS.window_size
                inf_sliced = inf_wav[start_slice:end_slice]
                inf_sliced = inf_sliced.reshape([1,FLAGS.window_size])
                if j == 0:
                    inf_batches = inf_sliced
                else:
                    inf_batches = np.append(inf_batches, inf_sliced, axis=0)

        # DATA LENGTH IS ODD FOR WINDOWSIZE
        else:
            print ('ODD WINDOW')
            inf_slice = int(len(inf_wav)/FLAGS.window_size) + 1
            for j in range(inf_slice):
                if j == inf_slice-1:
                    start_slice = j*FLAGS.window_size
                    n_pad = FLAGS.window_size-(len(inf_wav)-start_slice)
                    end_slice = len(inf_wav)
                    pad = np.zeros(n_pad)
                    pad = np.array(pad, dtype=np.int16)
                    inf_sliced = inf_wav[start_slice:end_slice]
                    inf_sliced = np.append(inf_sliced,pad)
                else:
                    start_slice = j*FLAGS.window_size
                    end_slice = FLAGS.window_size + j*FLAGS.window_size
                    inf_sliced = inf_wav[start_slice:end_slice]
                inf_sliced = inf_sliced.reshape([1,FLAGS.window_size])
                if j == 0:
                    inf_batches = inf_sliced
                else:
                    inf_batches = np.append(inf_batches, inf_sliced, axis=0)
        print (inf_batches.shape)
########## SLICE DATA ##############


################################ SPLIT AND PROPAGATE OR JUST PROPAGATE #########################################
        if inf_batches.shape[0] >= FLAGS.batch_size:
            print ('bigger')
        # SPLIT BATHCES
            # EVEN
            if inf_batches.shape[0]%FLAGS.batch_size ==0:
                print ("EVEN BATCHES")
                for j in range(int(inf_batches.shape[0]/FLAGS.batch_size)):
                    inf_input = inf_batches[j*FLAGS.batch_size:(j+1)*FLAGS.batch_size]
                    print ('split batches', inf_input.shape)
                    
                    # NORMALIZE, PREEMPH AND PROPAGATE OVER N SPLITS
                    inf_input = pre_emph_test(FLAGS.preemph, inf_input)
                    print ('Normalized and preemph', inf_input.shape, type(inf_input[0][0]))
                    inf_result = G_instance.inference_AE(inf_input)
                    print ('PROPAGATE', inf_result.shape)
                    inf_result = de_emph_test(FLAGS.preemph, inf_result)
                    print ('deemph', inf_result.shape)
                    if j ==0:
                        inf_result_append = inf_result
                    else:
                        inf_result_append = np.append(inf_result_append,inf_result, axis = 0)
                inf_result = inf_result_append
                inf_result = inf_result.reshape(inf_result.shape[0]*inf_result.shape[1])
                inf_result = np.delete(inf_result, np.s_[inf_result.shape[0]-n_pad:inf_result.shape[0]], axis=0)
                print ('save shape', inf_result.shape)
                wavfile.write(os.path.join(FLAGS.inf_save_path, inf_name), 16e3, inf_result)
                print ("")
            
            # ODD
            else:
                print ("ODD BATCHES")
                for j in range(int(inf_batches.shape[0]/FLAGS.batch_size)+1):
                    if j == inf_batches.shape[0]/FLAGS.batch_size:
                        inf_input = inf_batches[j*FLAGS.batch_size:inf_batches.shape[0]]
                    else:
                        inf_input = inf_batches[j*FLAGS.batch_size:(j+1)*FLAGS.batch_size]
                    print ('split batches', inf_input.shape)

                    # NORMALIZE, PREEMPH AND PROPAGATE OVER N SPLITS
                    inf_input = pre_emph_test(FLAGS.preemph, inf_input)
                    print ('Normalized and preemph', inf_input.shape, type(inf_input[0][0]))
                    inf_result = G_instance.inference_AE(inf_input)
                    print ('PROPAGATE', inf_result.shape)
                    inf_result = de_emph_test(FLAGS.preemph, inf_result)
                    print ('deemph', inf_result.shape)
                    if j ==0:
                        inf_result_append = inf_result
                    else:
                        inf_result_append = np.append(inf_result_append,inf_result, axis = 0)
                inf_result = inf_result_append
                inf_result = inf_result.reshape(inf_result.shape[0]*inf_result.shape[1])
                inf_result = np.delete(inf_result, np.s_[inf_result.shape[0]-n_pad:inf_result.shape[0]], axis=0)
                print ('save shape', inf_result.shape)
                wavfile.write(os.path.join(FLAGS.inf_save_path, inf_name), 16000, inf_result)
                print ("")


        if inf_batches.shape[0] < FLAGS.batch_size:
            print ('smaller')
            inf_input = inf_batches
            print ('split batches', inf_input.shape)
            # NORMALIZE, PREEMPH AND PROPAGATE OVER N SPLITS
            inf_input = pre_emph_test(FLAGS.preemph, inf_input)
            print ('Normalized and preemph', inf_input.shape, type(inf_input[0][0]))
            inf_result = G_instance.inference_AE(inf_input)
            print ('PROPAGATE', inf_result.shape)
            inf_result = de_emph_test(FLAGS.preemph, inf_result)
            print ('deemph', inf_result.shape)
            inf_result = inf_result.reshape(inf_result.shape[0]*inf_result.shape[1])
            inf_result = np.delete(inf_result, np.s_[inf_result.shape[0]-n_pad:inf_result.shape[0]], axis=0)
            print ('save shape', inf_result.shape)
            wavfile.write(os.path.join(FLAGS.inf_save_path, inf_name), 16000, inf_result)
            print ("")

################################ SPLIT AND PROPAGATE OR JUST PROPAGATE #########################################

