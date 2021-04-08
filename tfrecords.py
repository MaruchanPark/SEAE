import tensorflow as tf
import numpy as np
import os
import scipy.io.wavfile as wf

def _bytes_feature(val):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[val]))
def norm(inp):
    return (2./65535.) * (inp  - 32767.) + 1.

def get_num_chunk(dataset_path,clean_path,noisy_path,strd):
    clean_data_list = os.listdir(dataset_path+clean_path)
    noisy_data_list = os.listdir(dataset_path+noisy_path)

    total_chunk = 0
    for i in clean_data_list:
        SR_c,clean_datas = wf.read(dataset_path+clean_path+i)
        SR_n,noisy_datas = wf.read(dataset_path+noisy_path+i)
        
        if SR_c != SR_n:
            raise ValueError(i+' : SR ERROR')
        if SR_c != 16000:
            raise ValueError(i+' : SR ERROR')
        if len(clean_datas) != len(noisy_datas):
            raise ValueError(i+' : DATA LENGTH ERROR')

        d_len = len(clean_datas)
        if int(d_len/strd) == float(d_len/strd):
            num_slice=int(d_len/strd)-1
        else :
            num_slice=int(d_len/strd)
        total_chunk = total_chunk+num_slice
    return total_chunk,clean_data_list

def write_TFR(out_file_name,totalchunk,clean_data_list,dataset_path,clean_path,noisy_path,window,strd):
    creat_TFR = tf.io.TFRecordWriter(out_file_name)
    indx = 0
    for i in clean_data_list:
        total_len = len(clean_data_list)
        if indx%100 == 1:
            print(indx,'/',total_len)
        SR_c,clean_datas = wf.read(dataset_path+clean_path+i)
        clean_datas = norm(clean_datas).astype(np.float32)
        SR_n,noisy_datas = wf.read(dataset_path+noisy_path+i)
        noisy_datas = norm(noisy_datas).astype(np.float32)
        
        d_len = len(clean_datas)
        if int(d_len/strd) == float(d_len/strd):
            num_slice=int(d_len/strd)-1
        else :
            num_slice=int(d_len/strd)

        for j in range(num_slice):
            if j != num_slice-1:
                clean_chunk = clean_datas[j*strd:j*strd+window]
                noisy_chunk = noisy_datas[j*strd:j*strd+window]

                clean_chunk = clean_chunk.tostring()
                noisy_chunk = noisy_chunk.tostring()
            else : 
                chunk_start = j*strd
                tail = d_len-chunk_start
                n_pad =window - tail
                pad = np.zeros(n_pad).astype(np.float32)
                chunk_end = d_len

                clean_chunk = np.append(clean_datas[j*strd:d_len],pad)
                noisy_chunk = np.append(noisy_datas[j*strd:d_len],pad)

                clean_chunk = clean_chunk.tostring()
                noisy_chunk = noisy_chunk.tostring()
            features = tf.train.Features(feature={'clean_data' : _bytes_feature(clean_chunk),
                                               'noisy_data' : _bytes_feature(noisy_chunk)})
            example = tf.train.Example(features = features)
            creat_TFR.write(example.SerializeToString())
        indx = indx+1
    creat_TFR.close()
    print('DONE')

dataset_path = 'data/'
clean_path = 'clean_trainset_wav_16k/'
noisy_path = 'noisy_trainset_wav_16k/'
save_path = dataset_path

out_file_name = dataset_path+'VCTK_DEMAND_16384_plain.tfrecords'

window = 16384
strd = int(window/2)
total_chunk,clean_data_list = get_num_chunk(dataset_path,clean_path,noisy_path,strd)

write_TFR(out_file_name,
          total_chunk,
          clean_data_list,
          dataset_path,
          clean_path,
          noisy_path,
          window,
          strd)


