#!/bin/bash
declare -a filter=("16,32,32,64,64,128,128,256,256,512,1024" )
declare -a enc_length=("31,31,31,31,31,31,31,31,31,31,16")
declare -a dec_length=("8,16,31,31,31,31,31,31,31,31,31" )
declare -a stride=("2,2,2,2,2,2,2,2,2,2,2")
declare -a skip=("True")
declare -a directory=("16384")
declare -a param=("60000")


n_filter=${#filter[@]}

for (( idx_filter=0; idx_filter<${n_filter}; idx_filter++));
do
  echo ${filter[$idx_filter]}
  python main.py --c_te_L_F ${enc_length[$idx_filter]} \
       	         --c_te_N_F ${filter[$idx_filter]} --d_te_L_F ${dec_length[$idx_filter]} \
                 --c_te_str ${stride[$idx_filter]} --window_size 16384 \
                 --data_path sample.tfrecord \
                 --save_path save/weights --Inference True \
                 --max_to_keep 1 --inf_path test_16bit --inf_save_path inferences/${directory[$idx_filter]} \
                 --load_path results/${directory[$idx_filter]}/weights/params-${param[$idx_filter]} --tr_iter 5 \
                 --preemph 0.95 --max_to_keep 1 --batch_size 100

done
