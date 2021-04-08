#!/bin/bash

#--load_path results/16384_0_0.0001/weights/params-60000 \ ADD it to train over pretrain

declare -a filter=("16,32,32,64,64,128,128,256,256,512,1024")
declare -a enc_length=("31,31,31,31,31,31,31,31,31,31,16")
declare -a dec_length=("8,16,31,31,31,31,31,31,31,31,31")
declare -a stride=("2,2,2,2,2,2,2,2,2,2,2")
declare -a skip=("True")
declare -a file_index=("16384_new")

n_filter=${#filter[@]}

for (( idx_filter=0; idx_filter<${n_filter}; idx_filter++));
do
  echo ${filter[$idx_filter]}
  python main.py --c_name encoder --c_te_L_F ${enc_length[$idx_filter]} \
   	       --c_te_N_F ${filter[$idx_filter]} --d_te_L_F ${dec_length[$idx_filter]} \
               --c_te_str ${stride[$idx_filter]} --d_name decoder --window_size 16384 \
               --c_std 0.01 --d_std 0.01 --eta 0.0002 --data_path data/VCTK_DEMAND_16384_plain.tfrecords\
               --do_skip ${skip[$idx_filter]} \
               --preemph 0.95 --tr_iter 60000 --save_path results/${file_index[$idx_filter]}/weights --max_to_keep 1 --batch_size 100 > results/${file_index[$idx_filter]}.txt

done
