#!/usr/bin/env bash
array=("svt_p_645")

DLS_DATA_URL="/data/paper_data/test/"
DLS_TRAIN_URL=""
for i in "${array[@]}"
do
  echo $i
  CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --synthetic_train_data_dir xxx \
    --test_data_dir ${DLS_DATA_URL}$i \
    --batch_size 1 \
    --workers 8 \
    --height 64 \
    --width 256 \
    --voc_type ALLCASES_SYMBOLS \
    --arch ResNet_ASTER \
    --with_lstm \
    --logs_dir logs/baseline_aster \
    --real_logs_dir xxx \
    --max_len 100 \
    --evaluate \
    --STN_ON \
    --beam_width 5 \
    --tps_inputsize 32 64 \
    --tps_outputsize 32 100 \
    --tps_margins 0.05 0.05 \
    --stn_activation none \
    --num_control_points 20 \
    --resume /data/aster.pytorch-master/aster.pytorch-master/model_best.pth.tar
done