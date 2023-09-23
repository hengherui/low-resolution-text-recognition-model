#!/usr/bin/env bash
#
CUDA_VISIBLE_DEVICES=5,6 python main.py \
  --synthetic_train_data_dir /data/paper_data/NIPS2014/NIPS2014 /data/paper_data/CVPR2016/ \
  --test_data_dir /data/paper_data/test/IIIT5K_3000/ \
  --batch_size 1024 \
  --workers 0 \
  --epochs 300 \
  --height 64 \
  --width 256 \
  --voc_type ALLCASES_SYMBOLS \
  --arch ResNet_ASTER \
  --with_lstm \
  --logs_dir logs/baseline_aster \
  --real_logs_dir /data/mkyang/logs/recognition/aster.pytorch \
  --max_len 100 \
  --STN_ON \
  --tps_inputsize 32 64 \
  --tps_outputsize 32 100 \
  --tps_margins 0.05 0.05 \
  --stn_activation none \
  --num_control_points 20 \
  --resume /data/aster.pytorch-master/aster.pytorch-master/model_best.pth.tar
