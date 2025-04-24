#!/bin/sh

export PYTHONPATH=./
# eval "$(conda shell.bash hook)"
PYTHON=python

TRAIN_CODE=train.py

dataset=$1
exp_name=$2
exp_dir=exp/${dataset}/train/${exp_name}
model_dir=${exp_dir}/model
config=$3

mkdir -p ${model_dir}
cp tool/train.sh model/BGPSeg.py tool/${TRAIN_CODE} ${config} ${exp_dir}


$PYTHON ${exp_dir}/${TRAIN_CODE} \
  --config=${config} \
  save_path ${exp_dir}
