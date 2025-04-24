#!/bin/sh

export PYTHONPATH=./
# eval "$(conda shell.bash hook)"
PYTHON=python

TEST_CODE=test.py

dataset=$1
exp_name=$2
exp_dir=exp/${dataset}/test/${exp_name}
model_dir=${exp_dir}/model
predictions_dir=${exp_dir}/predictions
config=$3

mkdir -p ${model_dir}
mkdir -p ${predictions_dir}
cp tool/test.sh model/BGPSeg.py tool/${TEST_CODE} ${config} ${exp_dir}


$PYTHON ${exp_dir}/${TEST_CODE} \
  --config=${config} \
  save_path ${exp_dir}


$PYTHON tool/eval.py ${exp_dir}