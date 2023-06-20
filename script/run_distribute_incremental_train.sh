#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# echo "=============================================================================================================="
# echo "Please run the script as: "
# echo "bash run_distribute_train_incremental_train.sh DATA_DIR RANK_TABLE_FILE DEVICE_NUM"
# echo "for example: scripts/run_distribute_incremental_train.sh DATASET RANK_TABLE RANK_SIZE PARAM_INIT_TYPE \\"
# echo "MODE PER_BATCH STRATEGY_CKPT  CKPT_PATH CKPT_NAME"
# echo "It is better to use absolute path."
# echo "=============================================================================================================="


# bash scripts/run_distribute_incremental_train.sh /home/ma-user/work/notebook_code/data_prompt_mindrecord_code_thu_code_level 8 fp16 2.6B 4 /home/ma-user/work/notebook_code/pangucode_ckpt/strategy_load_ckpt/strategy.ckpt /home/ma-user/work/notebook_code/pangucode_ckpt/checkpoint_file filerted /home/ma-user/work/notebook_code/codepangu_ckpt_pro_level

ROOT_PATH=`pwd`
DATA_DIR=$1
RANK_SIZE=$2
PARAM_INIT_TYPE=$3
MODE=$4
PER_BATCH=$5
export STRATEGY=$6
export CKPT_PATH=$7
export CKPT_NAME=$8
export CKPT_SAVE_PATH=$9

for((i=0;i<${RANK_SIZE};i++));
do
    rm ${ROOT_PATH}/device$i/ -rf
    mkdir ${ROOT_PATH}/device$i
    cd ${ROOT_PATH}/device$i || exit
    export RANK_ID=$i
    export DEVICE_ID=$i
    python ${ROOT_PATH}/instruction_following.py --distribute=true --device_num=$RANK_SIZE --data_url=$DATA_DIR --run_type=train \
    --param_init_type=$PARAM_INIT_TYPE --mode=$MODE --incremental_training=1 --strategy_load_ckpt_path=$STRATEGY --start_lr=2e-5 --end_lr=1e-6 --epoch_size=1 --op_level_model_parallel_num 1 --padding_id=3 --vocab_size=130344 --eod_id=130005 \
    --decay_steps=8750 --load_ckpt_path=$CKPT_PATH --load_ckpt_name=$CKPT_NAME --per_batch_size=$PER_BATCH --save_checkpoint_path=$CKPT_SAVE_PATH --save_checkpoint_steps 2000 > log$i.log 2>&1 &
done

# --padding_id=3 --vocab_size=130344 --eod_id=130005 #nocode 2700 8750