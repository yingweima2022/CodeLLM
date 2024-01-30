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
# echo "bash run_distributed_train.sh 1.DATA_DIR 2.DEVICE_NUM 3.TYPE 4.MODE 5.STAGE_NUM 6.MICRO_SIZE"
# echo "7.PER_BATCH 8.RANK_START 9.LOCAL_DEVICE_NUM"
# echo "for example:"
# echo "#######no pipeline#######"
# echo "bash run_distributed_train.sh pangu_alpha_r1.7_huawei_1/output/ 4 fp16 2.6B 1 1 4 0 4"
# echo "#######pipeline#######"
# echo "bash run_distributed_train.sh /path/dataset /path/hccl.json 16 fp32 2.6B 2 4 16 0 8"
# echo "bash run_distributed_train.sh /path/dataset /path/hccl.json 16 fp32 2.6B 2 4 16 8 8"
# echo "It is better to use absolute path."
# echo "=============================================================================================================="

ROOT_PATH=`pwd`
DATA_DIR=$1
RANK_SIZE=$2
PARAM_INIT_TYPE=$3
MODE=$4
STAGE_NUM=$5
MICRO_SIZE=$6
PER_BATCH=$7
export CKPT_SAVE_PATH=$8

# bash scripts/run_distribute_train_myw.sh /home/ma-user/work/notebook_code/data_pretrain_code 8 fp16 2.6B 1 1 4 /home/ma-user/work/notebook_code/temp_ckpt


for((i=0;i<${RANK_SIZE};i++));
do
    rm ${ROOT_PATH}/device$i/ -rf
    mkdir ${ROOT_PATH}/device$i
    cd ${ROOT_PATH}/device$i || exit
    export RANK_ID=$i
    export DEVICE_ID=$i
    python ${ROOT_PATH}/train.py --distribute=true --device_num=$RANK_SIZE --data_url=$DATA_DIR --run_type=train --save_checkpoint_path=$CKPT_SAVE_PATH \
    --param_init_type=$PARAM_INIT_TYPE --mode=$MODE --stage_num=$STAGE_NUM --micro_size=$MICRO_SIZE --padding_id=3 --vocab_size=130344 --eod_id=130005 \
    --per_batch_size=$PER_BATCH > log$i.log 2>&1 &
done
