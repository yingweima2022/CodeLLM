#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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
execute_path=$(pwd)
self_path=$(cd "$(dirname "$0")" || exit; pwd)
export STRATEGY=$1
export TOKENIZER=$2
export CKPT_PATH=$3
export CKPT_NAME=$4
export MODE=$5
export PARAM_INIT_TYPE=fp16
export DEVICE_TARGET=$6 # Ascend or GPU
export TASK_NAME=$7
export EVAL_DATA_URL=$8
export RANK_SIZE=$9

# export FILE_PATH=/home/ma-user/work/notebook_code/codepangu_ckpt_pro
# bash scripts/run_distribute_eval_code.sh ${FILE_PATH}/strategy_load_ckpt/strategy.ckpt ${FILE_PATH}/tokenizer/ ${FILE_PATH}/checkpoint_file filtered 2.6B Ascend all /home/ma-user/work/notebook_code/data_downst/all 1


if [ $# != 9 ] ; then
  echo "The input argument number is not sufficient, please see the follow examples:"
  echo "USAGE: bash $0 STRATEGY TOKENIZER CKPT_PATH CKPT_NAME MODE DEVICE_TARGET TASK_NAME EVAL_DATA_URL"
  echo " e.g.: bash $0 /home/ckpts/strategy/strategy.ckpt /home/ckpts/tokenizer/ /home/ckpts/checkpoints 2.6B Ascend fp32 c3 /home/data/c3/data/"
  exit 1;
fi

for((i=0;i<${RANK_SIZE};i++));
do
    rm -rf ${execute_path}/device$i/
    mkdir ${execute_path}/device$i/
    cd ${execute_path}/device$i/ || exit
    export RANK_ID=$i
    export DEVICE_ID=$i
    python -s ${self_path}/../predict.py --strategy_load_ckpt_path=$STRATEGY --tokenizer_path=$TOKENIZER --load_ckpt_path=$CKPT_PATH \
                    --load_ckpt_name=$CKPT_NAME --mode=$MODE --run_type=predict --param_init_type=$PARAM_INIT_TYPE \
                    --distribute=false  --device_target=$DEVICE_TARGET --per_batch_size 1 --op_level_model_parallel_num 1 \
                    --eval_task=$TASK_NAME  --padding_id=3 --vocab_size=130344 --eod_id=130005 --end_token=130005 \
                    --eval_data_url=$EVAL_DATA_URL \
                    --tokenizer_path=$TOKENIZER >log0.log 2>&1 &
done

# --padding_id=3 --vocab_size=130344 --eod_id=130005 --end_token=130005