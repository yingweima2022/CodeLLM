# [CodePanGu2.6B Description](#contents)
We release code to explore training large models with hundreds of millions of parameters. Taking advantage of the parallel nature of MindSpore, we adopt efficient model parallelism and data parallelism technologies such as operator-level parallelism, which can be easily extended to thousands of NPUs and hundreds of billions of parameters with only minor modifications.


# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)
- **You can use Qizhi(OpenI) platform to obtain free computing resources.** [OpenI](https://www.openi.org.cn/)
  
## Dataset Generation

As the format of the downstream tasks can be various, the `process_prompt2mind.py` provides a basic usage of how to process your fine-tune text files(.json). Please prepare your data with following format, each line is a piece of continuous text for each file:

```text
{'input':'please describe your university life.', 'target':'My university life is rich and colorful. In addition to academic courses, there are also rich club activities.'}
{'input': xxx, 'target': xxx}
```
Suppose the text data is under the `./data` and **each text file ends with 'json'**, we can run the following command to generate the mindrecord files with seq_length=1025.

```bash
python -m process_prompt2mind --input_glob  'data/*.json' --tokenizer thu --data_column_name input_ids --seq_length 1025
```
The script will chunk the line with 1025 tokens. For the chunk with no more 1025 tokens, the chunk will be ignored.

The output files is under `./output`.  The default tokenizer adopts the transformers's tokenizer. Note the `vocab_szie` is determined by the vocab file.

- tokenizer: The tokenizer used for tokening the  text. It can be `thu`(required `transformers`) or `pangu4w`. Note the `thu` tokenizer requires the `transformers`,`pytorch` or `tensorflow`.  `pangu4w` tokenizer requires two addition files `vocab.model`. Click [here](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha/src/branch/master/tokenizer) to download them.
- data_column_name: The name of feature columns for mindrecord.
- seq_length: Default 1025. The preprocess will generate mindrecord with sequence length 1025 for each example.

### Instruction tuning

Before we start Incremental Training, the following two steps must be done:

1. Process the dataset using the released vocab, please refer to the [Increnmental Training in Dataset Generatiogn](#Incremental Training)
2. Download the`checkpoint` and `strategy` file according to the  [Download Checkpoint](#Download Checkpoint). Each host should own the complete checkpoint files.

Then run the following command to start incremental training with `2.6B` configure:

```bash
export FILE_PATH=/home/your_path/ckpts
bash scripts/run_distribute_incremental_train.sh DATASET RANK_TABLE 8 fp32 2.6B 8 ${FILE_PATH}/strategy_load_ckpt/strategy.ckpt  ${FILE_PATH}/checkpoint_file filitered
```
The above command involves some `args` described below:

- DATASET: The path to the mindrecord files's parent directory . For example: `/home/work/mindrecord/`.
- RANK_TABLE: The details of the rank table can be found [here](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html). It's a json file describes the `device id`, `service ip` and `rank`.
- RANK_SIZE: The device number. This can be your total device numbers. For example, 8, 16, 32 ...
- TYPE: The param init type. The parameters will be initialized with float32. Or you can replace it with `fp16`. This will save a little memory used on the device.
- MODE: The configure mode. This mode will set the `hidden size` and `layers` to make the parameter number near 2.6 billions. The other mode can be `13B` (`hidden size` 5120 and `layers` 40, which needs at least 16 cards to train.) and `200B`.
- STAGE_NUM: The number of pipeline stages. When the `stage_num` is large than 1, the pipeline parallel mode would be applied. This configure indicates the number of sub graphs in pipeline parallel mode.
- MICRO_SIZE: The number of micro batches in pipeline parallel mode. It should large than `stage_num`.
- PER_BATCH: The batch size for each data parallel-way. default 8.
- RANK_START: The start of rank_id in current machines, it helps to set the rank_id for each machine in multi-machine scenario.
- LOCAL_DEVICE_NUM: The device number of the local machine.
- EXPERT_NUM_PER_EP: Expert nums in one data parallel dim.
- ENABLE_ALLTOALL: Enable alltoall communication. default 0.

## [Prediction](#contents)

### Download Checkpoint

Please refer to the [website](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha) to download the following parts:

- tokenizer: vocab.model
- checkpoint file: \*.part\[0-4\] (need to extract) and *.npy under the same parameter size
- strategy file: a file described how the parameters are sliced across different devices.

Here we suppose the downloaded checkpoint, tokenizer and strategy file is organized as follows:

**CodePanGu2.6B** checkpoint files is coming soon.

**Note**: In the following sections, we will refer the path as `ckpts` as `/home/your_path/ckpts`.

```shell
ckpts
├── checkpoint_file
│   ├── filtered_*.ckpt
│   ├── word_embedding.npy
│   ├── top_query_embedding.npy
│   └── position_embedding.npy
├── strategy_load_ckpt
│   └── strategy.ckpt
└── tokenizer
    └── vocab.model
```
## Evaluation on Downstream Tasks

This script provides the evaluation of following tasks:

- [C3](https://github.com/nlpdata/c3)

### Download the Dataset

Click the link of above tasks and download the data. Take the C3 for example, unzip the dataset to
`/home/my_path/data/c3`

Its structure should be as followings:

```text
c3
├── data
│   ├── c3-d-dev.json
│   ├── c3-d-test.json
│   ├── c3-d-train.json
│   ├── c3-m-dev.json
│   ├── c3-m-test.json
│   └── c3-m-train.json
├── license.txt
└── README.md
```

### Download the Checkpoint

Please follow the instructions in section [Prediction](#prediction) to download the checkpoint.

### Run the Evaluation

The most of the arguments are same with the section [Prediction in Standalone mode](#prediction-in-standalone-mode),
except the last argument `TASK` and `TASK_PATH`. Currently, we support only `c3` task. The following commands will
launch the programs to start evaluation with 2.6B model.

```bash
export FILE_PATH=/home/your_path/ckpts
export DEVICE_TARGET=Ascend # or GPU
export TASK=c3
export TASK_PATH=/home/your_c3_data_path/data # You should point to the data directory under the c3 path
bash scripts/run_standalone_eval.sh ${FILE_PATH}/strategy_load_ckpt/strategy.ckpt \
${FILE_PATH}/tokenizer/  ${FILE_PATH}/checkpoint_file filitered 2.6B $DEVICE_TARGET $TASK $TASK_PATH
```

For the model with 2.6B, it takes about 13 minutes to get the results. Log can be found under the `device0/log0.log`.
It should look like this:

```text
Metric for dataset c3 is {'top1_acc': 0.5430}
```

Naturally, you should also cite the [PanGu-OpenI repo](https://openi.pcl.ac.cn/PCL-Platform.Intelligence/PanGu-Alpha.git) and the [MindSpore repo](https://gitee.com/mindspore/models.git).
