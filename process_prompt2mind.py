# !/usr/bin/env python
# -*- coding: utf-8 -*-

# @Date: 2022/7/28
# @Author: 2022 PCL taoht
"""
transform openwebtext dataset to mindrecord.
"""
import argparse
import glob
import random
import time
import os
import numpy as np
from multiprocessing import Process, Queue, Pool, Lock
from mindspore.mindrecord import FileWriter
import mindspore.dataset as ds
import jieba
from tqdm import tqdm
import traceback
import json
import copy

parser = argparse.ArgumentParser()

###########################
parser.add_argument('--data_path', type=str,
                    default='../data_prompt/code/*.json',
                    help="Location of sample txt files.")
parser.add_argument('--output_file', type=str,
                    default='../data_prompt_mindrecord_code_thu/code/transfered_mindrecord',
                    help="Save mindrecord path.")
parser.add_argument('--num_process', type=int, default=10,
                    help="Save checkpoint path.")
parser.add_argument('--tokenizer', type=str, default='thu',
                    help="Save checkpoint path.")
parser.add_argument('--SEQ_LEN', type=int, default=1024)
parser.add_argument('--dataset_type', type=str, default='openwebtext')

args = parser.parse_args()
print(args)
SEQ_LEN = args.SEQ_LEN + 1
# working_dir = os.getcwd()
working_dir = os.path.dirname(os.path.abspath(__file__))

##-------------------------------------------------------------------
if args.tokenizer == 'thu':
    try:
        from transformers import GPT2Tokenizer, AutoTokenizer, AutoModel
    except ModuleNotFoundError:
        print("module 'transformers' not installed.")
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained("F:\\THU-GLM", trust_remote_code=True)
elif args.tokenizer == 'pangu4w':
    from src.tokenization_jieba import JIEBATokenizer
    model_file = '/home/ma-user/work/notebook_code/pangu_ckpt/tokenizer/vocab.model'
    tokenizer = JIEBATokenizer(model_file)
##-------------------------------------------------------------------

PAD = 3
EOT = 130005
print('pad id :', PAD)  # 128297
print('eot id :', EOT)  # 128298
print('vocab size :', tokenizer.vocab_size)
# exit()

# 得到jsonl的路径列表
def get_jsonl_files(dir_path):
    files = os.listdir(dir_path)
    jsonlfiles = [(dir_path+i) for i in files if i.split('.')[-1] == 'jsonl']
    return jsonlfiles


# 将一个list尽量均分成n份，限制len(list)==n，份数大于原list内元素个数则分配空list[]
def divideIntoNstrand(listTemp, n):
    twoList = [[] for i in range(n)]
    for i, e in enumerate(listTemp):
        twoList[i % n].append(e)
    return twoList


def chunks(lst, n):
    """ yield n sized chunks from list"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def padding_eot(chunk):
    pad = [PAD] * (SEQ_LEN - len(chunk))
    chunk.extend(pad)
    return chunk

EN_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

ZN_PROMPT_DICT = {
    "prompt_input": (
        "以下是一个描述任务的指令，并附有提供进一步背景信息的输入。 "
        "编写一个响应来适当的完成任务。\n\n"
        "### 指令:\n{instruction}\n\n### 输入:\n{input}\n\n### 响应:"
    ),
    "prompt_no_input": (
        "以下是一个描述任务的指令。 "
        "编写一个响应来适当的完成任务。\n\n"
        "### 指令:\n{instruction}\n\n### 响应:"
    ),
}

def tokenize_openwebtext(iterator, tokenizer):
    """ tokenize openwebtext dataset"""

    for file_path in iterator:
        if os.path.getsize(file_path) == 0:
            continue
        content = []
        if '.json' not in file_path:
            print(file_path, "jump...")
            continue
        if 'alpaca_data' in file_path:
            prompt_input, prompt_no_input = EN_PROMPT_DICT["prompt_input"], EN_PROMPT_DICT["prompt_no_input"]
        else:
            prompt_input, prompt_no_input = ZN_PROMPT_DICT["prompt_input"], ZN_PROMPT_DICT["prompt_no_input"]
            
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = json.load(file)
            tqdm_iterator = tqdm(enumerate(lines), desc='tokenize', total=len(lines))
            for idx, data in tqdm_iterator:
                # data = json.loads(data)
                # if data.get('input')!="":
                #     source=prompt_input.format_map(data)
                # else:
                #     source=prompt_no_input.format_map(data)       
                # print('data', data)
                source = data['input']
                target=data['target']
                
                # print('source', source)
                # print('target', target)
                # exit()
                
                # input
                tokenized_source = tokenizer.tokenize(source)
                source_ids = tokenizer.convert_tokens_to_ids(tokenized_source)
                
                example = source+target
                tokenized_example = tokenizer.tokenize(example)
                example_ids = tokenizer.convert_tokens_to_ids(tokenized_example) + [EOT]
                
                labels_ids = copy.deepcopy(example_ids)
                labels_ids[:len(source_ids)]=[PAD]*len(source_ids)
                
                if len(example_ids)>SEQ_LEN:
                    example_ids=example_ids[:SEQ_LEN]
                    labels_ids=labels_ids[:SEQ_LEN]
                # print('idx', idx)
                # print('source', source)
                # print('target', target)
                # print('example', example)
                # print('tokenized_example', tokenized_example)
                # print('example_ids', example_ids)
                # print('example_ids', np.array(padding_eot(example_ids)))
                # print('labels_ids', labels_ids)
                # print('labels_ids', np.array(padding_eot(labels_ids)))
                # if idx==5:
                #     exit()
                    
                sample = {}
                if len(example_ids) == SEQ_LEN:
                    sample['input_ids'] = np.array(example_ids, dtype=np.int32)
                    sample['label_ids'] = np.array(labels_ids, dtype=np.int32)
                    # if len(sample['input_ids'])!=SEQ_LEN or len(sample['label_ids'])!=SEQ_LEN:
                    #     print(len(example_ids) , SEQ_LEN)
                    #     print(len(labels_ids) , SEQ_LEN)
                    #     print('error len0', sample['input_ids'].shape) 
                    #     print('error len0', sample['label_ids'].shape) 
                    #     exit()
                    yield sample
                else:
                    sample['input_ids'] = np.array(padding_eot(example_ids), dtype=np.int32)
                    sample['label_ids'] = np.array(padding_eot(labels_ids), dtype=np.int32)
                    # if len(sample['input_ids'])!=SEQ_LEN or len(sample['label_ids'])!=SEQ_LEN:
                    #     print('error len1', sample['label_ids'].shape) 
                    #     exit()
                    yield sample

                    
def write_to_mindrecord(input_q, mindrecord_filename, tokenizer):
    try:
        print("==>> Begin to create mindrecord file: {}".format(mindrecord_filename), flush=True)

        if os.path.exists(mindrecord_filename):
            os.remove(mindrecord_filename)
        if os.path.exists(mindrecord_filename + ".db"):
            os.remove(mindrecord_filename + ".db")

        writer = FileWriter(file_name=mindrecord_filename, shard_num=1)

        # step3: define you own mindrecord schema
        schema = {"input_ids": {"type": "int32", "shape": [-1]}, "label_ids":{"type": "int32", "shape": [-1]}, }
        writer.add_schema(schema, args.dataset_type)

        time0 = time.time()
        files_list = input_q.get()
        if len(files_list) > 0:
            # if not input_q.empty():, .empty() bug for true
            item_iter = tokenize_openwebtext(files_list, tokenizer)
            # write data to mindrecord file
            while 1:
                data = []
                try:
                    for _ in range(500):
                        onedata=next(item_iter)
                        if len(onedata['input_ids'])!=SEQ_LEN or len(onedata['label_ids'])!=SEQ_LEN:
                            print('error len', (onedata['label_ids'].shape), (onedata['label_ids'].shape))
                            exit()
                        data.append(onedata)
                except Exception as error:
                    print('StopIteration error', error)
                if len(data) > 0 and isinstance(data, list):
                    writer.write_raw_data(data)
                    # print("==>> write_raw_data : {} nums in mindrecord file: {}".format(str(len(data)), mindrecord_filename), flush=True)
                else:
                    time1 = time.time()
                    writer.commit()
                    print("==>> End to create mindrecord file: {}".format(mindrecord_filename), flush=True)
                    print("==>> create mindrecord file {} time use: {}".format(mindrecord_filename, time1 - time0),
                          flush=True)
                    break
        else:
            print("Empty file iteration...", flush=True)
            time.sleep(0.01)
    except Exception as e:
        print("Error in write_to_mindrecord():", flush=True)
        print(traceback.format_exc(), flush=True)


if __name__ == '__main__':
    ###
    out_dir, out_file = os.path.split(os.path.abspath(args.output_file))
    os.system(f'rm -rf {out_dir}')
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    if args.dataset_type == 'openwebtext':
        file_iter = list(glob.iglob(args.data_path))
        file_iter.sort()
        random.seed(10)
        random.shuffle(file_iter)
        print('num files of this machine : ', len(file_iter))

        #####################################################################################
        # step1: define your number of mindrecord files
        # step2: define your subprocess number, it should be equal number of mindrecord_filenames
        subprocess_number = args.num_process
        mindrecord_filenames = [f'{args.output_file}_{str(i)}.mindrecord' for i in range(subprocess_number)]
        print("==>> Start {} subprocess to create mindrecord file".format(subprocess_number), flush=True)

        time_start = time.time()
        # step3: Bind subprocesses and MindRecord files
        process_list = {}
        input_q_list = {}
        current_index = 0
        for filename in mindrecord_filenames:
            input_q = Queue(1)
            p1 = Process(name=filename, target=write_to_mindrecord, args=(input_q, filename, tokenizer))
            p1.start()
            process_list[filename] = p1
            input_q_list[filename] = input_q

            current_index += 1
            if current_index >= subprocess_number:
                break

        print("==>> Begin to create mindrecord files.", flush=True)
        # end step4
        file_iter_list = divideIntoNstrand(file_iter, subprocess_number)
        print([len(i) for i in file_iter_list])
        # put the data to multi child process queue
        for idx, filename in enumerate(input_q_list):
            if not input_q_list[filename].full():
                input_q_list[filename].put(file_iter_list[idx])

        # waiting for all the data write success
        for filename in input_q_list:
            while input_q_list[filename].full():
                time.sleep(0.01)
            # send end to the child process, child process will commit and exit
            input_q_list[filename].put("end")
            # print(filename, input_q_list[filename].full())

            # wait child process exit
            while process_list[filename].is_alive():
                # print(filename, process_list[filename].is_alive())
                time.sleep(0.01)
        time_end = time.time()
        time_use = int(time_end - time_start)
        print(time_use)
        print("Time consuming: {} h.{} m.{} s".format(int((time_use / 3600)), (int(time_use / 60)) % 60, time_use % 60),
              flush=True)
        
        
        
    # 合并并重洗mindreocrd文件
    data_path = "/home/ma-user/work/notebook_code/data_prompt_mindrecord_code_thu/code/"
    home_path = os.path.join(os.getcwd(), data_path)
    
    files = os.listdir(data_path)
    data = [
        os.path.join(home_path, name) for name in files
        if not (name.endswith(".db") or name == '.ipynb_checkpoints')
    ]
    print('datalist', data)
    # Ensure the order of mindrecords is same in all machines, otherwise it will meet loss converge problem.
    data.sort()

    # Load data files and preprocess
    # print('data', data[data_start_index:])
    ds.config.set_seed(1)
    # Control the size of data queue in the consideration of the memory
    ds.config.set_prefetch_size(1)

    dataset = ds.MindDataset(data[0:], columns_list=['input_ids', 'label_ids'], shuffle=True)
    dataset.save('/home/ma-user/work/notebook_code/data_prompt_mindrecord_code_thu/code/all_instruction.mindrecord', num_files=1)

    print('结束写入')
        
        
        
    # # read the mindrecord files
    # count_all = 0
    # print("==>> Begin to read data from mindrecord files.", flush=True)
    # for i in range(current_index):
    #     data_set = ds.MindDataset(dataset_files=[mindrecord_filenames[i]])
    #     count = 0
    #     for item in data_set.create_dict_iterator(output_numpy=True):
    #         count += 1
    #     count_all += count
    #     print("Got {} samples from {}".format(count, mindrecord_filenames[i]))
    # print("Got {} samples all".format(count_all))