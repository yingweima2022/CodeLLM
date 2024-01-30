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
from multiprocessing import Process, Queue, Pool
from mindspore.mindrecord import FileWriter
import mindspore.dataset as ds
from src.tokenization_jieba import JIEBATokenizer
import jieba

parser = argparse.ArgumentParser()

###########################
parser.add_argument('--data_path', type=str,
                    default='/cache/data_sample/*.txt',
                    help="Location of sample txt files.")
parser.add_argument('--output_file', type=str,
                    default='./output/transfered_mindrecord',
                    help="Save mindrecord path.")
parser.add_argument('--num_process', type=int, default=20,
                    help="Save checkpoint path.")
parser.add_argument('--tokenizer', type=str, default='gpt',
                    help="Save checkpoint path.")
parser.add_argument('--SEQ_LEN', type=int, default=1024)
parser.add_argument('--dataset_type', type=str, default='openwebtext')

args = parser.parse_args()
print(args)
SEQ_LEN = args.SEQ_LEN + 1
# working_dir = os.getcwd()
working_dir = os.path.dirname(os.path.abspath(__file__))

##-------------------------------------------------------------------
if args.tokenizer == '4w-bpe':
    vocab_path = working_dir + '/bpe_4w_pcl/vocab.vocab'
    model_file = working_dir + '/bpe_4w_pcl/vocab.model'
    tokenizer = JIEBATokenizer(vocab_path, model_file)
    langs_ID = {'zh': 39999, 'en': 39998}
    translate_ID = 39997
elif args.tokenizer == 'spm_13w':
    from spm_13w.tokenizer import SpmTokenizer, langs_ID, translate_ID

    vocab_file = working_dir + '/spm_13w/spm.128k.model.1'
    tokenizer = SpmTokenizer(vocab_file)
elif args.tokenizer == 'spm_25w':
    from spm_25w.tokenizer_spm import SpmTokenizer, langs_ID, translate_ID

    vocab_file = working_dir + '/spm_25w/spm.250k.model'
    tokenizer = SpmTokenizer(vocab_file)
elif args.tokenizer == 'gpt':
    try:
        from transformers import GPT2Tokenizer
    except ModuleNotFoundError:
        print("module 'transformers' not installed.")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
##-------------------------------------------------------------------

PAD = tokenizer.eos_token_id
EOT = tokenizer.eos_token_id
print('pad id :', PAD)  # 128297
print('eot id :', EOT)  # 128298
print('vocab size :', tokenizer.vocab_size)


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


def tokenize_openwebtext(iterator):
    """ tokenize openwebtext dataset"""

    for file_path in iterator:
        if os.path.getsize(file_path) == 0:
            continue
        content = []
        with open(file_path, 'r', encoding='utf-8') as f:
            if '.txt' not in file_path:
                print(file_path, "jump...")
                continue
            else:
                ###########################################
                read_flags = False
                if len(f.read().split('\n\n')) <= 10:
                    read_flags = True
                with open(file_path, 'r', encoding='utf-8') as fl:
                    if read_flags:
                        fl_new = fl.readlines()
                    else:
                        fl_new = fl.read().split('\n\n')
                    ##############################################
                    for para in fl_new:
                        if para:
                            para_jieba = ''.join(jieba.cut(para))
                            tokenized_text_langs = tokenizer.tokenize(para_jieba)
                            langs_id = tokenizer.convert_tokens_to_ids(tokenized_text_langs)
                            content.append(langs_id + [EOT])

        random.shuffle(content)
        content_new = []
        for i in content:
            content_new += i

        for chunk in chunks(content_new, SEQ_LEN):
            sample = {}
            if len(chunk) == SEQ_LEN:
                sample['input_ids'] = np.array(chunk, dtype=np.int32)
                yield sample
            else:
                sample['input_ids'] = np.array(padding_eot(chunk), dtype=np.int32)
                yield sample


def tokenize_openwebtext_mpangu(iterator):
    """ tokenize openwebtext dataset"""

    for file_path in iterator:
        if os.path.getsize(file_path) == 0:
            continue
        content = []

        with open(file_path, 'r', encoding='utf-8') as f0:
            ###########################################
            read_flags = False
            if len(f0.read().split('\n\n')) <= 10:
                read_flags = True
            ###########################################
            with open(file_path, 'r', encoding='utf-8') as f:
                if read_flags:
                    fl_new = fl.readlines()
                else:
                    fl_new = fl.read().split('\n\n')

                ##----------------------单语语料处理--------------------
                if 'corpus' not in file_path.split('/')[-1]:
                    langs = file_path.split('/')[-1][:2]
                    id_langs = langs_ID[langs]
                    for para in fl_new:
                        if para:
                            para = '' + para
                            tokenized_text_langs = tokenizer.tokenize(para)
                            langs_id = tokenizer.convert_tokens_to_ids(tokenized_text_langs)
                            content.append([id_langs] + langs_id + [tokenizer.eot_id])
                else:
                    ##----------------双语语料处理------------------------------
                    src_langs = file_path.split('/')[-1][:2]
                    tag_langs = file_path.split('/')[-1][3:5]

                    try:
                        for para in fl_new:
                            if para:
                                src_data, tag_data = para.split("\t")
                                # pangu 4w词表中文文本需要jiaba分词，spm中文文本不需要jieba分词
                                src_data_new = '' + src_data
                                tag_data_new = '' + tag_data

                                # zh corpus save ,other corpus save mono random
                                if isinstance(src_data_new, str) and isinstance(tag_data_new, str):
                                    tokenized_text_src = tokenizer.tokenize(src_data_new)
                                    src_id = tokenizer.convert_tokens_to_ids(tokenized_text_src)

                                    tokenized_text_tag = tokenizer.tokenize(tag_data_new)
                                    tag_id = tokenizer.convert_tokens_to_ids(tokenized_text_tag)

                                    content.append([langs_ID[src_langs]] + \
                                                   src_id + \
                                                   [translate_ID] + \
                                                   [langs_ID[tag_langs]] + \
                                                   tag_id + [tokenizer.eot_id])
                                    content.append([langs_ID[tag_langs]] + \
                                                   tag_id + \
                                                   [translate_ID] + \
                                                   [langs_ID[src_langs]] + \
                                                   src_id + [tokenizer.eot_id])
                                else:
                                    print("Not 2 para str...\n")
                    except:
                        print("Split error, jump...", para)
                        ###########################################

        random.shuffle(content)
        print("111111111111111", len(content))
        content_new = []
        for i in content:
            content_new += i

        for chunk in chunks(content, SEQ_LEN):
            sample = {}
            if len(chunk) == SEQ_LEN:
                sample['input_ids'] = np.array(chunk, dtype=np.int32)
                yield sample
            else:
                sample['input_ids'] = np.array(padding_eot(chunk), dtype=np.int32)
                yield sample


def write_to_mindrecord(input_q, mindrecord_filename):
    print("==>> Begin to create mindrecord file: {}".format(mindrecord_filename), flush=True)

    if os.path.exists(mindrecord_filename):
        os.remove(mindrecord_filename)
    if os.path.exists(mindrecord_filename + ".db"):
        os.remove(mindrecord_filename + ".db")

    writer = FileWriter(file_name=mindrecord_filename, shard_num=1)

    # step3: define you own mindrecord schema
    schema = {"input_ids": {"type": "int32", "shape": [-1]}, }
    writer.add_schema(schema, args.dataset_type)

    time0 = time.time()
    files_list = input_q.get()
    if len(files_list) > 0:
        # if not input_q.empty():, .empty() bug for true
        item_iter = tokenize_openwebtext(files_list)
        # write data to mindrecord file
        while 1:
            data = []
            try:
                for _ in range(100):
                    data.append(next(item_iter))
            except Exception as error:
                print(error)
            if len(data) > 0 and isinstance(data, list):
                writer.write_raw_data(data)
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


if __name__ == '__main__':
    ###
    out_dir, out_file = os.path.split(os.path.abspath(args.output_file))
    # out_dir = './tmp'
    # out_file = 'test_mindrecord'
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
            p1 = Process(name=filename, target=write_to_mindrecord, args=(input_q, filename))
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

        # read the mindrecord files
        count_all = 0
        print("==>> Begin to read data from mindrecord files.", flush=True)
        for i in range(current_index):
            data_set = ds.MindDataset(dataset_files=[mindrecord_filenames[i]])
            count = 0
            for item in data_set.create_dict_iterator(output_numpy=True):
                count += 1
            count_all += count
            print("Got {} samples from {}".format(count, mindrecord_filenames[i]))
        print("Got {} samples all".format(count_all))
