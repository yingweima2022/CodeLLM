# Copyright 2022 Huawei Technologies Co., Ltd
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
"""
Downstream tasks
"""
import os
import json
import jsonlines
from collections import defaultdict

import pandas as pd


def read_jsonl(data_path):
    """
    Load the json lines from the specific file.
    Args:
        data_path: The json file path.

    Returns:
        The read json data.
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = f.readlines()
        lines = []
        for line in reader:
            lines.append(json.loads(line.strip()))
        return lines


def load_qa_dataset_c3(data_dir, split, in_context, tokenizer):
    if split.lower() in ('train',):
        train_path = os.path.join(data_dir, 'c3-d-train.json')
        source_data = pd.read_csv(open(train_path, 'r'))
    elif split.lower() in ('validation',):
        dev_path = os.path.join(data_dir, 'c3-d-dev.json')
        source_data = json.load(open(dev_path, 'r'))
    elif split.lower() in ('test',):
        source_data = []
    else:
        raise ValueError(f"The split {split} is not supported. "
                         f"Current only supports 'train', 'dev', 'validation'")

    examples = []
    total_count = 0
    for instance in source_data:
        context = "".join(instance[0])
        queries = instance[1]
        for query in queries:
            question = query['question']
            choices = query['choice']
            answer_true = query['answer']
            total_count += 1
            for choice in choices:
                query_text = f"问：{question}\n答：{choice}\n该答案来自对话：{context}"
                input_str = f"{query_text}"
                prompt = f"问：{question}\n答：{choice}\n该答案来自对话："
                
                if in_context:
                    context1 = "问：女的最喜欢哪种电影?\n答：喜剧片\n该答案来自对话：男：你今天晚上有时间吗?我们一起去看电影吧?女：你喜欢恐怖片和爱情片，但是我喜欢喜剧片，科幻片一般。所以……\n"
                    context2 = "问：这个孩子以前学习怎么样?\n答：不努力\n该答案来自对话：男：妈妈，这次数学考试我得了100分。女：你看，我说对了吧?只要你努力学习，认真写作业，及时复习，就一定能学好。\n"
                    context3 = "问：从对话中，我们可以知道什么?\n答：天气很不好\n该答案来自对话：男：这场雨下得又大又急，马路上到处都是水。女：对呀，都下了一个多小时了，怎么还不停?\n"
                    context_all = context1 + context2 + context3
                    input_str = context_all + input_str
                    prompt = context_all + prompt
                input_str.replace('?', '？')
                prompt.replace('?', '？')
                examples.append({
                    "idx": total_count,
                    "input_str": input_str,
                    "prompt": prompt,
                    "is_correct": answer_true == choice,
                })
    return examples


def load_qa_dataset_ocnli(data_dir, split, in_context, tokenizer):
    if split.lower() in ('train',):
        train_path = os.path.join(data_dir, 'train.50k.json')
        source_data = pd.read_csv(open(train_path, 'r'))
    elif split.lower() in ('validation',):
        dev_path = os.path.join(data_dir, 'dev.json')
        with open(dev_path, 'r', encoding="utf-8") as f:
            source_data = f.readlines()
    elif split.lower() in ('test',):
        source_data = []
    else:
        raise ValueError(f"The split {split} is not supported. "
                         f"Current only supports 'train', 'dev', 'validation'")
        
    label_map = {"entailment": "蕴含", "neutral": "中立", "contradiction": "矛盾"}

    examples = []
    total_count = 0
    for instance in source_data:
        instance = json.loads(instance)
        sentence1 = instance["sentence1"]
        sentence2 = instance["sentence2"]
        if instance["label"] not in label_map:
            continue
        label = label_map[instance["label"]]
        answer_true = label
        
        example = ""  # zero-shot
        input_str_one = f"{example}{sentence1}？对，{sentence2}"
        input_str_two = f"{example}{sentence1}？或，{sentence2}"
        input_str_thr = f"{example}{sentence1}？错，{sentence2}"
        
        total_count += 1
        
        prompt_one = f"{example}{sentence1}？对，"
        prompt_two = f"{example}{sentence1}？或，"
        prompt_thr = f"{example}{sentence1}？错，"
        
        if in_context:
            context1 = "他以身殉职,终年59岁？对，他已经去世了\n他以身殉职,终年59岁？或，他是在今年去世的\n他以身殉职,终年59岁？错，他活到了70岁\n"
            context2 = "健全国家安全体系？对，国家安全体系需要健全\n健全国家安全体系？或，国家安全体系对于国防非常重要\n健全国家安全体系？错，国家安全体系无关紧要\n"
            context = context1 + context2
            input_str_one = context + input_str_one
            input_str_two = context + input_str_two
            input_str_thr = context + input_str_thr
            prompt_one = context + prompt_one
            prompt_two = context + prompt_two
            prompt_thr = context + prompt_thr
            
        examples.append({
            "idx": total_count,
            "input_str": input_str_one,
            "prompt": prompt_one,
            "is_correct": answer_true == "蕴含",
        })
        examples.append({
            "idx": total_count,
            "input_str": input_str_two,
            "prompt": prompt_two,
            "is_correct": answer_true == "中立",
        })
        examples.append({
            "idx": total_count,
            "input_str": input_str_thr,
            "prompt": prompt_thr,
            "is_correct": answer_true == "矛盾",
        })
    return examples


def load_qa_dataset_cosqa(data_dir, split, in_context, tokenizer):
    if split.lower() in ('train',):
        train_path = os.path.join(data_dir, 'train.50k.json')
        source_data = pd.read_csv(open(train_path, 'r'))
    elif split.lower() in ('validation',):
        dev_path = os.path.join(data_dir, 'cosqa-dev.json')
        source_data = json.load(open(dev_path, 'r'))
    elif split.lower() in ('test',):
        source_data = []
    else:
        raise ValueError(f"The split {split} is not supported. "
                         f"Current only supports 'train', 'dev', 'validation'")
        
    examples = []
    total_count = 0
    for instance in source_data:
        sentence1 = instance["doc"]
        sentence2 = instance["code"]
        answer_true = instance["label"]
        
        input_str_one = f"{sentence1}? Answered code is correct:{sentence2}"
        input_str_two = f"{sentence1}? Answered code is wrong:{sentence2}"
        
        total_count += 1
        
        prompt_one = f"{sentence1}? Answered code is correct:"
        prompt_two = f"{sentence1}? Answered code is wrong:"
        
        if in_context:
            context1 = "python code to write bool value 1? Answered code is wrong:def writeBoolean(self, n):\n        \"\"\"\n        Writes a Boolean to the stream.\n        \"\"\"\n        t = TYPE_BOOL_TRUE\n\n        if n is False:\n            t = TYPE_BOOL_FALSE\n\n        self.stream.write(t)\n"
            context2 = "are python strings hashable? Answered code is correct:def _string_hash(s):\n    \"\"\"String hash (djb2) with consistency between py2/py3 and persistency between runs (unlike `hash`).\"\"\"\n    h = 5381\n    for c in s:\n        h = h * 33 + ord(c)\n    return h\n"
            context = context1+context2
            input_str_one = context + input_str_one
            input_str_two = context + input_str_two
            prompt_one = context + prompt_one
            prompt_two = context + prompt_two
        examples.append({
            "idx": total_count,
            "input_str": input_str_one,
            "prompt": prompt_one,
            "is_correct": answer_true == 1,
        })
        examples.append({
            "idx": total_count,
            "input_str": input_str_two,
            "prompt": prompt_two,
            "is_correct": answer_true == 0,
        })
    return examples


def load_qa_dataset_jecqa(data_dir, split, in_context, tokenizer):
    if split.lower() in ('train',):
        train_path = os.path.join(data_dir, 'train.50k.json')
        source_data = pd.read_csv(open(train_path, 'r'))
    elif split.lower() in ('validation',):
        dev_path = os.path.join(data_dir, 'dev.json')
        source_data = json.load(open(dev_path, 'r'))
    elif split.lower() in ('test',):
        source_data = []
    else:
        raise ValueError(f"The split {split} is not supported. "
                         f"Current only supports 'train', 'dev', 'validation'")
        
    examples = []
    total_count = 0
    for instance in source_data:
        statement = instance["statement"]
        option_list = instance["option_list"]
        answer_true = instance["answer"]
        total_count += 1
        for idx, choice in enumerate(option_list):
            input_str_one = f"该答案：{choice}，可以回答以下问题：{statement}"
            prompt_one = f"该答案：{choice}，可以回答以下问题："
            
            if in_context:
                context1 = "该答案：人民检察院可以建议法院，将案件退回补充侦查，可以回答以下问题：张某因涉嫌强奸而被人民检察院提起公诉，在法庭审理过程中，出庭的公诉人员发现该案件需要补充侦查，则人民检察院应当如何处理?\n"
                context2 = "该答案：藏语，可以回答以下问题：五一黄金周期间，游客某甲与某乙因购买西藏大昭寺门票发生冲突并将某乙打伤。经查，某甲，汉族，在福建生活多年，略懂藏语。在该案中，拉萨警方讯问某甲，应当使用的语言是什么?\n"
                context3 = "该答案：收购违反规定出售的木材的，可以回答以下问题：根据有关司法解释，下列哪些情形(有证据证明确属被蒙骗的除外)可以认定(或推定)行为人“非法收购明知是盗伐、滥伐的林木”?\n"
                context = context1+context2+context3
                input_str_one = context + input_str_one
                prompt_one = context + prompt_one
                
            examples.append({
                "idx": total_count,
                "input_str": input_str_one,
                "prompt": prompt_one,
                "is_correct": answer_true == idx,
            })
    return examples


def load_qa_dataset_math(data_dir, split, in_context, tokenizer):
    if split.lower() in ('train',):
        train_path = os.path.join(data_dir, 'train.50k.json')
        source_data = pd.read_csv(open(train_path, 'r'))
    elif split.lower() in ('validation',):
        dev_path = os.path.join(data_dir, 'all_math.json')
        source_data = json.load(open(dev_path, 'r'))
    elif split.lower() in ('test',):
        source_data = []
    else:
        raise ValueError(f"The split {split} is not supported. "
                         f"Current only supports 'train', 'dev', 'validation'")
        
    examples = []
    total_count = 0
    for idx, instance in enumerate(source_data):
        # idx = data['id']
        problem = instance["problem"]
        answer = instance["answer"]
        choices = instance["answer_list"]
        cot1 = instance["cot1"]
        cot2 = instance["cot2"]
        cot3 = instance["cot3"]
        cot4 = instance["cot4"]
        
        for label_id, choice in enumerate(choices):
            if idx<25:
                input_str = f"问题：{problem}，答案：{choice}。可以根据如下推导得出：{cot1}"
                prompt = f"问题：{problem}，答案：{choice}。可以根据如下推导得出："
            elif idx<50:
                input_str = f"问题：{problem}，答案：{choice}。可以根据如下推导得出：{cot1}，{cot2}"
                prompt = f"问题：{problem}，答案：{choice}。可以根据如下推导得出：{cot1}，"
            elif idx<75:
                input_str = f"问题：{problem}，答案：{choice}。可以根据如下推导得出：{cot1}，{cot2}，{cot3}"
                prompt = f"问题：{problem}，答案：{choice}。可以根据如下推导得出：{cot1}，{cot2}，"
            elif idx<100:
                input_str = f"问题：{problem}，答案：{choice}。可以根据如下推导得出：{cot1}，{cot2}，{cot3}，{cot4}"
                prompt = f"问题：{problem}，答案：{choice}。可以根据如下推导得出：{cot1}，{cot2}，{cot3}，"

            examples.append({
                "idx": idx,
                "input_str": input_str,
                "prompt": prompt,
                "is_correct": label_id == answer,
            })    
    return examples


def load_qa_dataset_logic(data_dir, split, in_context, tokenizer):
    if split.lower() in ('train',):
        train_path = os.path.join(data_dir, 'train.50k.json')
        source_data = pd.read_csv(open(train_path, 'r'))
    elif split.lower() in ('validation',):
        dev_path = os.path.join(data_dir, 'logic_val.json')
        source_data = json.load(open(dev_path, 'r'))
    elif split.lower() in ('test',):
        source_data = []
    else:
        raise ValueError(f"The split {split} is not supported. "
                         f"Current only supports 'train', 'dev', 'validation'")
        
    examples = []
    total_count = 0
    for idx, instance in enumerate(source_data):
        # idx = data['id']
        problem = instance["problem"]
        answer = instance["answer"]
        choices = instance["answer_list"]
        # cot1 = instance["cot1"]
        # cot2 = instance["cot2"]
        # cot3 = instance["cot3"]
        # cot4 = instance["cot4"]

        for label_id, choice in enumerate(choices):
            # if idx<5:
            #     input_str = f"问题：{problem}，答案：{choice}。可以根据如下推导得出：{cot1}"
            #     prompt = f"问题：{problem}，答案：{choice}。可以根据如下推导得出："
            # elif idx<10:
            #     input_str = f"问题：{problem}，答案：{choice}。可以根据如下推导得出：{cot1}，{cot2}"
            #     prompt = f"问题：{problem}，答案：{choice}。可以根据如下推导得出：{cot1}，"
            # elif idx<15:
            #     input_str = f"问题：{problem}，答案：{choice}。可以根据如下推导得出：{cot1}，{cot2}，{cot3}"
            #     prompt = f"问题：{problem}，答案：{choice}。可以根据如下推导得出：{cot1}，{cot2}，"
            # elif idx<20:
            #     input_str = f"问题：{problem}，答案：{choice}。可以根据如下推导得出：{cot1}，{cot2}，{cot3}，{cot4}"
            #     prompt = f"问题：{problem}，答案：{choice}。可以根据如下推导得出：{cot1}，{cot2}，{cot3}，"
            input_str = f"该答案：{choice}，可以回答如下问题：{problem}。"
            prompt = f"该答案：{choice}，可以回答如下问题："

            examples.append({
                "idx": idx,
                "input_str": input_str,
                "prompt": prompt,
                "is_correct": label_id == answer,
            })    
    return examples


def load_qa_dataset_copa(data_dir, split, in_context, tokenizer):
    if split.lower() in ('train',):
        train_path = os.path.join(data_dir, 'train.50k.json')
        source_data = pd.read_csv(open(train_path, 'r'))
    elif split.lower() in ('validation',):
        dev_path = os.path.join(data_dir, 'test-explained.jsonl')
        source_data = jsonlines.Reader(open(dev_path, 'r+'))
    elif split.lower() in ('test',):
        source_data = []
    else:
        raise ValueError(f"The split {split} is not supported. "
                         f"Current only supports 'train', 'dev', 'validation'")
        
    examples = []
    total_count = 0
    for idx, instance in enumerate(source_data):
        # idx = data['id']
        problem = instance["p"]
        answer1 = instance["a1"]
        answer2 = instance["a2"]
        answer = int(instance["most-plausible-alternative"])-1
        cot = instance['human-explanations'][0]['text']

        # input_str1 = f"The content:{problem}，can infer:{answer1}"
        # input_str2 = f"The content:{problem}，can infer:{answer2}"
        # prompt = f"The content:{problem}，can infer:"
        
        input_str1 = f"The content:{problem}, can infer:{answer1}, because:{cot}"
        input_str2 = f"The content:{problem}, can infer:{answer2}, because:{cot}"
        prompt = f"The content:{problem}, can infer:"
        
        examples.append({
            "idx": idx,
            "input_str": input_str1,
            "prompt": prompt,
            "is_correct": answer == 0,
        })
        
        examples.append({
            "idx": idx,
            "input_str": input_str2,
            "prompt": prompt,
            "is_correct": answer == 1,
        })
        
    return examples

def load_qa_dataset_scienceqa(data_dir, split, in_context, tokenizer):
    if split.lower() in ('train',):
        train_path = os.path.join(data_dir, 'train.50k.json')
        source_data = pd.read_csv(open(train_path, 'r'))
    elif split.lower() in ('validation',):
        dev_path = os.path.join(data_dir, 'ScienceQA_test.json')
        source_data = json.load(open(dev_path, 'r'))
    elif split.lower() in ('test',):
        source_data = []
    else:
        raise ValueError(f"The split {split} is not supported. "
                         f"Current only supports 'train', 'dev', 'validation'")
        
    examples = []
    total_count = 0
    for instance in source_data:
        question = instance['question']
        choices = instance['choices']
        answer_true = instance['answer']
        lecture = instance['lecture']
        solution = instance['solution']
        if lecture=='' or solution=='':
            continue
        total_count += 1
        for idx, choice in enumerate(choices):
            # query_text = f"Lecture:{lecture}\n question：{question}\n anwser：{choice}\n This answer comes from the following reasoning：{solution}"
            # input_str = f"{query_text}"
            # prompt = f"Lecture:{lecture}\n question：{question}\n anwser：{choice}\n This answer comes from the following reasoning："
            
            # query_text = f"Lecture:{lecture}\n question：{question}\n anwser：{choice}\n"
            # input_str = f"{query_text}"
            # prompt = f"Lecture:{lecture}\n question：{question}\n anwser："
            
            query_text = f"Lecture:{lecture}\n anwser：{choice} can answer the following question：{question}"
            input_str = f"{query_text}"
            prompt = f"Lecture:{lecture}\n anwser：{choice} can answer the following question："

            
            if in_context:
                context1 = "Lecture:Birds, mammals, fish, reptiles, and amphibians are groups of animals. The animals in each group have traits in common. Scientists sort animals into groups based on traits they have in common. This process is called classification.\n question：Select the fish below.\n anwser：cardinalfish\n This answer comes from the following reasoning：A cardinalfish is a fish. It lives underwater. It has fins, not limbs. Cardinalfish often live near coral reefs. They are nocturnal, which means that they are active mostly at night. A black howler is a mammal. It has hair and feeds its young milk. Howler monkeys have loud calls, or howls. Their calls can be heard over three miles away!\n"
                input_str = context1 + input_str
                prompt = context1 + prompt
                
            examples.append({
                "idx": total_count,
                "input_str": input_str,
                "prompt": prompt,
                "is_correct": answer_true == idx,
            })
    return examples


def load_qa_dataset_ekar(data_dir, split, in_context, tokenizer):
    if split.lower() in ('train',):
        train_path = os.path.join(data_dir, 'train.50k.json')
        source_data = pd.read_csv(open(train_path, 'r'))
    elif split.lower() in ('validation',):
        dev_path = os.path.join(data_dir, 'test.json')
        # source_data = json.load(open(dev_path, 'r'))
        with open(dev_path, 'r', encoding="utf-8") as f:
            source_data = f.readlines()
    elif split.lower() in ('test',):
        source_data = []
    else:
        raise ValueError(f"The split {split} is not supported. "
                         f"Current only supports 'train', 'dev', 'validation'")
        
    examples = []
    total_count = 0
    id_map= {"A":0, "B":1, "C":2, "D":3}
    for instance in source_data:
        instance = json.loads(instance)
        question = instance['question']
        choices = instance['choices']['text']
        answer_true = instance['answerKey']
        explanation = instance['explanation']
        
        total_count += 1
        for idx, choice in enumerate(choices):
            src = explanation[0]
            tgt = explanation[idx+1]
            # query_text = f"根据推理关系:{question}\n可以得到类比推理关系为：{choice}\n因为{src},所以{tgt}"
            # input_str = f"{query_text}"
            # prompt = f"根据推理关系:{question}\n可以得到类比推理关系为：{choice}\n"
            
            query_text = f"根据推理关系:{question}\n可以得到类比推理关系为：{choice}"
            input_str = f"{query_text}"
            prompt = f"根据推理关系:{question}\n可以得到类比推理关系为："
            
            if in_context:
                context1 = "根据推理关系:轿车:汽车\n可以得到类比推理关系为：民法:法律\n因为“轿车”是一种“汽车”,所以“民法”是一种“法律”。\n"
                context2 = "根据推理关系:中国:上海\n可以得到类比推理关系为：澳大利亚:悉尼\n因为“上海”是“中国”的组成部分，“上海”不是“中国”的首都。,所以“悉尼”是“澳大利亚”的组成部分，“悉尼”不是“澳大利亚”的首都。\n"
                context3 = "根据推理关系:上市:停牌:退市\n可以得到类比推理关系为：听题:抢答:得分\n因为先“上市”，再“停牌”，后“退市”，三者属于时间顺承对应。,所以先“听题”，再“抢答”，最后“得分”，三者属于时间顺承对应。\n"
                context = context1 + context2 + context3
                input_str = context + input_str
                prompt = context + prompt
                
            examples.append({
                "idx": total_count,
                "input_str": input_str,
                "prompt": prompt,
                "is_correct": id_map[answer_true] == idx,
            })
    return examples


def load_qa_dataset_dureader(data_dir, split, in_context, tokenizer):
    if split.lower() in ('train',):
        train_path = os.path.join(data_dir, 'train.50k.json')
        source_data = pd.read_csv(open(train_path, 'r'))
    elif split.lower() in ('validation',):
        dev_path = os.path.join(data_dir, 'dev.json')
        source_data = json.load(open(dev_path, 'r'))
        # with open(dev_path, 'r', encoding="utf-8") as f:
            # source_data = f.readlines()
    elif split.lower() in ('test',):
        source_data = []
    else:
        raise ValueError(f"The split {split} is not supported. "
                         f"Current only supports 'train', 'dev', 'validation'")
        
    examples = []
    total_count = 0
    source_data = source_data['data'][0]["paragraphs"]
    for document in source_data:
        # instance = json.loads(instance)
        para = document['context'].strip()
        for qa in (document['qas']):
            total_count += 1
            query_id = qa['id']
            query_text = qa['question'].strip()
            answers = [a['text'] for a in qa['answers']]
            
            input_str = f"阅读文章:{para}\n问：{query_text}\n答："
            # input_str = f"{query_text}"
            prompt = ""
            
            if in_context:
                context1 = "阅读文章:据机场最新规定，充电宝超过2万毫安不得上飞机，要被机场没收，根据民航安全要求，旅客乘机时，充电宝等锂电池产品，其单块额定能量值最好不超过100Wh(瓦特小时)，且必须随身携带，不能托运。微博发布者根据“初中物理”计算得出，2万毫安的充电宝额定能量值“正好”是100Wh，因此不能上飞机。\n问：飞机上充电宝不能超过多少毫安\n答：2万毫安\n"
                context2 = "阅读文章:授予学位 ：管理学学士。会计学：培养德智体全面发展，掌握管理学、经济学、会计学、审计和财务管理等方面知识，具有会计、审计实务处理能力及财务管理能力，能在各类企事业单位和政府部门从事会计、审计和财务管理等实际工作，具有会计专业技能的应用型高级专门人才。\n问：会计是什么学士学位\n答：管理学\n"
                context3 = "阅读文章:这样根据厚度，色彩，使用场所，原材料，人工设备租赁等组合算出来的，看图，问问他家。|普通C15透水混凝土的价格大概在350左右，彩色的因需要用彩色水泥，价格可能会翻番。具体的也不是很清楚，上海盼石股份是专门从事这一行的一家企业，不妨去他们官网看看。\n问：透水混凝土价格\n答：350左右\n"
                context_all = context1 + context2 + context3
                input_str = context_all + input_str

            examples.append({
                "idx": query_id,
                "input_str": input_str,
                "prompt": prompt,
                "ans": answers,
                "is_correct": 1,
            })
            
    return examples


def load_qa_dataset_mbpp(data_dir, split, in_context, tokenizer):
    if split.lower() in ('train',):
        train_path = os.path.join(data_dir, 'train.50k.json')
        source_data = pd.read_csv(open(train_path, 'r'))
    elif split.lower() in ('validation',):
        dev_path = os.path.join(data_dir, 'dev.json')
        source_data = json.load(open(dev_path, 'r'))
        # with open(dev_path, 'r', encoding="utf-8") as f:
            # source_data = f.readlines()
    elif split.lower() in ('test',):
        source_data = []
    else:
        raise ValueError(f"The split {split} is not supported. "
                         f"Current only supports 'train', 'dev', 'validation'")
        
    examples = []
    total_count = 0
    
    for document in source_data:
        # instance = json.loads(instance)
        query = document['prompt'].strip()
        code = document['code']
        prompt = ""
        total_count+=1 
        
        if in_context:
            context1 = "Write a function to increment the numeric values in the given strings by k. def increment_numerics(test_list, K): res = [str(int(ele) + K) if ele.isdigit() else ele for ele in test_list] return res\n"
            context2 = "Write a function to check whether the given amount has no profit and no loss. def noprofit_noloss(actual_cost,sale_amount): if(sale_amount == actual_cost): return True else: return False\n"
            context = context1 + context2
            query = context + query
        examples.append({
            "idx": total_count,
            "input_str": query,
            "prompt": prompt,
            "ans": code,
            "is_correct": 1,
        })
    return examples


def load_dataset(dataset, data_url, split='validation', in_context=False, tokenizer=None):
    examples = []
    if dataset == 'c3':
        examples = load_qa_dataset_c3(data_url, split, in_context, tokenizer=tokenizer)
    elif dataset == 'ocnli' or dataset == "cmnli":
        examples = load_qa_dataset_ocnli(data_url, split, in_context, tokenizer=tokenizer)
    elif dataset == 'cosqa':
        examples = load_qa_dataset_cosqa(data_url, split, in_context, tokenizer=tokenizer)
    elif dataset == 'scienceqa':
        examples = load_qa_dataset_scienceqa(data_url, split, in_context, tokenizer=tokenizer)
    elif dataset == 'ekar':
        examples = load_qa_dataset_ekar(data_url, split, in_context, tokenizer=tokenizer)
    elif dataset == 'jecqa':
        examples = load_qa_dataset_jecqa(data_url, split, in_context, tokenizer=tokenizer)
    elif dataset == 'dureader':
        examples = load_qa_dataset_dureader(data_url, split, in_context, tokenizer=tokenizer)
    elif dataset == 'mbpp':
        examples = load_qa_dataset_mbpp(data_url, split, in_context, tokenizer=tokenizer)
    elif dataset == 'math':
        examples = load_qa_dataset_math(data_url, split, in_context, tokenizer=tokenizer)
    elif dataset == 'logic':
        examples = load_qa_dataset_logic(data_url, split, in_context, tokenizer=tokenizer)
    elif dataset == 'copa':
        examples = load_qa_dataset_copa(data_url, split, in_context, tokenizer=tokenizer)
    else:
        raise ValueError(f"The eval task {dataset} is not supported now. Currently only support c3.")

    return examples


def get_c3_metric(examples):
    metric = {"top1_acc": 0}
    acc_top1 = 0
    total_count = 0
    score_on_each_example = defaultdict(list)
    for item in examples:
        predicted = item['predict']  # should be score
        idx = item['idx']
        score_on_each_example[idx].append((predicted, item['is_correct']))

    for k in score_on_each_example.keys():
        made_choices = score_on_each_example[k]
        predicted_choice = min(made_choices, key=lambda x: x[0])
        if predicted_choice[1]:
            acc_top1 += 1
        total_count += 1
    metric['top1_acc'] = acc_top1 / total_count
    return metric


def get_math_metric(examples):
    metric = {"top1_acc": 0}
    f_math = open('math_eval.txt', 'w')
    acc_top1 = 0
    total_count = 0
    score_on_each_example = defaultdict(list)
    for item in examples:
        predicted = item['predict']  # should be score
        idx = item['idx']
        score_on_each_example[idx].append((predicted, item['is_correct']))
    for k in score_on_each_example.keys():
        ppl_diff = 0.0
        made_choices = score_on_each_example[k]
        predicted_choice = min(made_choices, key=lambda x: x[0])
        if predicted_choice[1]:
            acc_top1 += 1
            sorted_choices = sorted(made_choices, key=lambda x: x[0])
            ppl_diff = sorted_choices[1][0] - predicted_choice[0]
            f_math.write('true: '+str(ppl_diff)+'\n')
        else:
            sorted_choices = sorted(made_choices, key=lambda x: x[0])
            minPPL = predicted_choice[0]
            for choice in sorted_choices:
                if choice[1]:
                    truePPL = choice[0]
                    ppl_diff = truePPL - minPPL
                    f_math.write('false: '+str(ppl_diff)+'\n')
                    break
        total_count += 1
    metric['top1_acc'] = acc_top1 / total_count
    return metric


# add f1/em from baidu https://github.com/baidu/DuReader/blob/master/DuReader-Robust/evaluate.py#L188
def _tokenize_chinese_chars(text):
    """
    :param text: input text, unicode string
    :return:
        tokenized text, list
    """

    def _is_chinese_char(cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    output = []
    buff = ""
    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp) or char == "=":
            if buff != "":
                output.append(buff)
                buff = ""
            output.append(char)
        else:
            buff += char

    if buff != "":
        output.append(buff)

    return output

def _normalize(in_str):
    """
    normalize the input unicode string
    """
    in_str = in_str.lower()
    sp_char = [
        u':', u'_', u'`', u'，', u'。', u'：', u'？', u'！', u'(', u')',
        u'“', u'”', u'；', u'’', u'《', u'》', u'……', u'·', u'、', u',',
        u'「', u'」', u'（', u'）', u'－', u'～', u'『', u'』', '|'
    ]
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


def find_lcs(s1, s2):
    """find the longest common subsequence between s1 ans s2"""
    m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
    max_len = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i+1][j+1] = m[i][j]+1
                if m[i+1][j+1] > max_len:
                    max_len = m[i+1][j+1]
                    p = i+1
    return s1[p-max_len:p], max_len

def calc_f1_score(answers, prediction):
    f1_scores = []
    for ans in answers:
        ans_segs = _tokenize_chinese_chars(_normalize(ans))
        prediction_segs = _tokenize_chinese_chars(_normalize(prediction))
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        prec = 1.0*lcs_len/len(prediction_segs)
        rec = 1.0*lcs_len/len(ans_segs)
        f1 = (2 * prec * rec) / (prec + rec)
        f1_scores.append(f1)
    return max(f1_scores)


def calc_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = _normalize(ans)
        prediction_ = _normalize(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em
# add f1/em end


def get_gen_metric(examples):
    metric = {"f1": 0, "em": 0}
    f1 = 0
    em = 0
    total_count = 0
    skip_count = 0
    
    for item in examples:
        total_count += 1
        idx = item['idx']
        answers = item['ans']
        try:
            prediction = item['predict'].strip()  # should be text
        except:
            skip_count += 1
            print("Skipped")
            print('----------------------------')
            continue
        _f1 = calc_f1_score(answers, prediction)
        f1 += _f1
        em += calc_em_score(answers, prediction)
        
    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count

    metric['f1'] = f1_score
    metric['em'] = em_score
    return metric



def load_metric(dataset):
    if dataset in ('c3', 'ocnli', 'cmnli', 'cosqa', 'scienceqa', 'ekar', 'ocnli', 'jecqa', 'logic', 'copa'):
        return get_c3_metric
    elif dataset in ('math'):
        return get_math_metric
    elif dataset in ('dureader', 'mbpp'):
        return get_gen_metric
    raise ValueError(f"The input dataset {dataset} not found in the list ['c3', 'ocnli', 'cmnli', 'cosqa', 'scienceqa', 'ekar', 'dureader', 'jecqa', 'mbpp', 'math', 'copa']")
