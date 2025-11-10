import torch
import argparse
from transformers import AutoTokenizer, AutoModel
import os
import cv2
import json
import pandas as pd
from tqdm import tqdm
import re
from bert_score import score
import numpy as np

tokenizer = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

import numpy as np

# 分组统计
def unique_count(series):
    # print(len([item for sublist in series for item in sublist]), '->', len(set(item for sublist in series for item in sublist)))
    # print(set(item for sublist in series for item in sublist))
    return len(set(item for sublist in series for item in sublist))

# 计算平均数量
def average_count(series):
    total_count = sum(len(sublist) for sublist in series)
    return total_count / len(series) if len(series) > 0 else 0

def average_num(series):
    total_count = sum(num for num in series)
    return total_count/len(series) if len(series) > 0 else 0

# 计算标准差
def std_deviation(series):
    avg = average_num(series)
    n = len(series)
    variance = sum((num - avg) ** 2 for num in series) / n
    return variance ** 0.5  # 对方差求平方根

# 应用 Sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 定义换底函数
def log(base, x):
    return np.log(x) / np.log(base)

def entropy(x):
    return -1* np.log(x)*x - (1-x) * np.log(1-x)



def get_bert_embeddings(texts, model_name="roberta-large", cache_dir='ckpt'):
    # Load tokenizer and model
    global tokenizer, model
    if not tokenizer and not model:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        # Move model to GPU if available
        model.to(device)
    # Process texts in batch
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    # Get model output
    with torch.no_grad():
        outputs = model(**encoded_input)
    # Use embeddings from the last layer
    embeddings = outputs.last_hidden_state
    # Remove padding tokens
    attention_mask = encoded_input['attention_mask']
    embeddings = [emb[mask.bool()] for emb, mask in zip(embeddings, attention_mask)]
    return embeddings

def token_cosine_similarity(embeddings1, embeddings2):
    # Normalize embeddings for cosine similarity
    embeddings1_norm = embeddings1 / embeddings1.norm(dim=1, keepdim=True)
    embeddings2_norm = embeddings2 / embeddings2.norm(dim=1, keepdim=True)
    similarity_matrix = torch.matmul(embeddings1_norm, embeddings2_norm.transpose(0, 1))
    return similarity_matrix







def calculate_bertscore(candidate_embeddings, reference_embeddings_list):
    # List to store precision, recall, f1 scores for each reference set
    precision_list = []
    recall_list = []
    f1_list = []
    for reference_embeddings in reference_embeddings_list:
        # Compute similarity matrix
        sim_matrix = token_cosine_similarity(candidate_embeddings, reference_embeddings)
        
        # Compute precision (max similarity for each candidate token)
        precision = sim_matrix.max(dim=1)[0].mean().item()
        
        # Compute recall (max similarity for each reference token)
        recall = sim_matrix.max(dim=0)[0].mean().item()
        
        # Compute F1
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        # Append the scores to the list
        precision_list.append(precision) 
        recall_list.append(recall)
        f1_list.append(f1)
        
    return precision_list, recall_list, f1_list




def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        json_str = file.read()
    data = json.loads(json_str)
    return data

def save_json(fn,data, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)

def time_to_seconds(time_str):
    time_parts = time_str.split(':')
    if len(time_parts) != 3:
        raise ValueError("时间格式不正确，应该是 HH:MM:SS")
    
    hours = int(time_parts[0])
    minutes = int(time_parts[1])
    seconds = int(time_parts[2])
    
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds

def extract_times(input_str):
    # 使用正则表达式匹配 HH:MM:SS 格式
    time_pattern = r'\d{2}:\d{2}:\d{2}'
    matches = re.findall(time_pattern, input_str)
    # 将匹配到的时间转换为秒，并加入到列表中
    seconds_list = [time_to_seconds(time) for time in matches]
    return seconds_list

def calculate_entropy(F1):
    # 确保 F1 中的元素是有效的概率 (0, 1)
    if np.any(F1 < 0) or np.any(F1 > 1):
        raise ValueError("F1 中的值必须在 0 和 1 之间")
    
    # 将 F1 的值视为概率，并计算熵
    F1 = F1[F1 > 0]  # 过滤掉 0 的概率值，以避免 log(0)
    entropy = -np.sum(F1 * np.log(F1))
    
    return entropy


def calculate_entropy(similarities):
    # 将相似度值转化为概率
    probabilities = similarities / similarities.sum()
    
    # 计算熵
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))  # +epsilon防止log(0)
    
    return entropy



def extract_qdict_info(q_dict,video_length_dict):
    qid = q_dict['id']
    vid = q_dict['vid']
    question = q_dict['question']
    choices = q_dict['choices']
    answer_index = ord(q_dict['option']) - ord('A') 
    P, R, F1 = score([question], [choices[answer_index]],lang="en", verbose=False)
    question_answer_score = F1.mean().item()
    choices1, choices2 =[],[]
    for i in range(len(choices)):
        if i != answer_index:
            choices1.append(choices[answer_index])
            choices2.append(choices[i])
    # print(choices1, choices2)
    P, R, F1 = score(choices1, choices2,lang="en", verbose=False)
    answer_score = F1.mean().item()
    choice_list = F1.tolist()



    # print(f"System level F1 score: {F1.mean():.3f}")
    # 'characters': ['Arya', 'Sansa'], 'locations': ['Winterfell'], 'times'
    character = q_dict['characters'] if type(q_dict['characters'])==list else []
    locations = q_dict['locations'] if type(q_dict['locations'])==list else []
    times = q_dict['times'] if type(q_dict['times'])==list else []
    question_type = q_dict['question_type']
    element = q_dict['element']
    time_seconds = []
    for span in times:
        span_list = extract_times(span)
        if len(span_list)>2:
            print(qid, vid, span_list)
            for i in range(0, len(span_list) - 1, 2):
                pair = [span_list[i], span_list[i + 1]]  # 两两组合
                time_seconds.append(pair)  # 添加到 time_seconds 列表中
            # print(time_seconds)
        else:
            time_seconds.append(span_list)
            
    
    return {'id': qid, 
            'vid':vid,
            'question': question,
            'choices': choices,
            'GT': q_dict['GT'],
            'question_type':question_type,
            'story_element': element,  
            'character':character, 
            'location':locations, 
            'times':time_seconds, 
            'character_num': len(character),
            'location_num': len(locations),
            'span': sum([span_list[1]-span_list[0] if len(span_list)>=2 else 0 for span_list in time_seconds ]),
            'duration': video_length_dict[vid],
            'question_answer_score': question_answer_score,
            'answer_score' : answer_score,
            'choices_list': choice_list
        }

def get_video_duration(video_path):
    """获取视频文件的时长（以秒为单位）"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return 0
    
    # 获取视频的帧数和帧率
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 计算时长（秒）
    duration = frame_count / fps if fps > 0 else 0
    cap.release()  # 释放视频文件
    return duration



def parser():
    parser = argparse.ArgumentParser("StoryMind's Difficulty Measure", add_help=True)
    parser.add_argument("--questions_path", type=str, default="json/filter_QAs.json", help="QAs json path")
    parser.add_argument("--output_path", type=str, default="json/all_questions_info_with_difficulty.json", help="QAs json with diff")
    return parser.parse_args()


if __name__ == "__main__":
    args = parser()
    questions = load_json(args.questions_path)
    questions_dict = dict()
    video_length_dict = dict()
    video_length_dict = load_json("aligned_script/video_length.json")
    for q_dict in tqdm(questions):
        questions_dict[f"{q_dict['id']}"] = extract_qdict_info(q_dict, video_length_dict)

    # save_json("json/all_questions_info.json" ,questions_dict)


    common_questions = pd.DataFrame(questions_dict).T
    common_questions["choices_mean"] = common_questions["choices_list"].apply(np.mean)
    mean_span = common_questions[common_questions['span'] > 0]['span'].mean()
    print(f"所有非零 span 的均值为: {mean_span:.2f}")
    # 2. 使用均值替换 0
    common_questions['span'] = common_questions['span'].replace(0, mean_span)
    print("处理后 'span' 列中0的数量:", len(common_questions[common_questions['span'] == 0]))
    common_questions['content'] = common_questions['character_num'] + common_questions['location_num']
    cl_count = common_questions.groupby('vid').agg(
        character_count=('character', unique_count),
        location_count=('location', unique_count),
        avg_character=('character_num', average_num),
        avg_location=('location_num', average_num),
        avg_content =('content', average_num),
        avg_time=('span', average_num),
        std_character=('character_num', std_deviation),
        std_location=('location_num', std_deviation),
        std_time=('span', std_deviation),
        std_content= ('content', std_deviation),
    )

    common_questions_info = common_questions.join(cl_count, on='vid', how='left')

    common_questions_info['content_diff'] = (common_questions_info['content']-common_questions_info['avg_content'])/common_questions_info['std_content']
    common_questions_info['duration_diff'] =(common_questions_info['span']-common_questions_info['avg_time'])/common_questions_info['std_time']
    common_questions_info['content_diff'] = 1 - common_questions_info['content_diff'].apply(sigmoid)
    common_questions_info['duration_diff'] = 1 - common_questions_info['duration_diff'].apply(sigmoid)

    col_mean = np.mean(common_questions_info['answer_score'])   # 整列的均值
    col_std  = np.std(common_questions_info['answer_score'])    # 整列的标准差（总体标准差）
    common_questions_info['choices_score_remap'] = (common_questions_info['answer_score']- col_mean )/col_std
    common_questions_info['question_choices_score_remap'] = (common_questions_info['question_answer_score']-min(common_questions_info['question_answer_score']))/(max(common_questions_info['question_answer_score'])-min(common_questions_info['question_answer_score']))
    common_questions_info['q_diff'] = (common_questions_info['content_diff']+common_questions_info['duration_diff'])/2
    common_questions_info['c_diff'] = 1-common_questions_info['choices_score_remap'].apply(sigmoid).apply(entropy)
    common_questions_info['qc_diff'] = 1-common_questions_info['question_choices_score_remap']
    common_questions_info['diff'] = common_questions_info['q_diff'] + common_questions_info['qc_diff']  + common_questions_info['c_diff']
    common_questions_info.to_csv("Full_dataset_diff.csv")
   

    for qid, row in common_questions_info.iterrows():
        if qid in questions_dict:
            questions_dict[qid]['difficulty'] = float(row['diff'])  # 确保可以JSON序列化
        else:
            print(f"Warning: {qid} not found in questions_dict")

    save_json(args.output_path, questions_dict, indent=4)

