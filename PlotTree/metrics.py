import os
import json
import argparse

import numpy as np
import pandas as pd
from scipy import spatial


import re

def extract_options(input_string):
    # Define a regex to match (A)、(B)、(C)、(D)、(E)
    pattern = r'\(([A-E])\)'  # Match only A to E
    matches = re.findall(pattern, input_string)  # Find all matches
    return matches  
    
def extract_options_is(input_string):
    # Define a regex to match (A)、(B)、(C)、(D)、(E)
    pattern_parentheses = r'\(([A-E])\)'  
    # Define a regex to match "option is A/B/C..."
    pattern_option_is = r'option is ([A-E])'  

    # Find all matches
    matches_parentheses = re.findall(pattern_parentheses, input_string)
    matches_option_is = re.findall(pattern_option_is, input_string)

    if matches_option_is:
        matches_parentheses.extend(matches_option_is)

    return matches_parentheses  # 返回匹配项列表

def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        json_str = file.read()
    data = json.loads(json_str)
    return data

def save_json(fn,data, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)

        
def extract_dict_key(json_data, extract_keys=['id', 'vid', 'option', 'VILAMP_answer']):
    temp_dict = dict()
    for k in extract_keys:
        temp_dict[k] = []
    temp_dict['times'] = []
    temp_dict['type'] = []
    temp_dict['score'] = []
    accumulative_score = 0
    for q_dict in json_data:
        if not extract_keys[-1] in q_dict:
            break
        for k in extract_keys:
            temp_dict[k].append(q_dict[k])
        
        if q_dict['option']==q_dict[extract_keys[-1]]:
            accumulative_score = 1
        else:
            accumulative_score = 0
        temp_dict['type'].append(f"{q_dict['question_type']}-{q_dict['story_element']}")
        temp_dict['times'].append(q_dict['times'])
        temp_dict['score'].append(accumulative_score)
        
    return temp_dict



info_dict = { 
    "Models": [],
    "GOT": [],
    "BigBang": [],
    "Friends": [],
    "Movie": [],
}


def check_answer_json(results_json, model_name, ini_model_name=None):
    score_dict ={   
                    'P-C': 0,
                    'P-A': 0,
                    'P-L': 0,
                    'P-CA': 0,
                    'P-CL': 0,
                    'P-AL': 0,
                    'P-CAL': 0,
                    'I-C': 0,
                    'I-A': 0,
                    'I-L': 0,
                    'I-CA': 0,
                    'I-CL': 0,
                    'I-AL': 0,
                    'I-CAL': 0,
                    'P-TV':0,
                    'I-TV':0,
                    'P-Movie':0,
                    'I-Movie':0,
                    'TV':0,
                    'Movie': 0,
                    'Total': 0
                }
    question_num = {
                    'P-C': 0,
                    'P-A': 0,
                    'P-L': 0,
                    'P-CA': 0,
                    'P-CL': 0,
                    'P-AL': 0,
                    'P-CAL': 0,
                    'I-C': 0,
                    'I-A': 0,
                    'I-L': 0,
                    'I-CA': 0,
                    'I-CL': 0,
                    'I-AL': 0,
                    'I-CAL': 0,
                    'P-TV':0,
                    'I-TV':0,
                    'P-Movie':0,
                    'I-Movie':0,
                    'TV': 0,
                    'Movie': 0,
                    'Total': 0}
    answer_dicts = dict()
    if model_name in ['SINGULARITY','VIOLETv2','Vid-TLDR','VideoChatGPT','SeViLA']:
        # Process model that directly returns the answer
        for q_dict in results_json:
            vid_dir = q_dict['vid'].split("-")[0] if q_dict['vid'].split("-")[0] in ['Friends','GOT','BigBang'] else 'Movie' 
            if q_dict['GT'] == q_dict[f'{model_name}_answer']:
                score_dict['Total'] +=1
                score_dict[q_dict['question_type']+"-"+q_dict['story_element']] += 1
                score_dict['TV'] = score_dict['TV']+1 if vid_dir != 'Movie' else score_dict['TV']
                if vid_dir in ['Friends','GOT','BigBang']:
                    score_dict[q_dict['question_type']+"-"+'TV'] += 1
                else:
                    score_dict[q_dict['question_type']+"-"+'Movie'] += 1

            question_num[q_dict['question_type']+"-"+q_dict['story_element']] += 1
            question_num['TV'] = question_num['TV'] + 1 if vid_dir != 'Movie' else question_num['TV']
            question_num['Total'] +=1
            if vid_dir in ['Friends','GOT','BigBang']:
                question_num[q_dict['question_type']+"-"+'TV'] += 1
            else:
                question_num[q_dict['question_type']+"-"+'Movie'] += 1
            answer_dicts[vid_dir+ f"-{q_dict['id']}"] = q_dict[f'{model_name}_answer']
    elif model_name in ['VILAMP','PlotTree', 'PlotTree_wo_plot', 'VideoTree','Video2RAG']:
        # Process model that directly returns options
        for q_dict in results_json:
            vid_dir = q_dict['vid'].split("-")[0] if q_dict['vid'].split("-")[0] in ['Friends','GOT','BigBang'] else 'Movie' 
            if q_dict['option'] == q_dict[f'{model_name}_answer'] or q_dict['option'] == q_dict[f'{model_name}_answer'][0]:
                score_dict['Total'] +=1
                score_dict[q_dict['question_type']+"-"+q_dict['story_element']] += 1
                score_dict['TV'] = score_dict['TV']+1 if vid_dir != 'Movie' else score_dict['TV']
                if vid_dir in ['Friends','GOT','BigBang']:
                    score_dict[q_dict['question_type']+"-"+'TV'] += 1
                else:
                    score_dict[q_dict['question_type']+"-"+'Movie'] += 1
            question_num[q_dict['question_type']+"-"+q_dict['story_element']] += 1
            question_num['TV'] = question_num['TV'] + 1 if vid_dir != 'Movie' else question_num['TV']
            question_num['Total'] +=1
            if vid_dir in ['Friends','GOT','BigBang']:
                question_num[q_dict['question_type']+"-"+'TV'] += 1
            else:
                question_num[q_dict['question_type']+"-"+'Movie'] += 1
            answer_dicts[vid_dir+ f"-{q_dict['id']}"] = q_dict[f'{model_name}_answer']
                
    elif model_name in ['VideoChat2']:
        # Process model that directly return a model of A),B),..., and take the first letter as the option.
        for q_dict in results_json:
            vid_dir = q_dict['vid'].split("-")[0] if q_dict['vid'].split("-")[0] in ['Friends','GOT','BigBang'] else 'Movie' 
            if q_dict['option'] == q_dict[f'{model_name}_answer'][0]:
                score_dict['Total'] +=1
                score_dict[q_dict['question_type']+"-"+q_dict['story_element']] += 1
                score_dict['TV'] = score_dict['TV']+1 if vid_dir != 'Movie' else score_dict['TV']
                if vid_dir in ['Friends','GOT','BigBang']:
                    score_dict[q_dict['question_type']+"-"+'TV'] += 1
                else:
                    score_dict[q_dict['question_type']+"-"+'Movie'] += 1
            question_num[q_dict['question_type']+"-"+q_dict['story_element']] += 1
            question_num['TV'] = question_num['TV'] + 1 if vid_dir != 'Movie' else question_num['TV']
            question_num['Total'] +=1
            if vid_dir in ['Friends','GOT','BigBang']:
                question_num[q_dict['question_type']+"-"+'TV'] += 1
            else:
                question_num[q_dict['question_type']+"-"+'Movie'] += 1
            answer_dicts[vid_dir+ f"-{q_dict['id']}"] = q_dict[f'{model_name}_answer']
    elif model_name in ['videollama3','ChatUniVi','MALMM','TimeChat','VideoLLaMA2','Video-XL']:
        # Process model that directly return a model of (A),(B),..., and take the first letter as the option.
        for q_dict in results_json:
            vid_dir = q_dict['vid'].split("-")[0] if q_dict['vid'].split("-")[0] in ['Friends','GOT','BigBang'] else 'Movie' 
            extract_answers = extract_options_is(q_dict[f'{model_name}_answer'])
            if len(extract_answers)>0 and q_dict['option'] == extract_options_is(q_dict[f'{model_name}_answer'])[0]:
                score_dict['Total'] +=1
                score_dict[q_dict['question_type']+"-"+q_dict['story_element']] += 1
                score_dict['TV'] = score_dict['TV']+1 if vid_dir != 'Movie' else score_dict['TV']
                if vid_dir in ['Friends','GOT','BigBang']:
                    score_dict[q_dict['question_type']+"-"+'TV'] += 1
                else:
                    score_dict[q_dict['question_type']+"-"+'Movie'] += 1
            question_num[q_dict['question_type']+"-"+q_dict['story_element']] += 1
            question_num['TV'] = question_num['TV'] + 1 if vid_dir != 'Movie' else question_num['TV']
            question_num['Total'] +=1
            if vid_dir in ['Friends','GOT','BigBang']:
                question_num[q_dict['question_type']+"-"+'TV'] += 1
            else:
                question_num[q_dict['question_type']+"-"+'Movie'] += 1
            answer_dicts[vid_dir+ f"-{q_dict['id']}"] = q_dict[f'{model_name}_answer']
    
    for key in score_dict:
        score_dict[key] = score_dict[key]/question_num[key]*100 if question_num[key]>0 else -1
    return score_dict, answer_dicts
        

if __name__ == '__main__':
    baseline_dirs = {
        'PlotTree-36-10-32': 'results/QA/Gemini-2.0-flash_36_10.0_32', # default settring of PlotTree
    } 

    result_dict = dict()
    ini_model_name = ""
    for model_name in baseline_dirs.keys():
        base_dir = baseline_dirs[model_name]
        info_dict['Models'].append(model_name)
        questions = []
        size = 0
        ini_model_name = model_name
        if 'PlotTree' in model_name:
            model_name = 'PlotTree'
        if 'Video2RAG' in model_name:
            model_name = 'Video2RAG'
        for vid_dir in ['GOT', 'BigBang', 'Friends', 'Movie']:
            # Fully Automatic full set
            # if os.path.exists(f"{base_dir}/{vid_dir}_{model_name}.json"):
            #     questions += load_json(f"{base_dir}/{vid_dir}_{model_name}.json")
            # The manually annotated correct questions and the extracted overlapping parts
            # if os.path.exists(f"{base_dir}/{vid_dir}_{model_name}_correctv1.json"):
            #     questions += load_json(f"{base_dir}/{vid_dir}_{model_name}_correctv1.json")
            #     print(f"Loaded {base_dir}/{vid_dir}_{model_name}_correctv1.json", len(questions))
            if model_name == 'VideoTree' and os.path.exists(f"{base_dir}/{vid_dir}_{model_name}.json"):
                questions += load_json(f"{base_dir}/{vid_dir}_{model_name}.json")
                # print(f"Loaded {base_dir}/{vid_dir}_{model_name}", len(questions))
            if model_name in['Video2RAG', "PlotTree"] and os.path.exists(f"{base_dir}/{vid_dir}-{model_name}.json"):
                questions += load_json(f"{base_dir}/{vid_dir}-{model_name}.json")
        score_dict, answer_dicts = check_answer_json(questions, model_name, ini_model_name)
        print(ini_model_name, len(questions))
        result_dict[ini_model_name] = score_dict

    result_df = pd.DataFrame(result_dict).T
    print(result_df)