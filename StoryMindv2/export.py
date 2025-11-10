import os
import re
import json
import pandas as pd
import argparse
import numpy as np



def parse_check(file_path):
    check_list = []
    if not os.path.exists(file_path):
        return check_list
    temp = 0
    with open(file_path, 'r') as f:
        eval_str = f.read().strip()
        eval_list = eval_str.split("\n")
        for line in eval_list:
            line_split = line.split(" ")            
            while temp+1 != eval(line_split[0]):
                temp_line= [temp+1, 'False', 'XXX']
                check_list.append(temp_line)
                temp = temp+1
            if len(line_split)<3:
                line_split.append("XXX")
            selected_answer = " ".join(line_split[2:])
            line_split = [line_split[0], line_split[1], selected_answer]

            temp = eval(line_split[0])
            check_list.append(line_split)
    return check_list

def get_answer_option(answer_list, GT):
    for i, ans in enumerate(answer_list):
        if ans == GT:
            return chr(65+i)  # 65表示字符'A'
    return None

def check_same_num(choices, target):
    num = 0
    for choice in choices:
        if choice == target:
            num+=1
    return num

def auto_check_correct(q_dict):
    if not 'gemini_check' in q_dict or not 'claude_check' in q_dict  or  not 'gpt_check' in q_dict:
        print("Fatal Error")
    gemini_check = q_dict['gemini_check']
    claude_check = q_dict['claude_check']
    gpt_check = q_dict['gpt_check']
    gemini_check[1] = gemini_check[1].strip("'").strip('"')
    claude_check[1] = claude_check[1].strip("'").strip('"')
    gpt_check[1] = gpt_check[1].strip("'").strip('"')
    print([gemini_check[0], claude_check[0], gpt_check[0]], 'True')
    if check_same_num([gemini_check[0], claude_check[0], gpt_check[0]], 'True') < 3:
        return False
    if check_same_num([gemini_check[1] , claude_check[1], gpt_check[1]], q_dict['GT']) <2:
        return False
    return True

def filter_questions(questions_list):
    filtered_questions = []
    for q_dict in questions_list:
        if auto_check_correct(q_dict):
            filtered_questions.append(q_dict)
    return filtered_questions 

def parser():
    parser = argparse.ArgumentParser("StoryMind's Export", add_help=True)
    parser.add_argument("--vid_dir", type=str, default="Friends", help="csv directory")
    parser.add_argument("--output_path", type=str, default="json/filter_QAs.json", help="Export file path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parser()
    single_episode_questions = []
    questions_dir = f"csv/{args.vid_dir}"
    questions = []
    all_dialogs = dict()
    Nomatch = 0
    for filename in sorted(os.listdir(questions_dir))[:]:
        print(filename)
        if not '.csv' in filename:
            continue
        vid = filename.split(".")[0]
        save_csv_path = os.path.join(questions_dir, filename)
        question_csv = pd.read_csv(save_csv_path)
        video_name = vid
        claude_check_list = parse_check(f'check/claude/result_{vid}.txt')
        gemini_check_list = parse_check(f'check/gemini/result_{vid}.txt')
        gpt_check_list = parse_check(f'check/gpt/result_{vid}.txt')
        print(video_name)
        if len(claude_check_list) == 0 and len(gemini_check_list) ==0 and len(gpt_check_list)==0:
            print("Not check")
        else:
            if len(set([len(question_csv), len(gemini_check_list), len(claude_check_list), len(gpt_check_list )]))!=1:
                print('============================================fatal error, not match check list sizes=================================')
                print(len(claude_check_list),len(gemini_check_list), len(gpt_check_list),len(question_csv))

        print(len(claude_check_list),len(gemini_check_list), len(gpt_check_list),len(question_csv))
        if len(claude_check_list) == 0 or len(gemini_check_list) ==0 or len(gpt_check_list)==0:
            print("TBD")
        for i in range(len(question_csv)):
            question = question_csv.loc[i,'question']
            choices = eval(question_csv.loc[i,'choices_list'])
            video_name = question_csv.loc[i,'vid']
            answer_list = eval(question_csv.loc[i,'choices_list'])
            GT = question_csv.loc[i,'gt']
            option = get_answer_option(answer_list, GT)
            if len(claude_check_list) == 0 or len(gemini_check_list) ==0 or len(gpt_check_list)==0:
                check_g = ["TBD", "TBD"]
                check_c = ["TBD", "TBD"]
                check_o = ["TBD", "TBD"]
            else:
                check_g = gemini_check_list[i][1:]
                check_c = claude_check_list[i][1:]
                check_o = gpt_check_list[i][1:]
                # print(i)
                if check_g[-1] == 'A' or check_g[-1] == 'B' or check_g[-1] == 'C' or check_g[-1] == 'D' or check_g[-1] == 'E':
                    check_g[-1] = answer_list[ord(check_g[-1])-ord('A')] if (ord(check_g[-1])-ord('A')) < len(answer_list) else "XXX"
                if check_c[-1] == 'A' or check_c[-1] == 'B' or check_c[-1] == 'C' or check_c[-1] == 'D' or check_c[-1] == 'E':
                    check_c[-1] = answer_list[ord(check_c[-1])-ord('A')] if (ord(check_c[-1])-ord('A')) < len(answer_list) else "XXX"
                if check_o[-1] == 'A' or check_o[-1] == 'B' or check_o[-1] == 'C' or check_o[-1] == 'D' or check_o[-1] == 'E':
                    check_o[-1] = answer_list[ord(check_o[-1])-ord('A')] if (ord(check_o[-1])-ord('A')) < len(answer_list) else "XXX"
            if not option:
                print(f'Does not match, {answer_list}, {GT}')
                Nomatch+=1
                continue  # 答案不在选项里面，去掉这个题目
            
            q_dict = {
                'id': len(single_episode_questions)+1,
                'vid': video_name,
                'question': question, 
                'choices': eval(question_csv.loc[i,'choices_list']),
                'GT': question_csv.loc[i,'gt'],
                'option': option,
                'gemini_check': check_g,
                'claude_check': check_c,
                'gpt_check': check_o,
                'question_type': question_csv.loc[i,'question_type'],
                'element': question_csv.loc[i,'element'],
                'characters': eval(question_csv.loc[i,'related_person']),	
                'locations': eval(question_csv.loc[i,'related_location']),
                'times': eval(question_csv.loc[i,'related_times']), 
            }
            single_episode_questions.append(q_dict)
    print("Delete not match questions:", Nomatch)

    if not os.path.exists("json"):
        os.mkdir("json")
    # 将列表写入到json文件中
    with open(f'json/ini_QAs.json', 'w') as f:
        json.dump(single_episode_questions, f, indent=4)
    filter_question_list = filter_questions(single_episode_questions)
    with open(f'json/filter_QAs.json', 'w') as f:
        json.dump(filter_question_list, f, indent=4)
    print(len(single_episode_questions), "=>", len(filter_question_list))