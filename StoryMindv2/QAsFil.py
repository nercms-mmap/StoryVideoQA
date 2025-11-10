import csv
import re
import pandas as pd
import os
import time
import argparse
import traceback
import numpy as np
from utils.engine import script_to_str, prompt_engine
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser


def choices_str(choice_list):
    choices = ""
    for i, choice in enumerate(choice_list):
        choices += "(" + chr(65+i)+")" + f" {choice} "    # 65表示字符'A'
    return choices

def getQuestion(csv_path):
    Questions = ""
    Questions_list = []
    question_csv = pd.read_csv(csv_path)
    for i in range(len(question_csv)):
        question = question_csv.loc[i,'question']
        choice_str = choices_str(eval(question_csv.loc[i,'choices_list']))
        choice_str = choice_str.strip()
        answer = question_csv.loc[i,'gt']
        Questions += f"{i+1}; {question}; {choice_str}\n"
        Questions_list.append(f"{i+1}; {question}; {choice_str}")
    last_id = len(question_csv)
    return last_id, Questions, Questions_list


def getPrompt(start_id, last_id, video_info, generate_questions):
    prompt =f"""System: 
You are very good at reviewing QAs for correctness and answering given QAs. Here is a video description for a movie or TV show, and the corresponding test QAs for understanding the content of the movie or TV show. You are asked to evaluate each QA-pair, assessing its correctness (True or False). Assuming you are actually doing the test and can only watch the video, you only need to give the corresponding assessment in the order in which they are presented. 

Video description are as follow:
{video_info}   

Generated questions are as follow (Each line is separated by ` `):
{generate_questions}

Correctness Requirements: 
For each QA-pair, determine if the answer is correct based solely on the provided video description, using the following guidelines:
(1) Correctness: The answer must be correct based solely on the provided video description. The answer must be explicitly stated in the text or can be directly and logically inferred from the text.
(2) No Prior Knowledge: Do not use any prior knowledge. The QAs must be answerable after watching the video without any prior knowledge. No Assumptions: Do not make any assumptions.
(3) Example of 'True': If a question asks, 'What is the name of the character with a beard?' and the text says, '...a man with a beard named Hagrid...', the answer is 'True'. Example of 'False': If a question asks, 'What is the name of Harry’s pet owl?' and the text only says, '...Harry has an owl...', the answer is 'False' because the owl’s name is not given. Respond with 'True' if the answer meets all criteria, and 'False' if it does not. There’re about 20% incorrect questions totally, you need to find out them exactly. 
There're about 20% incorrect questions totally, you need to find out them exactly. 

Answer Requirements:
You need to select an answer from choices list for each QA-pair based solely on the provided video description. If it’s a wrong QA-pair or there are not any supported evidence in video description to select an answer, please respond ‘XXX’ as placeholder.


## Output
Each line is about the review result of these questions, including 4 elements, separated by ` `: id of the question, Correctness of question (True or False), Correct answer (select a correct answer from Choices list.).

### output examples:
1 True A
2 False XXX
3 True B
4 True XXX
5 True C
...

Please directly give the reviewing result according to the id from {start_id} to {last_id} without any explanation.:
Id Correctness Answer
"""
    return prompt




def parser():
    parser = argparse.ArgumentParser("StoryMind's Reviewer", add_help=True)
    # GPT
    parser.add_argument("--openai_model", default="gpt-4o",type=str, help="model name of GPT")
    parser.add_argument("--openai_key",  required=True, type=str, help="key for chatgpt")
    parser.add_argument("--openai_proxy",  required=True, type = str, help="api key of chatgpt") 
    # Gemini
    parser.add_argument("--gemini_model", default="gemini-2.0-flash-exp", type = str, help="model of gemini") 
    parser.add_argument("--gemini_key", required=True, type = str, help="api key of gemini")
    parser.add_argument("--gemini_proxy", required=True, type = str, help="api key of gemini") 
    # Claude
    parser.add_argument('--claude_model', default="claude-3-7-sonnet-20250219", type = str, help="model of claude")  
    parser.add_argument('--claude_key', required=True, type = str, help="api key of claude")  
    parser.add_argument('--claude_proxy', required=True, type = str, help="api key of claude") 
    # deepseek
    parser.add_argument("--deepseek_model", type=str, default="deepseek-chat", help="deepseek model name")
    parser.add_argument("--deepseek_key", type=str, required=True, help="api key of deepseek")
    parser.add_argument("--deepseek_proxy", type=str, required=True, help="proxy of deepseek")
    # qwen
    parser.add_argument("--qwen_model", type=str, default="qwen-turbo-latest", help="qwen model name")
    parser.add_argument("--qwen_key", type=str, required=True, help="api key of qwen")
    parser.add_argument("--qwen_proxy", type=str, required=True, help="proxy of qwen")
    parser.add_argument("--vid_dir", type=str, default="Friends", help="Friends/BigBang/GOT/Movie")
    parser.add_argument("--start", type=int, default=0, help="start")
    parser.add_argument("--end", type=int, default=999, help="end")
    return parser.parse_args()


def getllm(model, args, iters=None):
    if model == 'claude':
        llm = ChatOpenAI(model_name=args.claude_model,                      # "claude-3-5-sonnet-20240620",
                         openai_api_key=args.claude_key,                
                         base_url= args.claude_proxy,                       
                         streaming=True
                        )
        print("llm:",args.claude_model)
    elif model == 'gemini':
        llm = ChatOpenAI(model_name=args.gemini_model,                      # "claude-3-5-sonnet-20240620",
                         openai_api_key=args.gemini_key,                
                         base_url= args.gemini_proxy,                   
                         streaming=True
                        )
        print("llm:",args.gemini_model)
    elif model == "gpt":
        llm = ChatOpenAI(model_name=args.openai_model,                      
                             openai_api_key=args.openai_key,                
                             base_url= args.openai_proxy,
                             streaming=True                     
                            )
        print(llm, args.openai_model)
    elif model == "deepseek":
        llm = ChatOpenAI(model_name=args.deepseek_model,
                         openai_api_key = args.deepseek_key,
                        base_url = args.deepseek_proxy,
                        streaming=True)
        print(llm, args.deepseek_model)
    elif model == "qwen":
        llm = ChatOpenAI(model_name=args.qwen_model,
                         openai_api_key = args.qwen_key,
                        base_url = args.qwen_proxy,
                        streaming=True)
        print(llm, args.qwen_model)
    return llm

# prompt模版
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

prompt_template_new = ChatPromptTemplate.from_template("{messages}")

# llm问答函数
def llm_qa(llm, question, chat_history):
    chat_history.add_user_message(question)
    chain = prompt_template  | llm | StrOutputParser()
    answer = chain.invoke(
        {
            "messages": chat_history.messages
        }
    )
    chat_history.add_ai_message(answer)
    return answer, chat_history

def llm_qa_streaming(llm, question, chat_history):
    chat_history.add_user_message(question)
    chain = prompt_template_new | llm | StrOutputParser()
    result = ""
    for partial_answer in chain.stream({"messages": chat_history.messages}):
        result += partial_answer
        print(partial_answer, end='', flush=True)
    chat_history.add_ai_message(result)
    return result, chat_history


def split_list(input_list, n):
    output_list = []
    for i in range(0, len(input_list), n):
        output_list.append(input_list[i:i+n])
    return output_list


def reorganize_lines(text):
    lines = text.strip().split('\n')
    line_dict = {}
    minnumber = 0
    maxnumber = 0
    for line in lines:
        matches = re.findall(r'(\d+)\s+(True|False)(.*?)((?=\d+\s+)|$)', line)
        for match in matches:
            number = int(match[0])  # 提取数字
            minnumber = min(minnumber, number)
            maxnumber = max(maxnumber, number)
            content = f"{match[1]} {match[2].strip()}"  # 提取内容并去掉前后空格
            line_dict[number] = content
    sorted_lines = []
    for i in range(minnumber, maxnumber + 1):
        if i in line_dict:
            sorted_lines.append(f"{i} {line_dict[i]}")
    
    return '\n'.join(sorted_lines)


def parse_output_for_all_incorrect(answers, iters):
    lines = answers.strip().split("\n")
    incorrect_num = 0
    for line in lines:
        if len(line.split())>2:
            correctness, select_answer = line.split(" ")[1], line.split(" ")[-1]
            if select_answer == "XXX":
                incorrect_num+=1
    # Avoid Too much abnormal
    if incorrect_num > (0.41+(iters%8*0.1))* len(lines):
        print("\nAbnormal Rate: {0:.2f}%, Threshod: {1:.2f}%".format(incorrect_num/len(lines)*100,100*(0.41+(iters%8*0.1))))
        return True
    else:
        print("\nAbnormal Rate: {0:.2f}%, Threshod: {1:.2f}%".format(incorrect_num/len(lines)*100,100*(0.41+(iters%8*0.1))))
        return False

if __name__ == "__main__":
    args = parser()
    # 挑选的电影的csv文件名

    video_type = args.vid_dir
    subtitle_script_align_root = "aligned_script"
    file_list = []
    for name in sorted(os.listdir(f'csv/{video_type}')):
        if ".csv" in name:
            file_list.append(name)

    offset_dict = {'gemini': 300, 'claude':300, 'gpt': 800}
    minend = min(args.end, len(file_list))
    for i, name in enumerate(file_list[args.start: minend]):
        if not '.csv' in name:
            continue
        print(i,name)
        vid = "result_"+name.split(".")[0]
        for model in ['gpt', 'claude', 'gemini']:
            output_path = f'check/{model}/{vid}.txt'
            if not os.path.exists(f'check/{model}'):
                os.makedirs(f'check/{model}')
            if os.path.exists(os.path.join(subtitle_script_align_root, video_type, name.split(".")[0] + ".xlsx")):
                script_path = os.path.join(subtitle_script_align_root, video_type, name.split(".")[0] +".xlsx")
            else:
                script_path = os.path.join(subtitle_script_align_root, video_type, name.split(".")[0] +".xltx")

            questions_path = os.path.join('csv', video_type, name)
            last_id, questions, Questions_list = getQuestion(questions_path)
            video_info = script_to_str(script_path)
            if os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        reviewed_questions = content.split("\n")
                        start_id = int(reviewed_questions[-1].split(" ")[0][:])
                    else:
                        start_id = 0
            else:
                start_id = 0
            iters = 0 
            full_incorrect_iters = 0
            offset = offset_dict[model]
            while start_id+1 <= last_id:
                llm = getllm(model, args, iters)
                iters = iters+1
                t_start, t_end = start_id, min(start_id+offset, last_id)

                print(f"{iters+1} iters, {model} processing: ", start_id, "->", t_end)
                t_questions = "id; Question; Choices\n" + "\n".join(Questions_list[t_start: t_end])
                prompt = getPrompt(t_start+1, t_end, video_info, t_questions)
                chat_history = ChatMessageHistory()
                try:
                    answer, chat_history = llm_qa_streaming(llm, prompt, chat_history)
                    last_line = answer.split("\n")[-1]
                    if parse_output_for_all_incorrect(answer, full_incorrect_iters):
                        full_incorrect_iters += 1
                        offset = t_end - t_start+1
                        offset = offset//2
                        print("Too much abnormal")
                        if offset>10:
                            continue
                    else:
                        offset = offset_dict[model]
                        full_incorrect_iters = 0
                except Exception as e:
                    error_message = traceback.format_exc()
                    print("llm error", error_message)
                    continue
                if len(last_line.split(" ")) <= 5:
                    answer = "\n".join(answer.split("\n")[:])
                    
                with open(output_path, 'a') as f:
                    answer = reorganize_lines(answer)
                    if answer.strip():
                        if start_id == 0:
                            f.write(answer.strip())
                        else:
                            f.write("\n"+answer.strip())
                    # time.sleep(10)
                with open(output_path, 'r') as f:
                    reviewed_questions = f.read().split("\n")
                    start_id = int(reviewed_questions[-1].split(" ")[0][:])
            reorganize_str = ""
            with open(output_path, 'r') as f:
                reorganize_str = reorganize_lines(f.read())
            with open(output_path, "w") as f:  
                f.write(reorganize_str)
        # time.sleep(5)
