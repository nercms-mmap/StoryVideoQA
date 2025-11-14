# pip install accelerate
import os
import json
from transformers import AutoProcessor
from PIL import Image
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import re
import json
import torch
import pandas as pd
from tqdm import tqdm
from collections import Counter
from typing import Any, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
import re
import sys
import time

def find_first_letter_with_parenthesis(s):
    # 正则表达式模式：以左括号开头，后面跟着一个字母
    pattern = r'\(([A-Za-z])'
    
    # 使用 re.search 查找第一个匹配项
    match = re.search(pattern, s)
    
    # 如果找到匹配项，返回对应的字母
    if match:
        return match.group(1)  # 返回匹配的字母
    else:
        return None  # 如果没有找到匹配项，则返回 None


def find_choices_in_string(target_string, choices_list):
    # 创建正则表达式模式
    # 对 choices_list 进行排序，确保较长的模式优先匹配
    choices_list_sorted = sorted(choices_list, key=len, reverse=True)
    pattern = '|'.join(re.escape(choice) for choice in choices_list_sorted)
    matches = re.findall(pattern, target_string)
    return matches

def load_json(path):
    with open(path, 'r') as f:
        # 使用json.load方法加载数据
        data = json.load(f)
    return data

def append_to_csv(csv_path, data, columns):
    df = pd.DataFrame(data, columns=columns)
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, header=True, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)


def choices_to_str(choice_list):
    choices = ""
    for i, choice in enumerate(choice_list):
        choices += "(" + chr(65+i)+")" + f" {choice}\n"    # 65表示字符'A'
    return choices



def get_llm(llm_name, args):
    if 'GPT' in llm_name or 'Gemini' in llm_name :
        llm = ChatOpenAI(model_name=args.openai_model,                      
                             openai_api_key=args.openai_key,                
                             base_url= args.openai_proxy,
                             streaming=True                     
                            )
    else:
        llm = None
        print("Fatal error: Not supported LLM")
    return llm

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system","You are a helpful assistant.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
context = """XXXXXX"""


class QwenChatbot:
    def __init__(self, model_name="Qwen/Qwen3-30B-A3B", cache_dir = "/mnt/disk6new/wzq/LLM/ckpt"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = cache_dir, device_map='auto')
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = cache_dir, device_map='auto')
        self.history = []

    def generate_response(self, user_input, enable_thinking = False):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking  # True is the default value for enable_thinking.
        )
        
        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response

def read_json(json_path):
    data = None
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as file:
            json_str = file.read()
            # 使用json.loads()方法解析JSON字符串
            data = json.loads(json_str)
    return data

def save_json(json_path, data):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)


def choices_str(choice_list):
    choices = ""
    for i, choice in enumerate(choice_list):
        choices += "(" + chr(65+i)+")" + f" {choice}\n"    # 65表示字符'A'
    return choices


def gemma_chat(processor, model, question):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{question}"}  # {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded


def chat(llm, question, max_retries=3):
    chat_history = ChatMessageHistory()
    chat_history.add_user_message(question)
    chain = prompt_template | llm | StrOutputParser()
    # answer = chain.invoke({
    #     "messages": chat_history.messages
    # })

    for attempt in range(max_retries):
        try:
            answer = chain.invoke({
                "messages": chat_history.messages
            })
            return answer  # 如果成功，返回答案
        except Exception as e:  # 捕获所有异常
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(100)
            print(f"begin new attemp")
            if attempt == max_retries - 1:
                print("Exceeded maximum retries. Exiting...")
                return "No answer"
                sys.exit(1)  # 超过最大重试次数，杀掉程序

    return "Model return nothing"





# if __name__ =="__main__":
    
#     model_id = "/home/lrz/project/LLM/weight/gemma-3-27b-it"
#     model = Gemma3ForConditionalGeneration.from_pretrained(
#         model_id, device_map="auto", cache_dir = "/home/lrz/project/LLM/weight"
#     ).eval()
#     processor = AutoProcessor.from_pretrained(model_id, cache_dir = "/home/lrz/project/LLM/weight")
#     ans = gemma_chat(processor, model, question="How are you! Do you know Harry Potter?")
#     print(ans)


def check_ans_GT(response_dict, q_dict):
    criteria = False
    # 答案顺序 Qwen3 LLaMA3.1 Qwen2.5-7b Qwen2.5-14b Qwen2-7B
    for key in response_dict.keys():
        ans = response_dict[key]
        if find_first_letter_with_parenthesis(ans)==q_dict['option']:
            criteria = True
        else:
            criteria = False
            break
    return criteria

if __name__ == "__main__":
    json_path = "json/StoryMindv1.5-guineapig-0514.json"
    guinea_pig_json_path = 'json/StoryMindv1.5-guineapig-0514.json'
    questions = read_json(json_path)

 
    # llm = get_llm("meta-llama/Meta-Llama-3.1-8B-Instruct")
    # chatbot = QwenChatbot(model_name="Qwen/Qwen3-30B-A3B", cache_dir = "/mnt/disk6new/wzq/LLM/ckpt")
    # First input (without /think or /no_think tags, thinking mode is enabled by default)
    # user_input_1 = "How many r's in strawberries?"
    # print(f"User: {user_input_1}")


    llm_name = "Qwen/Qwen2-7B"
    llm_id = llm_name.split("/")[1]
    llm = get_llm(llm_name)
    score = 0
    # for i in tqdm(range(len(questions))):
    for i in range(len(questions)):
        q_dict = questions[i]
        if 'guinea_pig' in q_dict.keys():
            if llm_id in q_dict['guinea_pig'].keys():
                if check_ans_GT(q_dict['guinea_pig'], q_dict):
                    score +=1
                continue
        print(score, i+1)    
    
        question = f"Assuming you haven't seen the video or any priority knowledge of this video, answer the question directly.\nQuestion: {q_dict['question']}\nPlease select best option from following choices:\n" + choices_str(q_dict['choices'])+" \nAnswer:("
        # # print(f"User: {question}")
        # response = chatbot.generate_response(question)
        # print(f"Bot: {response}")
        response = chat(llm, question=question)
        if not 'guinea_pig' in q_dict.keys():
            questions[i]['guinea_pig'] = {llm_id: response}
        else:
            questions[i]['guinea_pig'][llm_id] = response
        if check_ans_GT(q_dict['guinea_pig'], q_dict):
            score += 1
        print(score, i+1, response.strip(), q_dict['GT'])
        if i%20==0 or i == len(questions)-1:
            save_json(guinea_pig_json_path, questions)
    print(score, len(questions))
    # print("----------------------")

    # Second input with /no_think
    # user_input_2 = "Then, how many r's in blueberries? /no_think"
    # print(f"User: {user_input_2}")
    # response_2 = chatbot.generate_response(user_input_2)
    # print(f"Bot: {response_2}") 
    # print("----------------------")

    # # Third input with /think
    # user_input_3 = "Really? /think"
    # print(f"User: {user_input_3}")
    # response_3 = chatbot.generate_response(user_input_3)
    # print(f"Bot: {response_3}")


