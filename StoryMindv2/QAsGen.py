import os
import json
from pathlib import Path
from pprint import pprint
import argparse
import tiktoken
import openai
import time
import pandas as pd
import json

#导入langchain的stool
from langchain.agents import tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
#from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain import hub

from utils.tools import saveQuestion, save_restricted_Question, deleteQuestion, get_temporary_question
from utils.engine import script_to_str, prompt_engine
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent, create_structured_chat_agent
import traceback

from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# index 部分导入没有变化
from langchain.indexes import SQLRecordManager, index
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import FAISS 
from langchain.tools.retriever import create_retriever_tool
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any, Dict, List, Optional, Union,Sequence
from uuid import UUID
import traceback
from langchain_core.documents import Document
os.environ["TOKENIZERS_PARALLELISM"] = "false"

encoding = tiktoken.get_encoding("o200k_base")



# --- 重新定义一个专注调试的回调处理器 ---
class DebugCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.llm_call_count = 0
        self.llm_outputs = [] # 存储所有 llm_output
   
    def on_chain_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        self.total_prompt_tokens += len(encoding.encode(str(prompts)))

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        traceback.print_exc() # 打印详细错误
    # 可以添加 on_agent_action, on_tool_end 等来观察 Agent 流程
    def on_agent_action(self, action, **kwargs: Any) -> Any:
        action_token =  len(encoding.encode(str(action)))
        self.total_completion_tokens += action_token
        self.total_prompt_tokens += (self.total_prompt_tokens + action_token)
        # print(f"\n>>> DEBUG: Agent Action: Tool={action.tool}, Input={action.tool_input}")
    def on_tool_end(self, output: str, **kwargs: Any
    )-> None:
        action_token =  len(encoding.encode(str(output)))
        # print("检索返回token", action_token)
        self.total_prompt_tokens += (self.total_prompt_tokens + action_token)
    def get_total_tokens(self) -> int:
        return self.total_prompt_tokens +  self.total_completion_tokens
    def get_prompt_tokens(self) -> int:
        return self.total_prompt_tokens
    def get_completion_tokens(self) -> int:
        return self.total_completion_tokens

# intfloat/multilingual-e5-large-instruct
# Alibaba-NLP/gte-Qwen2-7B-instruct
# Alibaba-NLP/gte-Qwen2-1.5B-instruct
model_name = "intfloat/multilingual-e5-large-instruct"
model_kwargs = {'device': 'cuda:0', 'trust_remote_code':True}
encode_kwargs = {'normalize_embeddings': False}
emb = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    cache_folder = "ckpt",
    show_progress = True
)
db = None

def parser():
    parser = argparse.ArgumentParser("StoryMind's generator", add_help=True)
    parser.add_argument("--gemini_model", default="gemini-2.0-flash", type = str, help="model of gemini") 
    parser.add_argument("--gemini_key", required=True, type = str, help="api key of gemini")
    parser.add_argument("--gemini_proxy", required=True, type = str, help="api key of gemini") 
    parser.add_argument("--temperature", default=0.8, type=float, help="number of workers")
    parser.add_argument("--each_type_num", default=120, type=int, help="'Least questions num of each type." )
    parser.add_argument('--vid_dir', type=str, default="Friends")
    return parser.parse_args()

def extend_history(chat_history, prompt, answer):
    chat_history.extend(
        [
            HumanMessage(content=prompt),
            AIMessage(content=answer),
        ]
    )
    return chat_history


def get_restricted_agent_with_tools(vid, csv_path, llm, qtype, element):
    question_save_tool = saveQuestion(csv_path=csv_path, video_id=vid)
    tools = [question_save_tool, tool_dict[f"{qtype}-{element}"]]
    prompt_react = hub.pull("hwchase17/structured-chat-agent")
    agent = create_structured_chat_agent(llm, tools, prompt_react)
    agent_executor= AgentExecutor(agent=agent,tools=tools,verbose=True,max_iteractions = 2,handle_parsing_errors=True, early_stopping_method='force')
    return agent_executor

# 修改 get_agent_with_tools 以使用并返回 DebugCallbackHandler
def get_agent_with_tools(vid, llm, needed_tool_name, tool_dict):
    tools=[]
    for name in needed_tool_name:
        tools.append(tool_dict[name])
    prompt_react = hub.pull("hwchase17/structured-chat-agent")
    agent = create_structured_chat_agent(llm, tools, prompt_react)
    debug_callback = DebugCallbackHandler()

    agent_executor= AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True, 
        max_iteractions=2,
        max_execution_time=30,
        handle_parsing_errors=True,
        early_stopping_method='force',
        callbacks=[debug_callback] 
    )
    return agent_executor, debug_callback
    
def get_restricted_agent_with_delete_tools(vid, csv_path, trash_path,llm, last_iter_index):
    global db
    question_delete_tool = deleteQuestion(csv_path=csv_path, trash_path = trash_path, video_id=vid)
    tools = [question_delete_tool]
    debug_callback = DebugCallbackHandler()

    if not os.path.exists(trash_path):
        retriever = None
        data = []
    else:
        loader = CSVLoader(file_path=trash_path, content_columns=['question','choices_list','gt','delete_reason'], source_column='vid')
        data = loader.load()[last_iter_index:]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=20)
        text_chunks = text_splitter.split_documents(data)
        if not db:
            db = FAISS.from_documents(text_chunks, embedding=emb)
        else:
            db.add_documents(text_chunks, embedding = emb)
        retriever = db.as_retriever(search_type="mmr",search_kwargs={'k':10, 'lambda_mult':0.25})
        retriever_tool = create_retriever_tool(
            retriever,
            "similar_deleted_questions_search", "Search for deleted questions with similar reasons from deleted questions database . After deleting questions, you must use this tool to retrieve similar wrong questions and the reason why they are wrong before you summarize and feedback to generator!"
        )

        tools = [question_delete_tool, retriever_tool]


    prompt_react = hub.pull("hwchase17/structured-chat-agent")
    agent = create_structured_chat_agent(llm, tools, prompt_react)
    agent_executor= AgentExecutor(agent=agent,
                                  tools=tools,
                                  verbose=True,
                                  max_iteractions=10, 
                                  handle_parsing_errors=True, 
                                  retriever = retriever,
                                  early_stopping_method='force',
                                  callbacks=[debug_callback] )
    
    return agent_executor, len(data), debug_callback

 


    return shots_text



def split_list(input_list, n):
    output_list = []
    for i in range(0, len(input_list), n):
        output_list.append(input_list[i:i+n])
    return output_list

def format_time(seconds):
    # 将传入的浮点数转换为整数
    total_seconds = int(seconds)
    days = total_seconds // (24 * 3600)
    total_seconds %= (24 * 360)
    hours = total_seconds // 3600
    total_seconds %= 3600
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{days:02d} day-{hours:02d} hour-{minutes:02d} min-{seconds:02d} s"


def json_load(json_path):
    # 从JSON文件加载数据
    with open(json_path, 'r') as json_file:
        loaded_data = json.load(json_file)
    return loaded_data

def json_save(data, json_path):
    # 将字典保存为JSON文件
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)  # indent=4用于格式化输出



if __name__=="__main__":
    args = parser()
    max_iters = 200
    subtitle_script_align_root = f"aligned_script/{args.vid_dir}"
    if not os.path.exists(f"csv/{args.vid_dir}"):
        os.mkdir(f"csv/{args.vid_dir}")

    for idx, name in enumerate(sorted(os.listdir(subtitle_script_align_root))[:]):
        file_path = os.path.join(subtitle_script_align_root, name)
        chat_history = []
        video_info = script_to_str(file_path)
        iters = 0
        feedback = None
        tool_dict = dict() 
        vid = name.split(".")[0]
        print("==================== Processing", vid, "====================")
        csv_path = f"csv/{args.vid_dir}/{vid}.csv"
        trash_path = f"delete/{vid}.csv"
        generated_questions, number_enough, needed_tool_name = get_temporary_question(csv_path, least_num=args.each_type_num)
        last_iter_id = 0

        for qtype in ['P', 'I']:
            for element in ['C','A','L','CA','CL','AL','CAL']:
                tool = save_restricted_Question(csv_path=csv_path, video_id=vid, qtype = qtype, element=element, least_num=args.each_type_num)
                tool_dict[f"{qtype}-{element}"] = tool

        
        while number_enough < 14:
            iters += 1
            print(f"#################################{iters}#################################################")
            print(idx, name, number_enough)
            callback = UsageMetadataCallbackHandler()
            llm = ChatOpenAI(model_name=args.gemini_model,                     
                         openai_api_key=args.gemini_key,               
                         base_url= args.gemini_proxy,   
                         streaming=True
                        )
            agent_executor, debug_callback_instance = get_agent_with_tools(vid, llm, needed_tool_name, tool_dict)
            prompt = prompt_engine(generated_questions, video_info, prompt_type="qg", feedback=feedback)
            try:
                answer = agent_executor.invoke({"input": prompt})
            except Exception as e:
                error_message = traceback.format_exc()
                print(error_message)
        
            generated_questions, number_enough, needed_tool_name = get_temporary_question(csv_path, least_num=args.each_type_num)
            # time.sleep(60)
            # Supervisor
            try:
                start_time = time.time()
                supevisor,last_iter_id, debug_callback_instance = get_restricted_agent_with_delete_tools(vid, csv_path, trash_path, llm, last_iter_index=last_iter_id)
                prompt = prompt_engine(generated_questions, video_info, prompt_type="qj")
                if last_iter_id<=2000:
                    feedback = supevisor.invoke({"input": prompt})
                    print("Supervisor:",feedback['output'])
                    generated_questions, number_enough, needed_tool_name = get_temporary_question(csv_path, least_num=args.each_type_num)
            except Exception as e:
                error_message = traceback.format_exc()
            # time.sleep(60)
            if iters >= max_iters:
                break

    
    