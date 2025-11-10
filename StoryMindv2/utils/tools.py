import os
import pandas as pd
from pydantic import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_community.utilities import SQLDatabase
from typing import Optional, Type
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import TextLoader
from typing_extensions import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime
import tiktoken
import json


encoding = tiktoken.get_encoding("o200k_base")

#from utils.db import append_csv_memory, csv_to_database

def json_load(json_path):
    # 从JSON文件加载数据
    with open(json_path, 'r') as json_file:
        loaded_data = json.load(json_file)
    return loaded_data

def json_save(data, json_path):
    # 将字典保存为JSON文件
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)  # indent=4用于格式化输出
        
def add_to_csv(data, csv_path):
    data['index'] = range(len(data))
    if not os.path.exists(csv_path):
        data.to_csv(csv_path, header=True, index=False)
    else:
        data.to_csv(csv_path, mode='a', header=False, index=False)
map_element = {'C':0,  'A': 1 , 'L':2 , 'CA':3, 'CL':4, 'AL':5, 'CAL':6,
              0:'C',  1:'A' , 2:'L' , 3:'CA',  4:'CL', 5:'AL', 6:'CAL'}

def append_to_csv(csv_path, data, save_type):
    if save_type == "question":
        columns = ['vid', 'question','gt','choices_list', 'question_type', 'element',  'basis', 'related_times', 'related_person', 'related_location', 'time']  
    elif save_type == "del_question":
        columns = ['vid', 'question','gt','choices_list', 'question_type', 'element',  'basis', 'related_times', 'related_person', 'related_location','delete_reason','time']
    else:
        columns = ['vid', 'question', 'gt','choices_list', 'question_type', 'element', 'basis', 'related_times', 'related_person', 'related_location','time'] ## to do to revise

    df = pd.DataFrame(data, columns=columns)
    add_to_csv(df, csv_path)

def get_temporary_question(csv_path, least_num=120):
    if type(csv_path) == str:
        if not os.path.exists(csv_path):
            return "You can use given tools to save questions !!", 0, ['P-C','P-A','P-L','P-CA','P-CL','P-CAL','I-C','I-A','I-L','I-CA','I-CL','I-CAL']
        df = pd.read_csv(csv_path)  
        df = df.drop_duplicates('question').reset_index(drop=True)
        df['index'] = range(len(df))
        df.loc[df['question_type']=='perception','question_type'] = 'P'
        df.loc[df['question_type']=='inference','question_type'] = 'I'
        df.to_csv(csv_path, header=True, index=False)
    else:
        df = csv_path
     
    questions_str = "\nUp to Now, questions of each time are as follow:\n"
    type_dict = {'P': [0,0,0,0,0,0,0], 'I': [0,0,0,0,0,0,0]}
    for index, row in df.iterrows():
        type_dict[row['question_type']][map_element[row['element']]] += 1
        questions_str += f"Index: {row['index']} Question_type: {row['question_type']}, Element: {row['element']}, Question: {row['question']}\n"
    questions_str += "\n"

    perception_type = ""
    inference_type = ""
    number_enough = 0 
    each_type_num = least_num
    print("每类需要", each_type_num)
    needed_tool_name = []
    for i in range(7):
        if type_dict['P'][i] >= each_type_num:
            number_enough += 1
        if type_dict['I'][i] >= each_type_num:
            number_enough += 1
        if type_dict['P'][i] < each_type_num:
            perception_type  += "Question of question type: P, story elements: " + map_element[i] + f", it's {type_dict['P'][i]} questions only. use 'Tools_of_saving_quetsion_with_P_{map_element[i]}' to generate more.\n"
            needed_tool_name.append(f"P-{map_element[i]}")
        if type_dict['I'][i] < each_type_num:
            inference_type  +="Question of question_type: I, story elements: " + map_element[i] + f", it's {type_dict['I'][i]} questions only, use 'Tools_of_saving_quetsion_with_I_{map_element[i]}' to generate more.\n"
            needed_tool_name.append(f"I-{map_element[i]}")
#    print("For perception:\n" + perception_type + "For inference:\n" + inference_type )
    questions_str += "For P:\n" + perception_type + "For I:\n" + inference_type + "\n### Don't generate same or similar question, think different please !!!" 
    return questions_str, number_enough, needed_tool_name

# define tool's input: question 
class questionInput(BaseModel):
    question: List[str] = Field(description="questions list you want to save.") 
    # to do
    choices: List[List[str]] = Field(description="choices list (4 choices) of each question. For each question, it should be [choice1, choice2, choice3, choice4].")
    gt: List[str] = Field(description="GroundTruth answer (actual answer, not 'A','B','C','D') for each question.") 
    question_type: List[str] = Field(description="question_types list of each question, P for perception, I for inference. Choose from 'P' and 'I'.")
    element: List[str] = Field(description="story elements list related to each problem, C for Character, A for Action, L for Location. Choose from 'C', 'A', 'L', 'CA', 'CL', 'AL', 'CAL'.")
#    format: List[str] = Field(description="formats list of each question, 'matching' or 'multiple-choice")
    basis: List[str] = Field(description="inference basis list of each question, inference basis is the reason why we can get such a question's answer from video, it's a short description.")
    related_times: List[List[str]] = Field(description="time span list of video clips related to each question. For each question, it should be ['start1-end1', 'start2-end2',...]")
    related_person: List[List[str]] = Field(description="characters list related to each question. For each question, it should be [XXX, XXX,...], XXX stands for character name from script.")
    related_location: List[List[str]] = Field(description="locations list related to each question. For each question, it should be [XXX, XXX,...], XXX stands for location name from script.")

# define tool's input: question 
# Please provide a sufficiently wide time range to ensure the relevant information/event is comfortably included and to minimize timing errors. 
class restrict_questionInput(BaseModel):
    question: List[str] = Field(description="questions list you want to save.") 
    choices: List[List[str]] = Field(description="choices list (5 choices) of each question. For each question, it should be [choice1, choice2, choice3, choice4, choice5].")
    gt: List[str] = Field(description="GroundTruth answer (actual answer, not 'A','B','C','D','E') for each question.") 
    basis: List[str] = Field(description="inference basis list of each question, inference basis is the reason why we can get such a question's answer from video, it's a short description.")
    related_times: List[List[str]] = Field(description="time span list of video clips related to each question. For each question, it should be ['start1-end1', 'start2-end2',...]")
    related_person: List[List[str]] = Field(description="characters list related to each question. For each question, it should be [XXX, XXX,...], XXX stands for character name from script.")
    related_location: List[List[str]] = Field(description="locations list related to each question. For each question, it should be [XXX, XXX,...], XXX stands for location name from script.")

class questiondeleteInput(BaseModel):
    indexs: List[int] = Field(description="the indexs of wrong/meaningless/question_type wrong/element wrong questions that you want to delete.")
    reasons: List[str] = Field(description="for each question you want to delete, give your reason.")

class save_restricted_Question(BaseTool):
    name: str = "Tools_of_saving_restricted_quetsion"
    description: str = "Follow the requirements of this tool to save questions with restricted type !"
    args_schema: Type[BaseModel] = restrict_questionInput
    # private config
    csv_path: str = None                             # default csv
    video_id: str = None
    qtype: str =  None
    element: str = None
    least_num: int = 15
    def __init__(self, csv_path: str, video_id: str, qtype: str, element: str, least_num: int):
        super().__init__() 
        self.csv_path = csv_path
        self.video_id = video_id
        self.qtype = qtype
        self.element = element
        self.least_num = least_num
        self.name = f"Tools_of_saving_quetsion_with_{qtype}_{element}"
        self.description = f"Follow the requirements of this tool and defined input format to save questions whose question_type is {qtype} and story elements combination is {element} !"
    def _run(
        self, question: List[str], 
              choices: List[List[str]],
              gt: List[str],
              basis: List[str], 
              related_times: List[List[str]],
              related_person: List[List[str]],
              related_location: List[List[str]],
              run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        now = datetime.now()
        # 将当前时间格式化为字符串
        time_string = now.strftime("%Y-%m-%d %H:%M:%S")
        if len(related_location) == 0 :
            related_location = [[] for i in range(len(question))]
        if len(related_person) == 0 :
            related_person = [[] for i in range(len(question))]
        if len(set([len(question), len(gt), len(choices),len(related_times), len(basis), len(related_person),len(related_location)]))!=1:
            return "The generated lists 'question', 'choices', 'gt', 'basis', 'related_times', 'related_person' and 'related_location' need to be of the same size and correspond one-to-one."
        data = [[self.video_id, 
                 question[i],
                 gt[i],
                 choices[i],
                 self.qtype,
                 self.element,
                 basis[i], 
                 related_times[i],
                 related_person[i],
                 related_location[i],
                 time_string]
                 for i in range(len(question)) if gt[i] in choices[i]
               ]
        # append to csv
        append_to_csv(self.csv_path, data, 'question')
        temporary_question_str, number_enough, needed_tool_name = get_temporary_question(self.csv_path, least_num=self.least_num)
        if not f"{self.qtype}-{self.element}" in needed_tool_name:
            return_str = f"{temporary_question_str}. Please Stop using this tool: {self.name}. Questions of this time have been enough!"
            
        if number_enough ==14:
            return_str = "questions have been enough. Stop generation!"
        else:
            return_str = f"{temporary_question_str}."

        # json_path="temp.json"
        # if os.path.exists(json_path):
        #     temp_json = json_load(json_path)
        # else:
        #     temp_json = {"save_temp": 0, "save_return": 0, "delete_temp":0, "delete_return": 0}
            
        # token_num = len(encoding.encode(str(return_str)))
        # temp_json["save_temp"] += temp_json["save_temp"]+ token_num
        # temp_json['save_return']+= temp_json["save_temp"]
        # json_save(temp_json, json_path)
        return return_str

    async def _arun(
        self, question: List[str], 
              choices: List[List[str]],
              gt: List[str],
              basis: List[str], 
              related_times: List[List[str]],
              related_person: List[List[str]],
              related_location: List[List[str]],
              run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError(f"{self.name} does not support async")

class saveQuestion(BaseTool):
    name: str = "Tools_of_saving_quetsion"
    description: str = "Follow the requirements of this tool to save questions !"
    args_schema: Type[BaseModel] = questionInput
    # private config
    csv_path: str = "csv/default.csv"                             # default csv
    video_id: str = "001.mp4"
    def __init__(self, csv_path: str, video_id: str):
        super().__init__() 
        self.csv_path = csv_path
        self.video_id = video_id
        
    def _run(
        self, question: List[str], 
              choices: List[List[str]],
              gt: List[str],
              question_type: List[str], 
              element: List[str], 
              basis: List[str], 
              related_times: List[List[str]],
              related_person: List[List[str]],
              related_location: List[List[str]],
              run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        if len(related_location) == 0 :
            related_location = [[] for i in range(len(question))]
        if len(related_person) == 0 :
            related_person = [[] for i in range(len(question))]
        data = [[self.video_id, 
                 question[i],
                 gt[i],
                 choices[i],
                 question_type[i],
                 element[i],
                 basis[i], 
                 related_times[i],
                 related_person[i],
                 related_location[i]]
                 for i in range(len(question))
               ]
        # append to csv
        append_to_csv(self.csv_path, data, 'question')
        temporary_question_str, number_enough, _  = get_temporary_question(self.csv_path)
        if number_enough ==14:
            return "questions have been enough. Stop generation!"
        return f"{temporary_question_str}"

    async def _arun(
        self, question: List[str], 
              choices: List[List[str]],
              gt: List[str],
              question_type: List[str], 
              element: List[str], 
              basis: List[str], 
              related_times: List[List[str]],
              related_person: List[List[str]],
              related_location: List[List[str]],
              run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError(f"{self.name} does not support async")

class saveCorssQuestion(BaseTool):
    name: str  = "Tools_of_saving_cross_quetsion"
    description: str  = "useful when you generate cross episodes questions and need to save them"
    args_schema: Type[BaseModel] = questionInput
    # private config
    csv_path: str = "csv/default.csv"                             # default csv
    video_id: str = "001.mp4"
    def __init__(self, csv_path: str, video_id: List[str]):
        super().__init__() 
        self.csv_path = csv_path
        self.video_id = video_id
        
    def _run(
        self, question: List[str], 
              choices: List[List[str]],
              gt: List[str],
              question_type: List[str], 
              element: List[str], 
              basis: List[str], 
              related_times: List[List[str]],
              related_person: List[List[str]],
              related_location: List[List[str]],
              run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        if len(related_location) == 0 :
            related_location = [[] for i in range(len(question))]
        if len(related_person) == 0 :
            related_person = [[] for i in range(len(question))]
        data = [[self.video_id, 
                 question[i],
                 gt[i],
                 choices[i],
                 question_type[i],
                 element[i],
                 basis[i], 
                 related_times[i],
                 related_person[i],
                 related_location[i]]
                 for i in range(len(question))
               ]
        # append to csv
        append_to_csv(self.csv_path, data, 'question')
        temporary_question_str, number_enough, _ = get_temporary_question(self.csv_path)
        if number_enough ==14:
            return "questions have been enough. Stop generation!"
        return f"{temporary_question_str}"

    async def _arun(
        self, question: List[str], 
              choices: List[List[str]],
              gt: List[str],
              question_type: List[str], 
              element: List[str], 
              basis: List[str], 
              related_times: List[List[str]],
              related_person: List[List[str]],
              related_location: List[List[str]],
              run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError(f"{self.name} does not support async")

class deleteQuestion(BaseTool):
    name: str  = "Tools_of_delete_invalid_quetsion"
    description: str  = "It's useful when you want to delete wrong/meaningless/question_type wrong/element wrong questions."
    args_schema: Type[BaseModel] = questiondeleteInput
    # private config
    csv_path: str = "csv/default.csv"                             # default csv
    trash_path: str = "delete/default.csv"
    video_id: str = "001.mp4"
    def __init__(self, csv_path: str, trash_path: str, video_id: str):
        super().__init__() 
        self.csv_path = csv_path
        self.video_id = video_id
        self.trash_path = trash_path
        
    def _run(
        self, indexs: List[int], reasons: List[str], run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        # Load the existing questions from the CSV file

        now = datetime.now()
        # 将当前时间格式化为字符串
        time_string = now.strftime("%Y-%m-%d %H:%M:%S")
        try:
            df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            return "There're no questions can be deleted."
        if len(indexs)>10:
            return f"For conservative, you can't delete {len(indexs)} questions (It's too radical, you can only delete less than 10 questions and break)."
        # Check if the indices are valid
        if any(index < 0 or index >= len(df) for index in indexs):
            return "One or more indices are out of range."
        deleted_questions_data = []
        rows_to_delete = df[df['index'].isin(indexs)]
        for index, reason in zip(indexs, reasons):
            if index in rows_to_delete['index'].values:
                deleted_row = rows_to_delete.loc[(rows_to_delete['index'] == index)].iloc[[0]]
                deleted_questions_data.append([
                    self.video_id,
                    deleted_row['question'].values.item(),
                    deleted_row['gt'].values.item(),
                    deleted_row['choices_list'].values.item(),
                    deleted_row['question_type'].values.item(),
                    deleted_row['element'].values.item(),
                    deleted_row['basis'].values.item(),
                    deleted_row['related_times'].values.item(),
                    deleted_row['related_person'].values.item(),
                    deleted_row['related_location'].values.item(),
                    reason,
                    time_string
                ])
        if deleted_questions_data:
            # Use the provided append_to_csv function to save to the trash file
            append_to_csv(self.trash_path, deleted_questions_data, 'del_question')

        if rows_to_delete.empty:
            return "No matching rows to delete.", None
        df = df.drop(rows_to_delete.index)
        # Reset the DataFrame's *row* index (important!)
        df.reset_index(drop=True, inplace=True)
        # Renumber the 'index' *column* sequentially
        df['index'] = range(len(df))  # This is what you specifically asked for

        return_str = f"Deleted questions at indices: {indexs}."
        # token_num = len(encoding.encode(str(return_str)))
        # json_path="temp.json"
        # if os.path.exists(json_path):
            # temp_json = json_load(json_path)
        # else:
            # temp_json = {"save_temp": 0, "save_return": 0, "delete_temp":0, "delete_return": 0}
        # temp_json["delete_temp"] += temp_json["delete_temp"]+ token_num
        # temp_json['delete_return']+= temp_json["delete_temp"]
        # json_save(temp_json, json_path)
        return return_str
  

    async def _arun(
        self, indexs: List[int], reasons: List[str], run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError(f"{self.name} does not support async")





#
#vid_name = "test"
#question_save_tool = saveCorssQuestion(csv_path=f"../csv_cross/{vid_name}.csv", video_id=['S01E01','S01E02','S01E03'])
#print(question_save_tool._run(['question'], [['a','b','c','D']],['a'],['perception'], ['A'], ['f'],[['00:11-01:22', '01:22-03:11']], [['character1', 'character2']], [['location1', 'location2']]))
#vid = "test"
#question_save_tool = saveQuestion(csv_path=f"csv/{vid}.csv", video_id=vid)
#print(question_save_tool._run('question', 'a', 't', 'f', 'b',[['00:01', '00:08'], ['00:11', '01:22']], [['character1', 'character2'], ['location1', 'location2']]))