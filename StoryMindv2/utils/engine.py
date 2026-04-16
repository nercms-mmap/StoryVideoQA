import os
import numpy as np
import pandas as pd

# subtitle_script_root = "/mnt/disk6new/wzq/experiment/AAAIext/HP/script_HP"
# subtitle_script_align_root = "/mnt/disk6new/wzq/experiment/AAAIext/HP/script_HP_align"
# subtitle_script_align_files = os.listdir(subtitle_script_align_root)

def convert_case(text):
    # Split into words
    words = text.split()
    if not words:
        return text       
    # Convert first word to title case, rest to lowercase
    result = [words[0].title()] + [w.lower() for w in words[1:]]
    # Join back together.
    return ' '.join(result)


def sheet_to_str(script_sheet, result_sheet,boundingbox = False):
#    sheet_str = "This is a script that follows the timeline.Each line we supply the character's name,location where the character locates and content including the character's dialogue and action.\n"
#    sheet_str += "characters - location - content\n"
    # dialog_shot, bbox = match_shot_subtitle(file_path)
    # print(dialog_shot,bbox)
    scene_descriptions = dict()
    temp_scene_id = 1
    for i in range(len(script_sheet)):
        if script_sheet.loc[i,'record_type'] == 'scene':
            scene_descriptions[temp_scene_id] = script_sheet.loc[i,'content']
            # print(temp_scene_id, scene_descriptions[temp_scene_id])
            temp_scene_id +=1

    sheet_str = ""
    scene_id = 0
    last_scene_id = 0
    temp_description = ""
    last_location = ""
    for i in range(len(result_sheet)):
        start = result_sheet.loc[i,'start_time'][:-4]
        end =  result_sheet.loc[i,'end_time'][:-4]
        scene_index = result_sheet.loc[i,'scene_index']
        dialog = result_sheet.loc[i,'dialog']
        character = convert_case(result_sheet.loc[i,'characters'])
        if not pd.isnull(result_sheet.loc[i,'location']):
            location = convert_case(result_sheet.loc[i,'location'])
        else:
            location = ""
        ################# 对于特殊的script，没有description ##############
        if 'description' in result_sheet.columns:
            description = str(result_sheet.loc[i,'description'])

            if pd.isnull(description):
                description = ""
            else:
                description = description.replace("[",'').replace("]",'')
        #################################################################

        if scene_index != scene_id and scene_id < scene_index:
            # Some scenes may be consistent
            while scene_id < scene_index-1:
                scene_id += 1
                if location:
                    sheet_str += f"Scene {scene_id}, Location: {location}\n"
                else:
                    sheet_str += f"Scene {scene_id}\n"
                scene_description = scene_descriptions[scene_id]
                if scene_description:
                    sheet_str += f"[{scene_description}]\n"


            scene_id = scene_index
            scene_description = scene_descriptions[scene_id]
            if pd.isnull(scene_description):
                scene_description = ""
            else:
                scene_description = scene_description.replace("[",'').replace("]",'')
            if location:
                sheet_str += f"Scene {scene_index}, Location: {location}\n"
            else:
                sheet_str += f"Scene {scene_id}\n"
            if scene_description:
                sheet_str += f"[{scene_description}]\n"

        elif last_location and last_location != location:
            sheet_str += f" Location: {location}\n"   # Some script's location is revised manually 
        sheet_str += f"({start}-{end}) {character}: {dialog}\n"
         ################# 对于特殊的script，没有description ##############
        if 'description' in result_sheet.columns:
            if description and temp_description != description:
                sheet_str += f"[{description}]\n"
                temp_description = description
         ################# 对于特殊的script，没有description ##############

        last_location = location
    return sheet_str

def script_to_str(file_path, boundingbox = True):
    script = pd.read_excel(file_path,sheet_name=None)
    result_sheet = script['results_human']
    if 'scripts' in script.keys():
        script_sheet = script['scripts']
    else:
        script_sheet = script['script']
    script_str = sheet_to_str(script_sheet, result_sheet, boundingbox = boundingbox)
    return script_str





fine_grained_des = """QAs for story video comprehension is divided into 2 question types,7 story element combinations.
Question type: P, I.
Description: For QAs of type P, it can be obtained from the appearance of video directly, while QAs of type I, need to analyze the content of the video and logical reasoning to get the results. Therefore, you need to focus on a more long-term understanding of the scripts, such as cross scene, even cross the whole script, generate less QAs about short-term understanding.
Story element combinations: C, A, L, CA, CL, AL, CAL. 
Description: We focus on 7 story element combinations of storylines. We focuses on 3 core story elements of the video: character, action, and location, where C stands for character, A stands for action, and L stands for location. Therefore, there are 7 possible story element combinations, i.e., 'C', 'A', 'L', 'CA', 'CL', 'AL', 'CAL'. It should be attention that there should be only story elements in QAs. For example, 'C' can only involve 'character', not anything about 'action' or 'location'. You can only generate QAs about 'character'. \n
"""
def prompt_engine(generated_questions ,video_info, prompt_type="qg", feedback = None):
    if prompt_type == "qg":
        prompt = "System: You are an expert in long story video comprehension and now need to come up with a series of question-answer pairs for them to answer. For each QA-pair you generate, please ignore any prior knowledge about these movies and TV series, and construct QA-pairs (QAs) based solely on the given video description."
        prompt += fine_grained_des
        prompt += f"Video description are as follow:\n{video_info}\n"        
        prompt += """QAs requirements:
(1) Video-only answerability: Please note that the generated QAs must be answerable after watching the video. Do not create QAs regarding information that is not visually or verbally conveyed in the video. Additionally, it is important to emphasize that, due to the length of the video, the QAs should be correspond uniquely to the plot of the video without ambiguity. Don’t give specific video time in QAs, because this reduces the difficulty of the QAs.
(2) Scope & detail: I hope relevant segments can cover long-from video understanding. I want to generate QAs of different difficulty, so for each type you need to help me generate QAs of different difficulty by either focusing on more long-from video clip, or focusing on more complex character’s relationship. You can even generate QAs cover the whole episode. And the generation of QAs should ideally be spread throughout the video rather than concentrated in certain scenes. In order to better validate the QA-pair, Please generate the basis of inference, a list of the start and end times of the relevant segments in the video, and a list of the relevant characters related to QA-pair.
(3) Tool & format compliance: Please carefully follow the instructions of the tool ‘Tools_of_saving_QAs_with_X_X’ to generate and save a series of QAs based on the following information for those types haven’t enough QAs. 
"""
        if generated_questions:
            prompt += "\nGenerated questions are as follow:\n\n"+ generated_questions + "\n"
        if feedback:
            prompt += f"\n\nFeedback from your supervisor is as follow, please generate questions according to these feedback:  {feedback}"
        return prompt
    elif prompt_type == "qj":
        prompt = "System: You are an expert in supervising the question answer pair (QAs)."
        prompt += fine_grained_des
        prompt += f"Video description are as follow:\n{video_info}\n"        
        prompt += "\nGenerated questions are as follow:\n\n"+ generated_questions + "\n"

        prompt += """Task:
(1) You need to delete invalid QAs and use “Tools of delete invalid QAs” to delete. Please accurately find out those invalid QAs and delete them. 
(2) You can retrieve similar QAs from fault archive to find out similar fault, by which you can summarize the feedback from the generator’s generated QAs history. 
(3) Give your feedback about the quality of these QAs for the improvements of generator. According to the retrieval results of the deleted QAs database and temporary state, summarize shortly and tell the generator which aspects are well done and should be promoted, which issues need improvement, and which mistakes should be remembered and not repeated.
"""
        return prompt
    else:
        return "To do."


