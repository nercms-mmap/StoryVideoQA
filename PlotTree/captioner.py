import os
import cv2
import json
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
from tqdm import tqdm
import pandas as pd
import pickle
import numpy as np
import ast
import srt
from typing import Dict, Any, List

import os
import json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, List, Tuple
import argparse


# 假设 insightface 和其他库已安装
import insightface
from insightface.app import FaceAnalysis


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return "{:02d}:{:02d}:{:02d}".format(hours,minutes,seconds)

def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        json_str = file.read()
    data = json.loads(json_str)
    return data

def save_json(fn,data, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)



class InsightfaceRecognizer:
    def __init__(self, 
                 app: FaceAnalysis, 
                 profile_dir: str, 
                 charbank_path: str,
                 video_to_imdb_json_path: str,
                 tv_char_base_dir: str,
                 tv_names: List[str] = ['Friends', 'BigBang', 'GOT'],
                 cache_path: str = "feature_database.pkl"
                 ):
        """
        Args:
            app (FaceAnalysis): a insightface.app.FaceAnalysis object。
            movie_profile_dir (str): movie character directory: a image per character。
            charbank_path (str): charbank JSON path,  {'movie_imdbid':[{'id', 'name', 'role'}]... }
            video_to_imdb_json_path (str): a json contains name to imdbid:   {'movie_name': 'movie_imdbid',...}

            Fot TV video: each character has more than one images, so we use directory to storage them.
            tv_char_base_dir (str):  character bank's base directory
            tv_names: List[str], all TV names

            cache_path: str = "cache database of character feature"
        """
        print("Initializing Insightface-based Recognizer...")
        self.app = app
        self.profile_dir = profile_dir
        self.colors = ["red", "green", "blue", "yellow", "magenta", "cyan"]

        # 加载映射文件和角色库
        with open(video_to_imdb_json_path, 'r') as f:
            self.video_to_imdb_map = json.load(f)
        with open(charbank_path, 'r') as f:
            self.charbank_dict = json.load(f)
        self.inverted_video_to_imdb_map = {v: k for k, v in self.video_to_imdb_map.items()}
        self.tv_char_base_dir = tv_char_base_dir
        self.tv_names = tv_names
        self.cache_path = cache_path
        # --- 缓存加载逻辑 ---
        if os.path.exists(self.cache_path):
            print(f"Loading known faces from cached file: {self.cache_path}")
            self.known_faces_db = self._load_db_from_cache()
        else:
            print("No cache found. Building feature database from scratch...")
            # 1. 从零开始构建数据库
            self.known_faces_db = self._build_db_from_scratch()
            # 2. 将新建的数据库保存到缓存
            self._save_db_to_cache()
            print(f"Feature database built and saved to cache: {self.cache_path}")

        print("Recognizer ready.")
    def _load_db_from_cache(self):
        """从 pickle 文件加载特征数据库。"""
        with open(self.cache_path, 'rb') as f:
            db = pickle.load(f)
        for imdbid in db:
            if isinstance(db[imdbid].get("features"), torch.Tensor):
                db[imdbid]["features"] = db[imdbid]["features"].cuda()
        return db
    
    def _save_db_to_cache(self):
        """将特征数据库保存到 pickle 文件。"""
        db_to_save = {}
        for imdbid, data in self.known_faces_db.items():
            db_to_save[imdbid] = data.copy() # 浅拷贝
            if isinstance(db_to_save[imdbid].get("features"), torch.Tensor):
                db_to_save[imdbid]["features"] = db_to_save[imdbid]["features"].cpu()
        
        with open(self.cache_path, 'wb') as f:
            pickle.dump(db_to_save, f)
    
    def _build_db_from_scratch(self) -> Dict:
        """ 提取并存储所有已知角色的面部特征。"""
        db = {}
        print("Building known faces feature database from imdb character bank...")
        
        all_profile_names = {name for name in os.listdir(self.profile_dir) if ".jpg" in name}

        for imdbid, cast_info in self.charbank_dict.items():
            print(self.inverted_video_to_imdb_map[imdbid]+ f", {len(cast_info)}")
            cast_features, cast_ids, cast_names = [], [], []
            for char_info in cast_info:
                char_id = char_info['id']
                if f"{char_id}.jpg" in all_profile_names:
                    try:
                        # img_path = os.path.join(self.profile_dir, f"{char_id}.jpg")
                        # image_np = np.array(Image.open(img_path).convert("RGB"))
                        image = np.array(Image.open(os.path.join(self.profile_dir, char_id+ ".jpg")).convert("RGB"))
                        faces = self.app.get(image)
                        if faces:
                            # 标准化 embedding
                            feat = torch.tensor(faces[0]['embedding'])
                            feat_norm = feat / feat.norm(dim=-1, keepdim=True)
                            cast_features.append(feat_norm)
                            cast_ids.append(char_id)
                            cast_names.append(char_info.get('role', 'Unknown'))
                    except Exception as e:
                        print(f"Warning: Failed to process profile for {char_id}: {e}")
        
            if cast_features:
                db[imdbid] = {
                    "features": cast_features, # 将特征放到GPU上以加速计算
                    "names": cast_names,
                    # "ids": cast_ids,
                }
                # print(f"  - Loaded {len(cast_ids)} characters for IMDB ID: {imdbid}")

        self.tv_characters_bank_dir = []
        for tv_name in self.tv_names:
            # print(f"Processing character bank of {tv_name}...")
            tv_characters_dir = os.path.join(self.tv_char_base_dir, tv_name)
            imdbid = self.video_to_imdb_map[tv_name]
            num_char =0
            char_features_temp = {} 
            for character_name in tqdm(os.listdir(tv_characters_dir)):
                character_dir = os.path.join(tv_characters_dir, character_name)
                # 临时存储每个角色的所有特征向量
                char_features_temp[character_name] = []
                num_char += len(os.listdir(character_dir))
                for image_name in os.listdir(character_dir):
                    image_path = os.path.join(character_dir, image_name)
                    try:
                        image = np.array(Image.open(image_path).convert("RGB"))
                        faces = self.app.get(image)
                        if faces:
                            feat = torch.tensor(faces[0]['embedding'])
                            char_features_temp[character_name].append(feat)
                    except Exception as e:
                        print(f"Warning: Failed to process image {image_path}: {e}")
            print(tv_name+f", {num_char}")
            if imdbid not in db:
                db[imdbid] = {"features": [], "names": []}
                
            for char_name, feats_list in char_features_temp.items():
                if feats_list:
                    # 将一个角色的所有特征向量求平均，得到一个更鲁棒的特征
                    avg_feat = torch.stack(feats_list).mean(dim=0)
                    # 对平均后的特征进行归一化
                    avg_feat_norm = avg_feat / avg_feat.norm(dim=-1, keepdim=True)
                    
                    # 将最终的特征和名字添加到数据库
                    db[imdbid]["features"].append(avg_feat_norm)
                    db[imdbid]["names"].append(char_name)
            if char_features_temp:
                print(f"    - Aggregated and loaded {len(char_features_temp)} characters for '{tv_name}'.")

        print("\nFinalizing feature database...")
        for imdbid in db.keys():
            # 将 list of tensors 堆叠成一个 tensor，并放到 GPU
            db[imdbid]["features"] = torch.stack(db[imdbid]["features"]).cuda()

        return db

    def recognize_and_draw(self, image: Image.Image, video_face_key: str, score_thresh: float = 0.25) -> Tuple[Image.Image, str]:
        """
        对单张图片进行识别和绘制。
        Args:
            image (Image.Image): 输入的PIL图片。
            video_face_key (str): 当前图片所属的视频/剧集名 (用于查找imdbid)。
            score_thresh (float): 识别的余弦相似度阈值。
        Returns:
            Tuple[Image.Image, str]: 带标注的图片 和 描述文本。
        """

        imdbid = self.video_to_imdb_map.get(video_face_key)
        if not imdbid or imdbid not in self.known_faces_db:
            # print("========Return without Face Recognition========")
            return image, "", ""
        # print(f"Face Recognition is based on {video_face_key}:{imdbid} face database.")
        known_data = self.known_faces_db[imdbid]
        known_features = known_data["features"]
        known_names = known_data["names"]
        # print(known_names)
        img_array = np.array(image.convert("RGB"))
        faces_in_frame = self.app.get(img_array)
        if not faces_in_frame:
            print("No Face Found !!!")
            return image, "", ""
        print("Face Found !!!")
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        recognized_people = {} # {name: color}

        for face in faces_in_frame:
            bbox = face['bbox']
            feat = torch.tensor(face['embedding']).cuda()
            feat_norm = feat / feat.norm(dim=-1, keepdim=True)
            
            # 计算余弦相似度
            cos_sim = feat_norm @ known_features.transpose(0, 1)
            best_match = torch.max(cos_sim, -1)
            
            if best_match[0].item() > score_thresh:
                pred_idx = best_match[1].item()
                name = known_names[pred_idx]
                
                # 分配颜色并绘制
                if name not in recognized_people:
                    color = self.colors[len(recognized_people) % len(self.colors)]
                    recognized_people[name] = color
                
                color_to_draw = recognized_people[name]
                left, top, right, bottom = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                draw.rectangle(((left, top), (right, bottom)), outline=color_to_draw, width=3)
                # ... (省略了标签绘制代码以保持简洁，可以像之前一样添加)

        # 3. 生成描述字符串
        char_text = ", ".join([f"{name} ({color})" for name, color in recognized_people.items()])
        characters_info = ", ".join([f"{name}" for name, color in recognized_people.items()])
        print(char_text)
        return annotated_image, char_text, characters_info

# New method to get subtitles for a specific scene
def get_subtitles_for_scene(temp_second,  subtitles: List[srt.Subtitle]) -> str:
    """
    Extracts subtitle text for a given scene based on timecodes.
    """
    scene_start_seconds = temp_second
    scene_end_seconds = temp_second
    scene_subtitles = []
    for sub in subtitles:
        sub_start_seconds = sub.start.total_seconds()
        sub_end_seconds = sub.end.total_seconds()
        # print(sub_start_seconds, sub_end_seconds)
        if sub_start_seconds<= temp_second and temp_second <= sub_end_seconds:
            scene_subtitles.append(sub.content)
    return " ".join(scene_subtitles)



def plot_captioning(image_path, video_face_key, subtitle, recognizer) -> str:
    original_image = Image.open(image_path)
    # Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image") 

    annotated_image, face_info_text, characters_info = recognizer.recognize_and_draw(original_image, video_face_key)
    save_base_dir = "/mnt/disk6new/wzq/repetition/VideoTree/data/annotated_images"
    try:
        path_components = image_path.split(os.sep)[-3:]
        
        save_path = os.path.join(save_base_dir, *path_components)
        
        save_directory = os.path.dirname(save_path)
        os.makedirs(save_directory, exist_ok=True)
        
        annotated_image.save(save_path)

    except Exception as e:
        print(f"Warning: Could not save annotated image for {image_path}. Error: {e}")

    # 最基本的Caption方法
    base_prompt_text = "Please generate a brief caption of this image (Describe 'Who' is doing 'What' in 'Where')"
    final_prompt_text = f"{base_prompt_text}. Recognized faces are annotated by boundingbox with color: {face_info_text}. Directly use character's name to narrate, and don't use boundingbox's color." if face_info_text else base_prompt_text
    # final_prompt_text = base_prompt_text 
    final_prompt_text = final_prompt_text + f" Subtitle: {subtitle}" if subtitle else final_prompt_text
    conversation = [
        {
        "role": "user",
        "content": [
            {"type": "text", "text": final_prompt_text},
            {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=annotated_image, text=prompt, return_tensors="pt").to("cuda:0")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=128)
    caption = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).replace("\"", "")
    return caption, face_info_text, characters_info 


def captioning(image_path, video_face_key, subtitle, recognizer) -> str:
    original_image = Image.open(image_path)
    face_info_text, characters_info = "", "" 
    # 最基本的Caption方法
    base_prompt_text = "Please generate a brief caption of this image (Describe 'Who' is doing 'What' in 'Where')"
    final_prompt_text = base_prompt_text
    conversation = [
        {
        "role": "user",
        "content": [
            {"type": "text", "text": final_prompt_text},
            {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=original_image, text=prompt, return_tensors="pt").to("cuda:0")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=128)
    caption = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).replace("\"", "")
    return caption, face_info_text, characters_info 

# video_face_key决定了使用哪个人脸库（GOT是权游，Friends是老友记，BigBang是生活大爆炸，电影类似于IMDB-XXX-XX的形式）
def caption_dir(images_dir, video_face_key, srt_path, fps, recognizer, captype):
    image_names_list = sorted(os.listdir(images_dir), key=lambda x: int(x[:-4]))
    captions_list = []

    with open(srt_path, 'r', encoding='utf-8') as f:
        subtitles = list(srt.parse(f.read()))

    for id, name in enumerate(tqdm(image_names_list, desc="Captioning frames")):
        image_path = os.path.join(images_dir, name)
        second = int(name.split(".")[0])/fps
        subtitle = get_subtitles_for_scene(second,  subtitles)
        if captype != 'VideoTree':
            image_caption, face_info_text, characters_info  = plot_captioning(image_path, video_face_key, subtitle, recognizer)
        else:
            image_caption, face_info_text, characters_info  = captioning(image_path, video_face_key, subtitle, recognizer)
        # if subtitle:
        #     image_caption += f" Subtitle: {subtitle}"
        print(name, image_caption)
        print(face_info_text, characters_info)
        print(format_time(second),":", subtitle)
        caption_dict = {'caption': image_caption, 'image_name': name.split(".")[0], "time": format_time(second), 'subtitle': subtitle, 'face_bbox': face_info_text, 'charcters_info': characters_info}
        captions_list.append(caption_dict)
    return captions_list


def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)



processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf",cache_dir="/mnt/disk6new/wzq/ckpt")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True, cache_dir="/mnt/disk6new/wzq/ckpt") 
model.to("cuda")

if __name__== "__main__":
    arg_parser = argparse.ArgumentParser()
    # primary setting
    arg_parser.add_argument('--vid_dir', type=str, default="Movie")
    arg_parser.add_argument('--captype', type=str, default="PlotTree_test")
    args = arg_parser.parse_args()
    

    fpsjson = load_json("data/videofps.json") 
    print("Initializing Insightface FaceAnalysis model...")
    face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    print("Insightface model ready.")

    PROFILE_DIR = "data/Character/Movie" # 存放角色肖像照
    CHARBANK_PATH = "data/Character/Movie.json" # 角色库
    VIDEO_TO_IMDB_PATH = "data/name2imdbid.json" # 视频名到imdbid的映射
    TV_CHAR_BASE_DIR = "data/Character"

    recognizer = InsightfaceRecognizer(
        app=face_app,
        profile_dir=PROFILE_DIR,
        charbank_path=CHARBANK_PATH,
        video_to_imdb_json_path=VIDEO_TO_IMDB_PATH,
        tv_char_base_dir = TV_CHAR_BASE_DIR,
        tv_names = ['Friends', 'BigBang', 'GOT']
    )

    frames_base_dir =  "data/frames"
    videotype = args.vid_dir
    captype =  args.captype
    Datasets_caption = dict()
    json_path = f"data/captions/{videotype}_{captype}.json"
    last_vid = None
    if os.path.exists(json_path):
        Datasets_caption = load_json(json_path)
        name_split = list(Datasets_caption.keys())[-1].split("-")
        if 'Movie' == name_split[0]:
            last_vid = "-".join(list(Datasets_caption.keys())[-1].split("-")[1:])
        else:
            last_vid = "-".join(list(Datasets_caption.keys())[-1].split("-")[1:])

    all_video_names =sorted( os.listdir(os.path.join(frames_base_dir, videotype)) )
    video_names = [vid.split(".")[0]  for vid in all_video_names]
    print(video_names)
    for vid in sorted(os.listdir(os.path.join(frames_base_dir, videotype))):
        if last_vid and last_vid >= vid:
            continue
        if not vid in video_names:
            continue
        print("Processing:", vid)
        if videotype == 'Movie':
            srt_path = os.path.join(f"data/srt/{videotype}", vid+".srt")
        else:
            srt_path = os.path.join(f"data/srt/{videotype}", f"{videotype}-"+vid+".srt")
        
        fps_ori = fpsjson[f"{videotype}-{vid}"]
        print("=====================",fps_ori,"=====================")
        video_frames_path = os.path.join(frames_base_dir, videotype, vid)
        if videotype == 'Movie':
            video_face_key =  vid
        else:
            video_face_key = videotype
        captions_list = caption_dir(video_frames_path, video_face_key, srt_path, fps_ori, recognizer, captype)
        Datasets_caption[f'{videotype}-{vid}'] = captions_list
        save_json(Datasets_caption, f"data/captions/{videotype}_{captype}.json")