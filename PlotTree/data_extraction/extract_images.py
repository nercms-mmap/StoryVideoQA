import os
import cv2
from pathlib import Path
from tqdm import tqdm
import json
import argparse

def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)


def extract_es(video_dir, save_dir='data/frames', video_names=[]):
    input_base_path = Path(video_dir)
    output_base_path = Path(save_dir)
    fps = 1.0
    pbar = tqdm(total=len(list(input_base_path.iterdir())))
    for video_fp in sorted(input_base_path.iterdir())[:]:
        video_name = video_fp.stem
        # print(video_name)
        if len(video_names)==0 or video_name in video_names:
            output_path = output_base_path / video_fp.stem
            output_path.mkdir(parents=True, exist_ok=True)
            vidcap = cv2.VideoCapture(str(video_fp))
            count = 0
            success = True
            fps_ori = int(vidcap.get(cv2.CAP_PROP_FPS))   
            frame_interval = int(1 / fps * fps_ori)
            while success:
                success, image = vidcap.read()
                if not success:
                    break
                if count % (frame_interval) == 0 :
                    # print(f'{output_path}/{count}.jpg')
                    cv2.imwrite(f'{output_path}/{count}.jpg', image)
                count+=1
        pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    vid_dir = 'Movie'
    video_dir = f"data/video/{vid_dir}"
    save_dir = f"data/frames/{vid_dir}"
    video_names = ['IMDB-001-The Shawshank Redemption',
                'IMDB-002-The Godfather',
                'IMDB-004-The Dark Knight',
                'IMDB-032-Psycho',
                'IMDB-034-Rear Window',
                'IMDB-039-The Terminator',
                'IMDB-041-The Pianist',
                'IMDB-046-The Departed',
                'IMDB-211-Harry Potter 1'
                ]
    video_names = sorted(os.listdir(video_dir))
    video_names = [name.split('.')[0] for name in video_names]
    print(video_names)
    extract_es(video_dir, save_dir, video_names)


