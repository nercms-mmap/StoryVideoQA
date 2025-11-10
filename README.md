# StoryVideoQA: Scaling Deep Video Understanding with a Large-Scale, Multi-Genre and Auto-Generated Dataset
Offical repository for "StoryVideoQA: Scaling Deep Video Understanding with a Large-Scale, Multi-Genre and Auto-Generated Dataset"



------

### StoryMindv2 for the construction of StoryVideoQA

![StoryMindv2](./assets/StoryMindv2.png)

**StoryMindv2 consists of 4 stage:**

- Data Preparation

  - The final result of script-subtitle alignment are directly stored in `StoryMindv2/aligned_script`,  i.e., 

```bash
StoryMindv2/
    └── aligned_script/
        ├── BigBang           # Aligned script-subtitle files of TV series: The BigBang Theory
        ├── Friends           # Aligned script-subtitle files of TV series: Friends
        ├── GOT               # Aligned script-subtitle files of TV series: GOT
        ├── Movie             # Aligned script-subtitle files of Movies from IMDB and Douban
        ├── Movie_ini         # Aligned script-subtitle files of Movies with Chinese file name
        └── video_length.json # Different video length of TV series/Movie in StoryVideoQA    
```

- QAs Generation

  -  This stage can be executed by running the script `sh sh/QAsGen.sh` directly, i.e., 

```bash
# Run the QAsGen script with specified parameters
CUDA_VISIBLE_DEVICES=0 python QAsGen.py \
        --gemini_model gemini-2.0-flash \                           
        --gemini_key "Replace with your Gemini API key" \           
        --gemini_proxy "Replace with your Gemini proxy address" \ 
        --each_type_num 100 \                    # Number of QAs for each fine-grained topic
        --vid_dir Friends                        # The script will use data from aligned_script/{vid_dir}
```

- QAs Filtration

  - This stage can be executed by running the script `sh sh/QAsFil.sh` directly, i.e., 

```bash
python QAsFil.py --openai_model gpt-4.1-2025-04-14 \
                 --openai_key "Replace with your GPT API key" \
                 --openai_proxy "Replace with your GPT proxy address" \
                 --gemini_model gemini-2.0-flash \
                 --gemini_key "Replace with your Gemini API key" \           
                 --gemini_proxy "Replace with your Gemini proxy address" \ 
                 --claude_model claude-3-7-sonnet-20250219 \
                 --claude_key "Replace with your Claude API key" \          
                 --claude_proxy "Replace with your Claude proxy address" \ 
                 --vid_dir Friends \   # The script will use data from aligned_script/{vid_dir} for Reviewer
                 --start 0 \           # Start index of generated csv
                 --end 100             # End index of generated csv

python export.py --vid_dir Friends \                           # Filter QAs based on Filtration result
                 --output_path json/filter_QAs.json \          # output path of filtered QAs
```

- Difficulty Measure
  - This stage can be executed by running the script `sh sh/QAsDiff.sh` directly, i.e., 


```bash
python diff_measure.py --questions_path json/filter_QAs.json \               # path of filtered QAs
          --output_path "json/all_questions_info_with_difficulty.json"       # output QAs with diff score
```








------

### PlotTree for Deep Video Understanding Task

![PlotTree](./assets/PlotTree.png)

