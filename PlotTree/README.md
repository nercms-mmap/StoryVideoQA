3. ## ⚙️ PlotTree for Deep Video Understanding Task

   ![PlotTree](C:\Users\wzq\Desktop\IJCV-Github\StoryVideoQA-github\assets\PlotTree.png)

   
   
   ### Environment Setup

   Before running the code, please make sure to prepare the Python environment properly.
   
   1. **Install dependencies**
   
      We recommend creating a clean virtual environment (e.g., via `conda` or `venv`) and then installing the required dependencies:
   
      ```bash
      conda create --name PlotTree python=3.10
      conda activate PlotTree
      cd PlotTree
      pip install -r requirements.txt
      ```
   
   2. **Update `kmeans-pytorch` library  (Since we revise the distance for K-means)**
   
      ```bash
      cd kmeans_new
      git clone https://github.com/subhadarship/kmeans_pytorch
      cd kmeans_pytorch
      cp __init__.py kmeans_pytorch/kmeans_pytorch
      ```
   
      By doing this, you can replace the `__init__.py` file in the `kmeans_pytorch` directory with the modified version we provided in `./kmeans_new` of this repository.
   
      Then, install the updated package locally:
   
      ```bash
      pip install --editable .
      ```
   
   ### Video Data Preparation
   
   - Due to copyright restrictions associated with TV series and movies, researchers are therefore required to **obtain the relevant videos independently** and place them in the `data/video` directory. In addition, please download and unzip the character library from [Character.zip · ZQFive/StoryVideoQA](https://huggingface.co/datasets/ZQFive/StoryVideoQA/blob/main/Character.zip)  to  `data/Character` directory.
   - Once the videos are prepared, you can extract keyframes using the script `data_extraction/extract_images.py`, which samples frames at a default rate of `1 fps`.
   - After obtaining keyframes, you can perform plot captioning using the provided shell script `sh/cap.sh`.
   
   For convenience, we have provided **pre-generated plot captioning results** for the *StoryVideoQA-G* subset in the `data/captions` directory.
   
   
   
   ### PlotTree Construction and PlotTreeQA
   
   1. **PlotTree Construction**
   
      This stage can be executed by running the script `sh sh/plotTree.sh` directly, i.e.,
   
      ```bash
      SHARED_PARAMS="--output_base_dir results/PlotTree \
                     --llm_model Gemini-2.0-flash \
                     --openai_model gemini-2.0-flash \
                     --openai_key "Replace with your LLM API Key" \
                     --openai_proxy "Replace with your LLM API Proxy" \
                     --compression_factor 36 \    # compression rate between two layers
                     --temporal_weight 10 \       # scaling factor of distance function for K-means
                     --min_nodes_in_cluster 2"
      
      
      CUDA_VISIBLE_DEVICES=0 python plotTree.py --captions_path data/captions/Friends_PlotTree.json $SHARED_PARAMS
      CUDA_VISIBLE_DEVICES=0 python plotTree.py --captions_path data/captions/BigBang_PlotTree.json $SHARED_PARAMS
      CUDA_VISIBLE_DEVICES=0 python plotTree.py --captions_path data/captions/GOT_PlotTree.json $SHARED_PARAMS
      CUDA_VISIBLE_DEVICES=0 python plotTree.py --captions_path data/captions/Movie_PlotTree.json $SHARED_PARAMS
      ```
   
   2. **PlotTree QA**
   
      This stage can be executed by running the script `sh sh/plotTreeqa.sh` directly, i.e.,
   
      ```bash
      SHARED_PARAMS="--plotTree_name PlotTree \
                     --description_type full \
                     --tree_output_base_dir results/PlotTree \
                     --llm_model Gemini-2.0-flash \
                     --compression_factor 36  \   # compression rate between two layers
                     --temporal_weight 10 \       # scaling factor of distance function for K-means
                     --max_rag_nodes 32 \         # rag node on PlotTree
                     --openai_model gemini-2.0-flash \
                     --openai_key "Replace with your LLM API Key" \
                     --openai_proxy "Replace with your LLM API Proxy" \
                     --json_ouput_dir results/QA"
      
      CUDA_VISIBLE_DEVICES=0 python plotTreeqa.py --vid_dir Friends $SHARED_PARAMS
      CUDA_VISIBLE_DEVICES=0 python plotTreeqa.py --vid_dir BigBang $SHARED_PARAMS
      CUDA_VISIBLE_DEVICES=0 python plotTreeqa.py --vid_dir GOT $SHARED_PARAMS
      CUDA_VISIBLE_DEVICES=0 python plotTreeqa.py --vid_dir Movie $SHARED_PARAMS
      ```
   
   3. Calculate Metrics 
   
      If use our default setting of PlotTree, you can directly use following script to calculate metrics for PlotTree
   
      ```bash
      python metrics.py
      ```
   
      
   
   