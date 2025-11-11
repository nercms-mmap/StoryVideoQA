
SHARED_PARAMS="--plotTree_name PlotTree \
               --description_type full \
               --tree_output_base_dir results/PlotTree \
               --llm_model Gemini-2.0-flash \
               --compression_factor 36  \
               --temporal_weight 10 \
               --max_rag_nodes 32 \
               --openai_model gemini-2.0-flash \
               --openai_key "Replace with your LLM API Key" \
               --openai_proxy "Replace with your LLM API Proxy" \
               --json_ouput_dir results/QA"

CUDA_VISIBLE_DEVICES=0 python plotTreeqa.py --vid_dir Friends $SHARED_PARAMS

CUDA_VISIBLE_DEVICES=0 python plotTreeqa.py --vid_dir BigBang $SHARED_PARAMS

CUDA_VISIBLE_DEVICES=0 python plotTreeqa.py --vid_dir GOT $SHARED_PARAMS

CUDA_VISIBLE_DEVICES=0 python plotTreeqa.py --vid_dir Movie $SHARED_PARAMS
