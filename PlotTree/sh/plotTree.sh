SHARED_PARAMS="--output_base_dir results/PlotTree \
               --llm_model Gemini-2.0-flash \
               --openai_model gemini-2.0-flash \
               --openai_key "Replace with your LLM API Key" \
               --openai_proxy "Replace with your LLM API Proxy" \
               --compression_factor 36 \
               --temporal_weight 10 \
               --min_nodes_in_cluster 2"


CUDA_VISIBLE_DEVICES=0 python plotTree.py --captions_path data/captions/Friends_PlotTree.json $SHARED_PARAMS

CUDA_VISIBLE_DEVICES=0 python plotTree.py --captions_path data/captions/BigBang_PlotTree.json $SHARED_PARAMS

CUDA_VISIBLE_DEVICES=0 python plotTree.py --captions_path data/captions/GOT_PlotTree.json $SHARED_PARAMS

CUDA_VISIBLE_DEVICES=0 python plotTree.py --captions_path data/captions/Movie_PlotTree.json $SHARED_PARAMS