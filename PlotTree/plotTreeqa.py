# 补充
import os
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from utils.localllm import QwenChatbot, get_llm, chat


os.environ['SENTENCE_TRANSFORMERS_HOME'] = "ckpt"

def read_json(json_path):
    data = None
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as file:
            json_str = file.read()
            # Use the json.loads() method to parse JSON strings.
            data = json.loads(json_str)
    return data

def save_json(json_path, data):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

def choices_str(choice_list):
    choices = ""
    for i, choice in enumerate(choice_list):
        choices += "(" + chr(65+i)+")" + f" {choice}\n"    # 65 stands for char 'A'
    return choices



def get_embedding_from_text(text: str, model: SentenceTransformer) -> torch.Tensor:
    return model.encode([text], convert_to_tensor=True, device=model.device, prompt_name="query").squeeze(0)


def cosine_similarity_distance(points, centroid):
    """
    Calculate cosine similarity between points and centroid.
    Returns the cosine distances (1 - similarity).
    """
    # Cast to float32 to ensure compatibility with CPU operations
    points = points.to(torch.float32)
    centroid = centroid.to(torch.float32)

    points_normalized = F.normalize(points, dim=1)
    centroid_normalized = F.normalize(centroid.unsqueeze(0), dim=1)
    return 1 - torch.mm(points_normalized, centroid_normalized.T).squeeze()

chatbot = None
llm = None
def get_llm_qa(prompt, llm_model, args) -> str:
    global chatbot, llm
    if llm_model == "Qwen3-30B-A3B":
        if not chatbot:
            chatbot = QwenChatbot(model_name="Qwen/"+"Qwen3-30B-A3B", cache_dir = "/mnt/disk6new/wzq/LLM/ckpt")
            print(f"Loading {llm_model} first Time.")
        summary = chatbot.generate_response(prompt)
        # print(f"{llm_model}: {summary}")
    else:
        if not llm:
            llm = get_llm(llm_model, args)
            print(f"Loading {llm_model} first Time.")
        summary = chat(llm, question=prompt)
         # print(f"{llm_model}: {summary}")
    if summary.startswith(prompt):
        summary = summary[len(prompt):].strip()
    # if len(summary) > 500: summary = summary[:500] + "..."
    return summary

def retrieve_context_with_bnsr(
    question_embedding: torch.Tensor,
    plot_tree_nodes_list: list[dict],
    candidate_embeddings: torch.Tensor, # rag_embedding_model: SentenceTransformer,
    max_rag_nodes : int =10,
    max_context_tokens: int = 2000
) -> list[str]:
    """
    Perform RAG contextual retrieval directly on the tree.

    Args:
        question_embedding (torch.Tensor): The query is embedded in (1, D).
        plot_tree_nodes_list (list[dict]): The entire list of nodes in the PlotTree.
        rag_embedding_model (SentenceTransformer): Used to embed nodes into devices.
        max_context_tokens (int): The maximum number of tokens in the final recall context.

    Returns:
        list[str]: A list of recalled PlotSummary texts, arranged chronologically.
    """
    if not plot_tree_nodes_list:
        return []

    node_map = {node["node_id"]: node for node in plot_tree_nodes_list}

    # Calculate the distance between the problem embedding and the embeddings of all candidate nodes.
    distances = cosine_similarity_distance(candidate_embeddings, question_embedding) # (N_candidates,)
    sorted_distances, sorted_indices = torch.sort(distances)
    top_k_indices = sorted_indices[:max_rag_nodes]
    collected_summaries_with_info = [] # [(start_frame_idx, plot_summary)]
    for idx in top_k_indices:
        node = plot_tree_nodes_list[int(idx.item())] # Assuming plot_tree_nodes_list maintains the original node order
        collected_summaries_with_info.append((node["node_id"], node["start_frame_idx"], node["plot_summary"]))


    # Context integration and sorting, and limiting the number of tokens.
    collected_summaries_with_info.sort(key=lambda x: x[1]) # Finally sorted by the starting frame index
    final_context_summaries = []
    for _, _, summary_text in collected_summaries_with_info:
        final_context_summaries.append(summary_text)

    return final_context_summaries, collected_summaries_with_info


def retrieve_context_with_bnsrv2(
    question_embedding: torch.Tensor,
    plot_tree_nodes_list: list[dict],
    candidate_embeddings: torch.Tensor, # rag_embedding_model: SentenceTransformer,
    max_rag_nodes : int =10,
    max_context_tokens: int = 2000,
    level_retrieve_counts:dict = None,
) -> list[str]:
    """
    Perform RAG contextual retrieval directly on the tree.

    Args:
        question_embedding (torch.Tensor): The query is embedded in (1, D).
        plot_tree_nodes_list (list[dict]): The entire list of nodes in the PlotTree.
        rag_embedding_model (SentenceTransformer): Used to embed nodes into devices.
        max_context_tokens (int): The maximum number of tokens in the final recall context.
    Returns:
        list[str]: A list of recalled PlotSummary texts, arranged chronologically.
    """
    if not plot_tree_nodes_list:
        return []

    node_map = {node["node_id"]: node for node in plot_tree_nodes_list}

    # Calculate the distance between the problem embedding and the embeddings of all candidate nodes
    distances = cosine_similarity_distance(candidate_embeddings, question_embedding) # (N_candidates,)
    sorted_distances, sorted_indices = torch.sort(distances)
   
    collected_nodes = {} # Store the number of nodes collected at each level
    final_selected_node_indices = set() # Store the index of the finally selected node in plot_tree_nodes_list
    # Traverse the sorted indices and attempt to collect nodes for each level
    if level_retrieve_counts:
        for idx in sorted_indices:
            node_idx = int(idx.item())
            node = plot_tree_nodes_list[node_idx]
            node_level = node.get("level", 0) # 假设默认level为0

            # If this level has not yet reached the target number, and this node has not been selected yet.
            if collected_nodes.get(node_level, 0) < level_retrieve_counts.get(node_level, 1):
                if node_idx not in final_selected_node_indices:
                    final_selected_node_indices.add(node_idx)
                    collected_nodes[node_level] = collected_nodes.get(node_level, 0) + 1
        
            # Check if the retrieval count for all levels has been met and if max_rag_nodes has been reached.
            if len(final_selected_node_indices) >= max_rag_nodes:
                    # all_levels_met = all(collected_nodes.get(lvl, 0) >= count for lvl, count in level_retrieve_counts.items())
                    # if all_levels_met:
                break
    # If the number of nodes retrieved through hierarchical retrieval is less than max_rag_nodes, then supplement with the remaining most similar nodes.
    # In default setting, we only use this to retrieve nodes
    if len(final_selected_node_indices) < max_rag_nodes:
        remaining_count = max_rag_nodes - len(final_selected_node_indices)
        for idx in sorted_indices:
            node_idx = int(idx.item())
            if node_idx not in final_selected_node_indices:
                final_selected_node_indices.add(node_idx)
                remaining_count -= 1
            if remaining_count == 0:
                break

    collected_summaries_with_info = []
    nodel_levels = dict()
    for node_idx in final_selected_node_indices:
        node = plot_tree_nodes_list[node_idx]
        if node['level'] in nodel_levels:
            nodel_levels[node['level']].append(node["node_id"])
        else:
            nodel_levels[node['level']] = [node["node_id"]]
        collected_summaries_with_info.append((node["node_id"], node["start_frame_idx"], node["level"], node["plot_summary"]))
    level_keys = sorted(list(nodel_levels.keys()))
    for key in level_keys:
        print(f"Level {key} extract nodes:",len(nodel_levels[key]))

    # 3. Context integration and sorting, and limiting the number of tokens.
    collected_summaries_with_info.sort(key=lambda x: (x[1], -x[2])) # Finally, sort by the starting frame index, and prioritize frames with higher levels.
    print("Extract summary", len(collected_summaries_with_info), collected_nodes)
    final_context_summaries = []
    for _, node_index, node_level, summary_text in collected_summaries_with_info:
        final_context_summaries.append((node_index,summary_text))

    return final_context_summaries, collected_summaries_with_info


plot_tree_nodes_list_dict = dict()
candidate_embeddings_dict = dict()

def extract_option(response):
    response = response.strip().upper()
    if len(response) > 1 and response[0] in ['A', 'B', 'C', 'D', 'E'] and response[1] in ['.', ')', ' ']:
        response = response[0]
        
    elif response not in ['A', 'B', 'C', 'D', 'E']:
        tqdm.write(f"[Not Direct Answer] LLM response format unexpected: {response}. Attempting to extract option.")
        import re
        match = re.search(r'\(([A-E])\)', response)
        if match:
            response = match.group(1)
        else:
            response = response # Unrecognized answer
    return response

def plot_tree_qa(args, question_text: str, choices: list[str], episode_id: str, 
                 rag_embedding_model: SentenceTransformer, llm_model: str,  tree_output_base_dir: Path,
                 max_rag_nodes: int = 5) -> str:
    """
    Answer questions by retrieving pre-built PlotTrees.

    Args:
        question_text (str): Original question text.
        choices (list[str]): Options to the question.
        episode_id (str): The episode ID of the corresponding video (e.g., "Friends-S01E01").
        rag_embedding_model (SentenceTransformer): A model for encoding problems and node embedding.
        llm_model (str): The name of the local LLM model used to generate the answer.

        max_rag_nodes (int): The maximum number of PlotTree nodes that can be recalled during retrieval.

    Returns:
        str: The answer generated by LLM (usually options A/B/C/D/E).
    """
    global plot_tree_nodes_list_dict, candidate_embeddings_dict
    
    if episode_id in plot_tree_nodes_list_dict:
        plot_tree_nodes_list = plot_tree_nodes_list_dict[episode_id]
        candidate_embeddings = candidate_embeddings_dict[episode_id]
    else:
        # 1. Load the PlotTree node for the corresponding episode.
        if episode_id.split("-")[0] in ['Friends', 'BigBang', 'GOT']:
            plot_tree_path = tree_output_base_dir / f"{episode_id}.json" 
        else:
            plot_tree_path = tree_output_base_dir / f"Movie-{episode_id}.json" 
        if not os.path.exists(plot_tree_path):
            print("####################################################################################")
            print(f"#")
            print(f"#    Please generate PlotTree at {plot_tree_path}")
            print(f"#")
            print("####################################################################################")
        plot_tree_nodes_list = read_json(plot_tree_path)
        # If it is not a PlotTree, use the one-dimensional node strategy of Video2RAG.
        if not 'PlotTree' in args.plotTree_name:
            new_node_list = []
            print(args.plotTree_name, "Only extract first level nodes !!!")
            for node in plot_tree_nodes_list:
                if node['level']==0:
                    new_node_list.append(node)
            plot_tree_nodes_list = new_node_list
        # Ablation study of PlotTree on plot captioning
        if args.description_type !='full':
            new_node_list = []
            print(args.plotTree_name, "Only extract summary text, not character and subtitle for all nodes !!!")
            for node in plot_tree_nodes_list:
                if len(node['plot_summary']) != len(node['plot_summary'].split(". Spoken dialogue:")[0]):
                    print(node['level'], "truncate spoken dialogue and character +1")
                node['plot_summary'] = node['plot_summary'].split(". Spoken dialogue:")[0]
                new_node_list.append(node)
            plot_tree_nodes_list = new_node_list


        candidate_nodes = [node for node in plot_tree_nodes_list if node is not None]
        nodes_summary = []
        for node in candidate_nodes:
            plot_summary_full = node['plot_summary']
            if len(plot_summary_full) < 1000:
                nodes_summary.append(plot_summary_full)
            else:
                print("####################################################################################")
                print(f"#")
                print(f"#    Too long Context, Cut to 1000 words.")
                print(f"#")
                print("####################################################################################")
                nodes_summary.append(" ".join(plot_summary_full.split()[:1000]) )
                                     
        candidate_embeddings = rag_embedding_model.encode(nodes_summary, convert_to_tensor=True, device=rag_embedding_model.device)
        plot_tree_nodes_list_dict, candidate_embeddings_dict = dict(), dict()
        plot_tree_nodes_list_dict[episode_id] = plot_tree_nodes_list
        candidate_embeddings_dict[episode_id] = candidate_embeddings
        
    question_embedding = get_embedding_from_text(question_text+ choices_str(choices), rag_embedding_model)


    retrieved_context_summaries = []
    retrieved_node_ids = []
    retrieved_start_frames = []

    # --- MODIFICATION: Using the new retrieval logic --- ---
    retrieved_context_summaries, node_list = retrieve_context_with_bnsrv2(
        question_embedding=question_embedding,
        plot_tree_nodes_list=plot_tree_nodes_list,
        candidate_embeddings=candidate_embeddings, max_rag_nodes = max_rag_nodes, level_retrieve_counts=None
    )

    # retrieved_nodes_info = [] # [(start_frame_idx, plot_summary)]
    # for idx in top_k_indices:
    #     node = plot_tree_nodes_list[idx.item()] 
    #     retrieved_nodes_info.append((node["start_frame_idx"], node["plot_summary"]))
    
    # retrieved_nodes_info.sort(key=lambda x: x[0]) # 按起始帧索引排序

    # for _, summary in retrieved_nodes_info:
    #     retrieved_context_summaries.append(summary)

    # 4. Construction of Prompt
    context_str = "\n".join([f"({index}): {text}" for index, text in retrieved_context_summaries])
    
    # Assembly issues and options based on LLM's desired format.
    context = f"You are presented with a textual description of a video clip, it consists of frame captions sparsely sampled from the video. Your task is to answer a question solely based on these textual description, choosing the correct option out of five possible answers. Please provide the answer with a single-letter (A, B, C, D, E) \n\n###\n\n  Description: \n{context_str}\n\n"
    question = f"{context}\nQuestion: {question_text}\n\nPlease select one best option from following choices directly:\n" + choices_str(choices) + "\nAnswer: ("
    

    
    response = get_llm_qa(question, llm_model, args)
    # print("initial_response",response)
    # Extracting the option letters from LLM answers
    response = response.strip().upper()
    response = extract_option(response)


    return response, node_list 


    
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    # primary setting
    arg_parser.add_argument('--llm_model', type=str, default="GPT4.1")
    arg_parser.add_argument('--compression_factor', type=int, default=36)
    arg_parser.add_argument('--temporal_weight', type=float, default=10)
    arg_parser.add_argument('--plotTree_name', type=str, default="PlotTree")
    arg_parser.add_argument('--description_type', type=str, default="full")
    arg_parser.add_argument('--vid_dir', type=str, default="Friends")
    arg_parser.add_argument('--tree_output_base_dir', type=str, default="results/PlotTree", help="Base directory where PlotTree JSONs are stored.")  
    arg_parser.add_argument('--max_rag_nodes', type=int, default=150, help="Max number of plot tree nodes to retrieve for RAG.") # 新增参数
    arg_parser.add_argument("--openai_key", required=True, type=str, help="key for llm")
    arg_parser.add_argument("--openai_proxy", required=True, type=str, help="proxy for llm")
    arg_parser.add_argument("--openai_model", default="gemini-2.0-flash",type=str, help="model name of llm")
    arg_parser.add_argument("--json_ouput_dir", default="results/QA", type=str, help='questions_dict_output_dir')
    args = arg_parser.parse_args()

    

    vid_dir = args.vid_dir
    model_name = args.plotTree_name
    json_path = f"data/golden_questions/{vid_dir}_golden.json"
    # loading PlotTree
    tree_output_base_dir = Path(f"{args.tree_output_base_dir}/{args.llm_model}_{args.compression_factor}_{args.temporal_weight}")
    print("PlotTree Loading Directory: ", tree_output_base_dir)
    tree_output_base_dir.mkdir(parents=True, exist_ok=True)
    hyper_insert_save_dir = f"{args.json_ouput_dir}/{args.llm_model}_{args.compression_factor}_{args.temporal_weight}_{args.max_rag_nodes}"

    compare_json = None

    compaere_model = "PlotTree"
    print("Result Output Directory: ", hyper_insert_save_dir)
    if not os.path.exists(hyper_insert_save_dir):
        os.mkdir(hyper_insert_save_dir)
    
    save_json_path = f"{hyper_insert_save_dir}/{vid_dir}-{model_name}.json"
    
    # Load the RAG Embedding model (loaded globally only once).
    tqdm.write("Loading RAG Embedding Model...")
    rag_embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    # Move the model to a suitable device (GPU/CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rag_embedding_model.to(device)

    finished_questions = []
    if os.path.exists(save_json_path):
        finished_questions = read_json(save_json_path)
    
    questions = read_json(json_path)
    score = 0
    
    for index in tqdm(range(len(questions[:]))):
        q_dict = questions[index]
        if index<len(finished_questions) and f'{model_name}_answer' in finished_questions[index]:
            questions[index] = finished_questions[index]
            if finished_questions[index][f'{model_name}_answer'] == q_dict['option'] or finished_questions[index][f'{model_name}_answer'][0] == q_dict['option']:
                score += 1
            continue
        print(q_dict['id'])
        question = q_dict['question']
        choices = q_dict['choices']
        episode_id = q_dict['vid']

        # --- Core: Using PlotTree RAG for question answering ---
        response, node_list = plot_tree_qa(
            args,
            question_text=q_dict['question'],
            choices=choices,
            episode_id=episode_id,
            rag_embedding_model=rag_embedding_model,
            llm_model=args.llm_model, # LLM model
            tree_output_base_dir = tree_output_base_dir,
            max_rag_nodes=args.max_rag_nodes
        )
        print("Response out:", response)

        questions[index][f'{model_name}_answer'] = response
        # questions[index][f'node_list'] = node_list
        if response == q_dict['option'] or response[0] == q_dict['option']:
            score += 1
        print(index, response, "GT:", q_dict['option'] )
        if index % 5 == 0 or index == len(questions)-1:
            print("Score / Size", f"{score}/{index+1}", score/(index+1)*100) 
            save_json(save_json_path, questions[:index+1])

    print("Score / Size", f"{score}/{len(questions)}", score/len(questions)*100)
