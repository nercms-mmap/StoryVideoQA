# PlotTree_builder.py

import os
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import time
# --- (占位符) 导入辅助函数 ---
from kmeans_pytorch import kmeans # 需要安装或提供该库
from utils.localllm import QwenChatbot, get_llm, chat
# from your_utils_file import parse_args, load_json, save_json, makedir # 假设这些在util文件中



#os.environ['CUDA_VISIBLE_DEVICES']= '0'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = "/mnt/disk6new/wzq/ckpt"




def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        json_str = file.read()
    data = json.loads(json_str)
    return data
    
def plot_similarity(points, centroid):
    """
    Calculate plot similarity distance between points and centroid.
    """
    points = points.to(torch.float32)
    centroid = centroid.to(torch.float32)
    points, points_ids = points[:, :-1], points[:, -1] 
    centroid, centroid_id = centroid[:-1], centroid[-1]

    points_normalized = F.normalize(points, dim=1)
    centroid_normalized = F.normalize(centroid.unsqueeze(0), dim=1)
    distance_ids = torch.abs(points_ids-centroid_id)

    return 1 - torch.mm(points_normalized, centroid_normalized.T).squeeze() + distance_ids

# --- (占位符) 定义 PlotTree 核心算法函数 ---
def get_embedding_from_text(text: str, model: SentenceTransformer) -> torch.Tensor:
    """
    [占位符]：将文本转换为RAG Embedding。
    输入:
        text (str): 单个文本字符串。
        model (SentenceTransformer): 预加载的RAG Embedding模型。
    输出:
        torch.Tensor: 文本对应的Embedding向量 (1D Tensor)。
    """
    return model.encode([text], convert_to_tensor=True, device=model.device, prompt_name="query").squeeze(0)


chatbot = None
llm = None




def get_llm_summary(args, plot_descriptions_list: list[str], llm_model) -> str:
    """
    [占位符]：调用本地LLM总结情节描述列表。
    输入:
        plot_descriptions_list (list[str]): 需要总结的子情节描述字符串列表。
        llm_model: 本地LLM模型。
    输出:
        str: LLM生成的连贯情节摘要。
    """
    # pass # 待实现
    # 拼接所有子情节描述，作为LLM的输入
    combined_input_text = " ".join(plot_descriptions_list)

    # 示例Prompt (你需要根据你的LLM和任务进行优化)
    prompt = (
        f"You are a master storyteller tasked with summarizing video content. "
        f"Given a sequence of plot descriptions from a video, your goal is to create a concise and coherent plot summary (or synopsis) "
        f"that captures the main narrative flow. "
        f"Focus on the the progression of events. "
        f"Organize the summary chronologically, highlighting major developments in the plot. "
        f"Keep the summary brief and in plain text, suitable for a plot overview. \n\n"
        f"Plot Descriptions: {combined_input_text}\n\n"
        f"Summary: -nothinking"
    )
    
    global chatbot, llm
    if llm_model == "Qwen/Qwen3-30B-A3B":
        if not chatbot:
            chatbot = QwenChatbot(model_name="Qwen/Qwen3-30B-A3B", cache_dir = "/mnt/disk6new/wzq/LLM/ckpt")
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
     # 简单截断以防过长
    # if len(summary) > 500: summary = summary[:500] + "..."
    return summary



def _decide_k_value(num_data_points: int, compression_factor: int = 4) -> int:
    """
    根据给定的压缩因子超参数，决定当前层级的K值（聚类数量）。
    此版本简化K值选择, 仅基于压缩因子，不考虑min_cluster_size，
    min_cluster_size的检查将在聚类后处理阶段进行。
    Args:
        current_nodes_embeddings (torch.Tensor): 当前层所有节点的Embedding (N, D)。
        num_data_points (int): 当前层有多少个节点。
        compression_factor (int): 压缩因子超参数，例如 4 (表示将节点数量压缩为原来的 1/4)。
    Returns:
        int: 计算出的最佳K值。
    """
    # 1. 处理边界情况
    if num_data_points <= 1:
        return 1 # 只有1个或0个节点，K只能是1

    # 2. 计算基于压缩因子的目标 K 值
    # 确保 compression_factor 至少为 2，避免除以 0
    if compression_factor < 2:
        compression_factor = 2 
    
    k_final = num_data_points // compression_factor
    
    # 3. 确保 K 值至少为 1 (聚类至少要产生一个簇)
    if k_final < 1:
        k_final = 1
    
    # 4. 确保 K 值不超过当前层节点数 (你不能把 N 个点分成 N+1 个簇)
    if k_final > num_data_points:
        k_final = num_data_points 
    
    return k_final

# PlotTree_builder.py

# ... (其他导入和函数定义保持不变) ...

def _post_process_clusters(
    cluster_ids_x: torch.Tensor,                 # K-Means 聚类结果 (每个点对应的簇ID)
    cluster_centers: torch.Tensor,               # K-Means 聚类中心
    current_nodes_embeddings: torch.Tensor,       # 当前层的所有节点embedding (原始数据点)
    min_nodes_in_cluster: int,                   # 新簇中必须包含的最小子节点数量
    rag_embedding_model_device: torch.device     # 用于张量操作的设备
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    聚类结果后处理，识别并合并“孤儿簇”（包含子节点数量过少的簇）。

    Args:
        cluster_ids_x (torch.Tensor): K-Means 聚类结果 (每个点对应的簇ID)。
        cluster_centers (torch.Tensor): K-Means 聚类中心。
        current_level_nodes (list[dict]): 当前层的所有节点对象 (每个节点是 K-Means 的一个数据点)。
        min_nodes_in_cluster (int): 一个簇被认为是“孤儿簇”的阈值，即其包含的子节点数量小于此值。
        rag_embedding_model_device (torch.device): 用于张量操作的设备。

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - refined_cluster_ids_x (torch.Tensor): 经过处理后的每个点对应的簇ID。
            - refined_cluster_centers (torch.Tensor): 经过处理后的簇中心。
    """
    tqdm.write("  -> Post-processing clusters to handle 'orphan' clusters based on node count...")

    original_num_clusters = len(cluster_centers)
    node_to_cluster_map = cluster_ids_x.tolist() 

    orphan_cluster_ids = set()
    cluster_to_nodes_map = {i: [] for i in range(original_num_clusters)}
    
    # 1. 识别孤儿簇 (基于子节点数量)
    for node_idx, cluster_id in enumerate(node_to_cluster_map):
        cluster_to_nodes_map[cluster_id].append(node_idx)

    for cluster_id, node_indices_in_cluster in cluster_to_nodes_map.items():
        # 如果一个簇没有包含任何节点，或者包含的子节点数量低于阈值
        if len(node_indices_in_cluster) < min_nodes_in_cluster:
            orphan_cluster_ids.add(cluster_id)
    
    tqdm.write(f"    Identified {len(orphan_cluster_ids)} orphan clusters out of {original_num_clusters}.")

    if not orphan_cluster_ids:
        return cluster_ids_x, cluster_centers

    # 2. 合并孤儿簇
    refined_cluster_ids_x = cluster_ids_x.clone() 
    
    non_orphan_cluster_centers = [
        cluster_centers[i] for i in range(original_num_clusters) if i not in orphan_cluster_ids
    ]
    
    if not non_orphan_cluster_centers: # 极端情况：所有簇都是孤儿簇，全部合并到一个
        tqdm.write("    All clusters are orphans. Merging all into a single cluster (cluster ID 0).")
        refined_cluster_ids_x[:] = 0 
        refined_cluster_centers = current_nodes_embeddings.mean(dim=0).unsqueeze(0) # 重新计算所有点的中心
        return refined_cluster_ids_x, refined_cluster_centers

    non_orphan_cluster_centers_tensor = torch.stack(non_orphan_cluster_centers).to(rag_embedding_model_device)

    # 将孤儿簇中的每个节点重新分配到最近的非孤儿簇
    for orphan_cluster_id in orphan_cluster_ids:
        nodes_in_orphan_cluster = cluster_to_nodes_map[orphan_cluster_id]
        if not nodes_in_orphan_cluster: 
            continue

        for node_idx in nodes_in_orphan_cluster:
            node_embedding = current_nodes_embeddings[node_idx]
            
            distances_to_non_orphans = plot_similarity(non_orphan_cluster_centers_tensor, node_embedding)
            
            closest_non_orphan_local_idx = torch.argmin(distances_to_non_orphans).item()
            
            # 获取最近的非孤儿簇的原始ID
            closest_non_orphan_global_id = [
                i for i in range(original_num_clusters) if i not in orphan_cluster_ids
            ][closest_non_orphan_local_idx]
            
            refined_cluster_ids_x[node_idx] = closest_non_orphan_global_id
    
    # 3. 重新计算新的簇中心和簇ID映射 (因为一些簇可能消失，簇ID需要重新映射)
    # 这部分逻辑与之前保持一致，用于规范化簇ID和中心
    new_cluster_id_map = {} 
    next_new_cluster_id = 0
    
    # 收集处理后的实际簇ID
    actual_new_cluster_ids = sorted(list(set(refined_cluster_ids_x.tolist())))
    for old_id in actual_new_cluster_ids:
        new_cluster_id_map[old_id] = next_new_cluster_id
        next_new_cluster_id += 1
    
    # 应用新的ID映射
    for node_idx in range(len(refined_cluster_ids_x)):
        old_id = refined_cluster_ids_x[node_idx].item()
        refined_cluster_ids_x[node_idx] = new_cluster_id_map[old_id]
    
    num_refined_clusters = next_new_cluster_id # 最终的簇数量
    refined_cluster_centers = torch.zeros(num_refined_clusters, cluster_centers.shape[1], device=rag_embedding_model_device)
    counts = torch.zeros(num_refined_clusters, device=rag_embedding_model_device)
    
    for node_idx in range(len(current_nodes_embeddings)):
        new_id = refined_cluster_ids_x[node_idx].item()
        refined_cluster_centers[new_id] += current_nodes_embeddings[node_idx]
        counts[new_id] += 1
        
    # 防止除以零，对于可能依然存在的空簇（理论上不会，但安全起见）
    counts[counts == 0] = 1 # 避免除以零，对空簇不影响均值
    refined_cluster_centers = refined_cluster_centers / counts.unsqueeze(1)
    
    tqdm.write(f"    Clusters refined from {original_num_clusters} to {num_refined_clusters}.")

    return refined_cluster_ids_x, refined_cluster_centers


def temporal_decay(l, alpha=1.0, epsilon=1e-2):
    """
    Calculate the temporal decay coefficient for hierarchical clustering.
    
    Parameters:
    l (int): Hierarchy level (deeper level has larger l value)
    alpha (float): Scaling factor controlling the decay rate (default: 1.0)
    epsilon (float): Small constant to prevent division by zero (default: 1e-2)
    
    Returns:
    float: Decay coefficient lambda(l)
    """
    return 1.0 / (alpha * l + epsilon)


# --- 主 PlotTree 构建函数 ---
def build_plottree_for_episode(args, captions_json_path: Path, episode_id: str, 
                           rag_embedding_model: SentenceTransformer, llm_model,
                           output_dir: Path,
                           compression_factor: int = 4,
                           min_nodes_in_cluster: int = 2,
                           temporal_weight_rate: float = 100) -> Path:
    """
    主函数：自底向上构建视频的叙事摘要树 (PlotTree) 的核心逻辑。
    
    输入:
        captions_json_path (Path): 包含视频所有帧captions的JSON文件路径。
                                   结构如: {"episode_id": [{"caption":..., "subtitle":..., "charcters_info":...}, ...]}
        episode_id (str): 处理的episode id
        rag_embedding_model (SentenceTransformer): 用于生成情节嵌入的模型。
        llm_model: 用于生成情节摘要的本地LLM模型名字。
        output_dir (Path): 存储结果JSON的目录。
        min_cluster_size (int): 聚类时，每个簇的最小帧数，也是递归停止的条件之一。
        max_k_per_level (int): 每层聚类的最大K值。
    
    输出:
        Path: 保存PlotTree节点列表的JSON文件路径。
    """
    # 0. 参数校验与环境准备
    if not captions_json_path.exists():
        raise FileNotFoundError(f"Captions JSON not found at {captions_json_path}")
    output_dir.mkdir(parents=True, exist_ok=True) # 确保输出目录存在

    # 1. 读取原始数据
    with open(captions_json_path, 'r', encoding='utf-8') as f:
        full_captions_data = json.load(f)
    
    # 假设JSON中只有一个剧集的数据，或者你需要遍历所有剧集
    # 这里我们只处理第一个剧集 (例如 "Friends-S01E01")
    frames_data = full_captions_data[episode_id]

    all_nodes_in_tree = [] # 存储所有节点，最终保存到JSON
    node_id_counter = 0    # 用于生成唯一节点ID
    
    # 2. 初始化：创建叶子节点层 (Level 0)
    # current_level_nodes: 列表，存储当前层级的节点对象
    # frame_narrative_embeddings_map: 字典，用于快速查找原始帧ID对应的情节嵌入
    tqdm.write(f"--- Processing Episode: {episode_id} ---")
    tqdm.write("2.1 Creating Leaf Nodes (Level 0)...")
    
    current_level_nodes = []
    frame_narrative_embeddings_map = {} 
    total_frames = len(frames_data)

    output_filepath = output_dir / f"{episode_id}.json"



    all_nodes_in_tree = []
    
    for i, frame_data in tqdm(enumerate(frames_data), total=len(frames_data), desc="Encoding Leaf Nodes"):
        # 将原始帧数据转换为统一的文本描述
        # 你的格式: {"caption":..., "subtitle":..., "face_bbox":..., "charcters_info":...}
        # 确保 charcters_info 处理为字符串
        char_str = ", ".join(frame_data['charcters_info']) if isinstance(frame_data.get('charcters_info'), list) else frame_data.get('charcters_info', '')
        combined_text = (
                f"Visual description: {frame_data.get('caption', '')} "
        )
        sub = frame_data.get('subtitle', '')
        char = char_str
        if sub:
            combined_text += f"Spoken dialogue: {frame_data.get('subtitle', '')} "
        if char:
            combined_text += f"Characters: {char_str}."
        # 获取情节嵌入
        narrative_embedding = get_embedding_from_text(combined_text, rag_embedding_model)
        # 归一化帧索引
        temporal_weight = temporal_decay(l=0, alpha=temporal_weight_rate)
        normalized_time = torch.tensor([float(i) / total_frames], device=rag_embedding_model.device, dtype=torch.float32)

        
        narrative_embedding = torch.cat((narrative_embedding, temporal_weight * normalized_time), dim=0)
        # rag_embedding_model.to(device=torch.device('cpu'))
        # narrative_embedding.to(device=torch.device('cpu'))
        # 存储原始帧ID到嵌入的映射
        frame_narrative_embeddings_map[i] = narrative_embedding
        


        # 创建叶子节点对象
        original_plot_description = combined_text
        node_id = f"leaf_{i}"
        leaf_node = {
                "node_id": node_id,
                "parent_id": None, 
                "level": 0,        
                "start_frame_idx": i,
                "end_frame_idx": i,
                "original_frame_indices": [i],
                "plot_summary": original_plot_description, # 原始描述作为摘要
                "plot_embedding": narrative_embedding.tolist(), # 转换为list便于JSON序列化
                "children_node_ids": [] 
        }
        current_level_nodes.append(leaf_node)
        all_nodes_in_tree.append(leaf_node) # 添加到所有节点列表
            
             
    tqdm.write(f"Leaf nodes created. Total: {len(current_level_nodes)}")

    # 3. 迭代聚合循环 (自底向上构建树)
    current_level = 0
    while len(current_level_nodes) > 1: # 只要还有多于一个节点，就继续聚合
        current_level += 1
        tqdm.write(f"\n3.1 Aggregating Level {current_level-1} to Level {current_level}...")
        
        next_level_nodes = [] # 存储新创建的父节点

        # 准备当前层所有节点的嵌入，用于 K-Means 聚类
        current_nodes_embeddings = torch.stack([torch.tensor(node["plot_embedding"], device=rag_embedding_model.device) for node in current_level_nodes])
        
        # 智能决定当前层级的 K 值
        # K-Means 聚类 (需要确保导入 kmeans 函数)
        k = _decide_k_value(
            num_data_points=current_nodes_embeddings.shape[0], # 直接传递节点数量
            compression_factor=compression_factor
        )
        tqdm.write(f"Decided K for Level {current_level}: {k}")

        if k <= 1: # 下一步是单根节点，直接聚合
            tqdm.write("Reached single root node. Stopping aggregation.")
            break
        elif k <= 0: # 如果K值不合理，也停止
             tqdm.write("Decided K is too small or zero. Stopping aggregation.")
             break
        current_nodes_embeddings.to(torch.device("cpu"))
        rag_embedding_model.to(torch.device("cpu"))
        # 执行 K-Means 聚类 (对当前层节点的嵌入进行聚类)
        # from kmeans_pytorch import kmeans # 再次提醒确保导入
        print("Using plot Distance")
        cluster_ids_x, cluster_centers = kmeans(
            X=current_nodes_embeddings, 
            num_clusters=k, 
            distance='plot', #'cosine' if current_level != 1 else 'plot', 
            device=rag_embedding_model.device # 确保在正确的设备上运行
        )
        
        current_nodes_embeddings.to(torch.device(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        rag_embedding_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 传入 min_nodes_in_cluster 来处理孤儿簇
        cluster_ids_x, cluster_centers = _post_process_clusters(
            cluster_ids_x, 
            cluster_centers, 
            current_nodes_embeddings, # 传入当前层节点的embeddings (作为KMeans的原始数据点)
            min_nodes_in_cluster, # 传入最小子节点数量阈值
            rag_embedding_model.device
        )
        # 重新确定处理后的实际K值
        k = len(cluster_centers)
        # 如果 post-process 后 K 变为 0 (所有簇都合并到一个，但可能没有被正确创建)，需要特殊处理
        if k == 0:
            tqdm.write("    Post-processing resulted in 0 clusters. Stopping aggregation.")
            break # 退出循环，最终会创建虚拟根节点
        tqdm.write(f"  K after post-processing: {k}.")

        
        # 遍历每个聚类簇，创建新的父节点
        for cluster_id in tqdm(range(k), desc=f"3.2 Creating Level {current_level} Nodes"):
            indices_in_current_level_cluster = torch.where(cluster_ids_x == cluster_id)[0]
            if indices_in_current_level_cluster.numel() == 0:
                continue # 跳过空簇

            # 获取子节点 (来自当前层)
            children_nodes_current_level = [current_level_nodes[idx.item()] for idx in indices_in_current_level_cluster]
            
            # 聚合所有原始帧索引 (用于新父节点)
            aggregated_original_frame_indices = sorted(list(
                set(idx for node in children_nodes_current_level for idx in node["original_frame_indices"])
            ))
            # represent_id = sum(aggregated_original_frame_indices)/len(aggregated_original_frame_indices)
            represent_id = min(aggregated_original_frame_indices)
            normalized_time = torch.tensor([ represent_id/ (total_frames - 1)], device=rag_embedding_model.device, dtype=torch.float32)
            temporal_weight = temporal_decay(l=current_level, alpha=temporal_weight_rate)
            # 本地 LLM 摘要：聚合子情节，生成父情节摘要
            plot_descriptions_to_summarize = [node["plot_summary"] for node in children_nodes_current_level]
            summary_text = get_llm_summary(args, plot_descriptions_to_summarize, llm_model)

            parent_node_plot_embedding = get_embedding_from_text(summary_text, rag_embedding_model) 
            parent_node_plot_embedding = torch.cat((parent_node_plot_embedding, temporal_weight * normalized_time), dim=0)
            # 创建新的父节点对象
            parent_node_id = f"level_{current_level}_cluster_{cluster_id}"
            parent_node = {
                "node_id": parent_node_id,
                "parent_id": None, # 暂时为None，在子节点中更新
                "level": current_level,
                "start_frame_idx": aggregated_original_frame_indices[0],
                "end_frame_idx": aggregated_original_frame_indices[-1],
                "original_frame_indices": aggregated_original_frame_indices,
                "plot_summary": summary_text,
                "plot_embedding": parent_node_plot_embedding.tolist(), # 转换为list便于JSON序列化
                "children_node_ids": [node["node_id"] for node in children_nodes_current_level]
            }
            next_level_nodes.append(parent_node)
            all_nodes_in_tree.append(parent_node) # 添加到总节点列表
            print([node["node_id"] for node in children_nodes_current_level])
            # 更新子节点的 parent_id (确保树的父子关系建立)
            for child_node in children_nodes_current_level:
                # 找到 all_nodes_in_tree 中的子节点并更新其 parent_id
                for node_in_all in all_nodes_in_tree:
                    if node_in_all["node_id"] == child_node["node_id"]:
                        node_in_all["parent_id"] = parent_node_id
                        break
        
        current_level_nodes = next_level_nodes # 进入下一轮循环
        tqdm.write(f"Level {current_level} created. Total nodes at this level: {len(current_level_nodes)}")

    # 4. 根节点处理 (循环结束，确保有一个明确的根节点)
    if len(current_level_nodes) == 1:
        root_node = current_level_nodes[0]
        root_node["parent_id"] = None # 确保最终的根节点父ID为None
        tqdm.write(f"PlotTree construction complete. Root node ID: {root_node['node_id']}")
    else:
        # 如果循环提前终止，且有多个顶层节点，则创建虚拟根节点
        tqdm.write("PlotTree construction ended with multiple top-level nodes, creating a dummy root.")
        final_root_id = "root_summary"
        final_root_original_frames = sorted(list(set(idx for node in current_level_nodes for idx in node["original_frame_indices"])))
        final_root_summary = get_llm_summary(args, [node["plot_summary"] for node in current_level_nodes], llm_model)
        final_root_embedding = get_embedding_from_text(final_root_summary, rag_embedding_model) 
        
        final_root_node = {
            "node_id": final_root_id,
            "parent_id": None,
            "level": current_level, # 使用最后一层级别作为根节点级别
            "start_frame_idx": final_root_original_frames[0],
            "end_frame_idx": final_root_original_frames[-1],
            "original_frame_indices": final_root_original_frames,
            "plot_summary": final_root_summary,
            "plot_embedding": final_root_embedding.tolist(),
            "children_node_ids": [node["node_id"] for node in current_level_nodes]
        }
        all_nodes_in_tree.append(final_root_node)

        # 更新这些顶层节点的父ID指向新创建的虚拟根节点
        for node_in_all in all_nodes_in_tree:
            if node_in_all["node_id"] in [n["node_id"] for n in current_level_nodes]:
                node_in_all["parent_id"] = final_root_id


    # 5. 保存结果
    output_filepath = output_dir / f"{episode_id}.json"
    for index, node in enumerate(all_nodes_in_tree):
        del all_nodes_in_tree[index]["plot_embedding"]
        
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(all_nodes_in_tree, f, indent=4)
    
    tqdm.write(f"PlotTree nodes saved to: {output_filepath}")
    return output_filepath, len(frames_data)


def main():
    arg_parser = argparse.ArgumentParser()
    # primary setting
    arg_parser.add_argument("--captions_path", type=str, default="data/captions/Friends.json")
    arg_parser.add_argument('--output_base_dir', type=str, default="results/PlotTree", help="Base directory where PlotTree JSONs are stored.")  
    arg_parser.add_argument('--llm_model', type=str, default="Gemini-2.0-flash")
    arg_parser.add_argument("--openai_model", default="gemini-2.0-flash",type=str, help="model name of GPT")
    arg_parser.add_argument("--openai_key", required=True, type=str, help="key for llm")
    arg_parser.add_argument("--openai_proxy", required=True, type=str, help="proxy for llm")
    arg_parser.add_argument('--compression_factor', type=int, default=12)
    arg_parser.add_argument('--temporal_weight', type=float, default=100)
    arg_parser.add_argument('--min_nodes_in_cluster', type=int, default=2)
    args = arg_parser.parse_args()

    # Make sure the directory exists !!!
    args.captions_path = Path(args.captions_path)
    args.output_base_dir = Path(f"results/PlotTree/{args.llm_model}_{args.compression_factor}_{args.temporal_weight}")
    print(args.output_base_dir )
    args.output_base_dir.mkdir(parents=True, exist_ok=True)

    tqdm.write("Loading RAG Embedding Model...")
    rag_embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rag_embedding_model.to(device)

    
    tqdm.write("RAG Embedding Model Loaded.")
    print("Loading", args.captions_path , "...")
    # 1. loading raw caption data
    with open(args.captions_path, 'r', encoding='utf-8') as f:
        full_captions_data = json.load(f)
    keys = full_captions_data.keys()
    
    for episode_id in list(keys)[:]:
        output_filepath = args.output_base_dir / f"{episode_id}.json"
        
        if output_filepath.exists():
            print(output_filepath, 'Existing!!!')
            continue
        tqdm.write(f"Building PlotTree for the episode... for {episode_id}")
        plottree_json_path, caption_size = build_plottree_for_episode(
            args = args,
            captions_json_path=args.captions_path,
            episode_id = episode_id,
            rag_embedding_model=rag_embedding_model,
            llm_model=args.llm_model, 
            output_dir=args.output_base_dir,
            compression_factor = args.compression_factor,
            min_nodes_in_cluster=args.min_nodes_in_cluster,
            temporal_weight_rate=args.temporal_weight
        )
        tqdm.write(f"PlotTree construction finished. Tree saved to: {plottree_json_path}")
        

if __name__ == '__main__':
    main()