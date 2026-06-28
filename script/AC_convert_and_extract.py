"""
AC_convert_and_extract.py
==========================
Convert Abaqus INP to VTU, extract element sets, and generate
measurement-point CSV. Must run before AD stage.

Usage:
    cd script && python AC_convert_and_extract.py
"""

import json
import os
import pickle
import random
import re
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

try:
    import meshio
except ImportError:
    print("错误: 未找到 meshio 库。请安装：")
    print("  pip install meshio")
    sys.exit(1)

# ========================================
# 参数配置区（按自然逻辑顺序编写）
# ========================================

# --- 1. 路径配置 ---
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'script', 'AC_convert_and_extract_output')

# --- 2. 输入文件 ---
INP_FILE_PATH = r"C:\SHM_abaqus_models\second_damage_12.inp"
HEALTH_INP_PATH = r"C:\SHM_abaqus_models\health.inp"       # 用于偏移测点生成

# --- 3. 输出文件 ---
OUTPUT_VTU_NAME = "whole_from_inp.vtu"                      # VTU网格文件
OUTPUT_MEASURES_CSV = "measures_ID_original.csv"            # 测点ID列表（重要：后续流程依赖）
OUTPUT_ELSETS_SUMMARY = "element_sets_summary.txt"          # 元素集汇总报告
OUTPUT_ID_MAPPING_CSV = "abaqus_id_to_vtu_index.csv"        # Abaqus元素ID到VTU索引映射表

# --- 4. 元素集配置 ---
TARGET_ELSET_NAME = 'all_measures'                          # 要提取的元素集名称
VERBOSE = True                                              # 是否显示详细信息

# --- 5. 偏移测点生成配置 ---
# 【配置来源说明】
#   - 通过 AAA_oneclick_run.py 调用时：从 TL_settings.jsonc 读取
#     (offset + cases 中的 use_offset)
#   - 独立运行本脚本时：使用下方默认配置
#
# ========================================

# 偏移测点生成开关（独立运行时的默认配置）
GENERATE_OFFSET = None  # None=由配置决定, True=强制生成, False=不生成

OFFSET_REGION_ELSET = 'potential_damage_region'             # 偏移搜索区域（来自health.inp）
OFFSET_RANDOM_SEED = 42                                     # 随机种子（用于可复现性）

# 要偏移的测点行索引（独立运行时的默认配置）
# 通过 run_from_config 调用时会被 TL_settings.jsonc 的 offset 配置覆盖
OFFSET_INDICES = None  # None=由配置决定

OUTPUT_ADJACENCY_GRAPH_PKL = "adjacency_graph.pkl"          # 邻接图pickle文件
OUTPUT_ADJACENCY_INFO_TXT = "adjacency_graph_info.txt"      # 邻接图统计信息
# 偏移测点文件名会动态生成，如 measures_ID_offset_2.csv 或 measures_ID_offset_2to4.csv
OUTPUT_OFFSET_MAPPING_TXT = "offset_mapping.txt"            # 偏移映射报告


def generate_nested_offset_sets(
    n_sensors: int,
    count_list: List[int],
    seed: int = 42
) -> Dict[int, List[int]]:
    """
    生成嵌套的偏移传感器索引集合。

    用 seed 初始化随机数生成器，从 n_sensors 个传感器中打乱顺序，
    然后按前缀取子集，保证嵌套关系：
    count=1的索引 ⊂ count=2的索引 ⊂ ... ⊂ count=max的索引

    Args:
        n_sensors: 总传感器数 (例如 252)
        count_list: 偏移个数列表，如 [0, 1, 2, 3, 4, 5]
        seed: 随机种子

    Returns:
        dict: {count: [row_indices]}
        例如：{0: [], 1: [47], 2: [47, 183], 3: [47, 183, 5], ...}
    """
    rng = random.Random(seed)
    all_indices = list(range(n_sensors))
    rng.shuffle(all_indices)

    max_count = max(count_list) if count_list else 0
    if max_count > n_sensors:
        raise ValueError(
            f"offset_count 最大值 {max_count} 超过传感器总数 {n_sensors}"
        )

    result = {}
    for count in count_list:
        result[count] = all_indices[:count]  # count=0 时为 []

    # 打印嵌套验证信息
    sorted_counts = sorted(count_list)
    print(f"\n[嵌套偏移] 生成 {len(count_list)} 档嵌套偏移索引集 (seed={seed})")
    for count in sorted_counts:
        indices = result[count]
        preview = str(indices) if len(indices) <= 10 else f"{indices[:10]}... (共{len(indices)}个)"
        print(f"  - count={count}: {preview}")

    # 验证嵌套关系
    for i in range(len(sorted_counts) - 1):
        c_small = sorted_counts[i]
        c_large = sorted_counts[i + 1]
        if not set(result[c_small]).issubset(set(result[c_large])):
            print(f"  [警告] 嵌套关系不满足: count={c_small} ⊄ count={c_large}")

    return result


def get_offset_count_measures_filename(count: int) -> str:
    """
    根据偏移传感器个数生成测点CSV文件名

    Args:
        count: 偏移传感器个数

    Returns:
        文件名，如 "measures_ID_offset_count_3.csv"
        count=0 时返回 "measures_ID_offset_count_0.csv"
    """
    return f"measures_ID_offset_count_{count}.csv"


def is_nested_offset(offset_value) -> bool:
    """
    检测offset配置是否为嵌套数组（批量offset方案）

    单个方案: [0] 或 [1,2,3] - 元素全是int
    批量方案: [[0],[1],[2]] 或 [[1,2],[3,4]] - 元素全是list

    Args:
        offset_value: offset配置值

    Returns:
        True 表示嵌套数组（批量方案），False 表示单个方案或无效
    """
    if not isinstance(offset_value, list) or len(offset_value) == 0:
        return False
    return isinstance(offset_value[0], list)


def get_offset_name(offset_indices: list) -> str:
    """
    根据偏移测点索引列表自动生成偏移方案名称

    命名规则:
    - [2] -> "offset_2"
    - [2, 3, 4] -> "offset_2to4"
    - [1, 3, 5] -> "offset_1_3_5" (非连续)
    - None 或 [] -> "none"

    Args:
        offset_indices: 偏移测点行索引列表，如 [2] 或 [2, 3, 4]

    Returns:
        偏移方案名称字符串
    """
    if offset_indices is None or len(offset_indices) == 0:
        return "none"

    sorted_indices = sorted(offset_indices)

    if len(sorted_indices) == 1:
        return f"offset_{sorted_indices[0]}"

    # 检查是否连续
    is_continuous = all(
        sorted_indices[i+1] - sorted_indices[i] == 1
        for i in range(len(sorted_indices) - 1)
    )

    if is_continuous:
        return f"offset_{sorted_indices[0]}to{sorted_indices[-1]}"
    else:
        # 非连续情况，用下划线连接
        return "offset_" + "_".join(str(i) for i in sorted_indices)


def get_offset_measures_filename(offset_indices: list) -> str:
    """
    生成偏移测点CSV文件名

    Args:
        offset_indices: 偏移测点行索引列表

    Returns:
        文件名，如 "measures_ID_offset_2.csv" 或 "measures_ID_offset_2to4.csv"
    """
    offset_name = get_offset_name(offset_indices)
    if offset_name == "none":
        return "measures_ID_offset.csv"  # fallback
    return f"measures_ID_{offset_name}.csv"




# ========================================
# 元素集解析函数
# ========================================

def parse_elsets_from_inp(inp_file_path):
    """
    从 INP 文件中提取所有元素集定义
    
    Args:
        inp_file_path: INP 文件路径
    
    Returns:
        dict: {elset_name: [element_ids...]}
    """
    print(f"\n[元素集解析] 从INP文件提取元素集...")
    
    element_sets = {}
    current_elset_name = None
    current_elset_data = []
    is_generate = False
    
    with open(inp_file_path, 'r', encoding='latin1') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # 跳过空行和注释
            if not line or line.startswith('**'):
                continue
            
            # 检测 *Elset 关键字
            if line.upper().startswith('*ELSET'):
                # 保存之前的元素集
                if current_elset_name and current_elset_data:
                    element_sets[current_elset_name] = current_elset_data
                    if VERBOSE:
                        print(f"  [{current_elset_name}]: {len(current_elset_data)} 个元素")
                
                # 重置
                current_elset_data = []
                is_generate = False
                
                # 解析元素集名称
                # 格式: *Elset, elset=all_measures, instance=Part-1-1
                match = re.search(r'elset\s*=\s*([^,\s]+)', line, re.IGNORECASE)
                if match:
                    current_elset_name = match.group(1).strip()
                    
                    # 检查是否为 generate 类型
                    if 'generate' in line.lower():
                        is_generate = True
                else:
                    current_elset_name = None
                
                continue
            
            # 遇到其他关键字则结束当前元素集
            if line.startswith('*') and not line.startswith('**'):
                if current_elset_name and current_elset_data:
                    element_sets[current_elset_name] = current_elset_data
                    if VERBOSE:
                        print(f"  [{current_elset_name}]: {len(current_elset_data)} 个元素")
                
                current_elset_name = None
                current_elset_data = []
                is_generate = False
                continue
            
            # 解析元素集数据
            if current_elset_name:
                if is_generate:
                    # Generate 格式: start, end, increment
                    # 例如: 128754, 128757, 1
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 2:
                        try:
                            start = int(parts[0])
                            end = int(parts[1])
                            increment = int(parts[2]) if len(parts) >= 3 else 1
                            
                            generated_ids = list(range(start, end + 1, increment))
                            current_elset_data.extend(generated_ids)
                        except ValueError:
                            pass
                else:
                    # 逗号分隔的单元ID列表
                    try:
                        ids = [int(x.strip()) for x in line.split(',') if x.strip()]
                        current_elset_data.extend(ids)
                    except ValueError:
                        pass
        
        # 保存最后一个元素集
        if current_elset_name and current_elset_data:
            element_sets[current_elset_name] = current_elset_data
            if VERBOSE:
                print(f"  [{current_elset_name}]: {len(current_elset_data)} 个元素")
    
    print(f"\n[元素集解析] 共找到 {len(element_sets)} 个元素集")
    
    return element_sets


def save_elsets_summary(element_sets, output_path):
    """
    保存元素集汇总报告
    
    Args:
        element_sets: 元素集字典
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("元素集汇总报告\n")
        f.write("=" * 70 + "\n\n")
        
        for name, ids in sorted(element_sets.items()):
            f.write(f"[{name}]\n")
            f.write(f"  - 元素数量: {len(ids)}\n")
            f.write(f"  - ID范围: {min(ids)} 到 {max(ids)}\n")
            f.write(f"  - 前10个ID: {ids[:10]}\n")
            f.write("\n")
    
    print(f"[汇总报告] 已保存到: {output_path}")


def generate_measures_csv(element_sets, output_csv_path, elset_name='all_measures'):
    """
    从元素集生成测点CSV文件

    Args:
        element_sets: 元素集字典
        output_csv_path: 输出CSV路径
        elset_name: 元素集名称 (默认 'all_measures')

    Returns:
        list: 测点ID列表
    """
    print(f"\n[测点生成] 从元素集 '{elset_name}' 生成测点CSV...")

    if elset_name not in element_sets:
        print(f"[警告] 未找到元素集 '{elset_name}'！")
        print(f"[信息] 可用的元素集: {list(element_sets.keys())}")
        return None

    measure_ids = sorted(element_sets[elset_name])

    # 创建 DataFrame (与原 measures_ID.csv 格式一致)
    df = pd.DataFrame({
        'all_measures': measure_ids
    })

    # 保存到CSV
    df.to_csv(output_csv_path, index=False)

    print(f"[成功] 测点CSV已保存: {output_csv_path}")
    print(f"  - 测点总数: {len(measure_ids)}")
    print(f"  - ID范围: {min(measure_ids)} 到 {max(measure_ids)}")
    print(f"  - 前10个ID: {measure_ids[:10]}")

    return measure_ids


def save_id_mapping(abaqus_id_to_vtu_index, output_csv_path):
    """
    保存Abaqus元素ID到VTU索引的映射文件

    Args:
        abaqus_id_to_vtu_index: 映射字典 {abaqus_id: vtu_index}
        output_csv_path: 输出CSV路径
    """
    print(f"\n[映射保存] 保存Abaqus ID到VTU索引映射...")

    # 创建DataFrame
    df = pd.DataFrame({
        'abaqus_id': list(abaqus_id_to_vtu_index.keys()),
        'vtu_index': list(abaqus_id_to_vtu_index.values())
    })

    # 按Abaqus ID排序
    df = df.sort_values('abaqus_id').reset_index(drop=True)

    # 保存到CSV
    df.to_csv(output_csv_path, index=False)

    print(f"[成功] 映射文件已保存: {output_csv_path}")
    print(f"  - 映射数量: {len(df)}")
    print(f"  - Abaqus ID范围: {df['abaqus_id'].min()} 到 {df['abaqus_id'].max()}")
    print(f"  - VTU索引范围: {df['vtu_index'].min()} 到 {df['vtu_index'].max()}")
    print(f"  - 前5个映射: {df.head().to_dict('records')}")


# ========================================
# 偏移测点生成函数（用于迁移学习实验）
# ========================================

def parse_element_connectivity(inp_file_path: str, target_element_ids: Set[int]) -> Dict[int, List[int]]:
    """
    从INP文件解析元素-节点连接性

    Args:
        inp_file_path: INP文件路径
        target_element_ids: 目标元素ID集合（只解析这些元素）

    Returns:
        dict: {element_id: [node_id1, node_id2, ...]}
    """
    print(f"\n[连接性解析] 解析元素节点连接性...")
    print(f"  - 目标元素数: {len(target_element_ids)}")

    element_connectivity = {}
    current_section = None
    in_part = False

    with open(inp_file_path, 'r', encoding='latin1') as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith('**'):
                continue

            if line.startswith('*'):
                line_upper = line.upper()

                # 检测Part部分
                if line_upper.startswith('*PART'):
                    in_part = True
                    continue
                elif line_upper.startswith('*END PART'):
                    in_part = False
                    continue
                elif line_upper.startswith('*ASSEMBLY'):
                    break  # Assembly之后不再解析

                if not in_part:
                    continue

                # 检测Element部分
                if line_upper.startswith('*ELEMENT'):
                    current_section = 'ELEMENT'
                else:
                    if current_section == 'ELEMENT':
                        current_section = None

                continue

            # 解析元素连接性
            if current_section == 'ELEMENT':
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        elem_id = int(parts[0].strip())

                        # 只解析目标元素
                        if elem_id in target_element_ids:
                            node_ids = [int(p.strip()) for p in parts[1:]]
                            element_connectivity[elem_id] = node_ids
                    except ValueError:
                        pass

    print(f"  - 成功解析: {len(element_connectivity)} 个元素")
    return element_connectivity


def build_adjacency_graph_edge_based(element_connectivity: Dict[int, List[int]]) -> Dict[int, Set[int]]:
    """
    基于边邻接构建邻接图（两个元素至少共享2个节点才算邻接）

    对于壳单元（S4R/S3），边邻接意味着共享一条边（2个节点）

    Args:
        element_connectivity: 元素连接性字典 {elem_id: [node_ids...]}

    Returns:
        dict: {elem_id: {neighbor_elem_ids...}}
    """
    print(f"\n[邻接图构建] 构建边邻接图...")
    print(f"  - 元素数: {len(element_connectivity)}")

    # 初始化邻接图
    adjacency = {elem_id: set() for elem_id in element_connectivity}

    # Step 1: 构建节点→元素的反向索引
    node_to_elements = defaultdict(set)
    for elem_id, nodes in element_connectivity.items():
        for node in nodes:
            node_to_elements[node].add(elem_id)

    print(f"  - 节点数: {len(node_to_elements)}")

    # Step 2: 对每个元素，找所有至少共享2个节点的邻居
    for elem_id, nodes in element_connectivity.items():
        node_set = set(nodes)

        # 收集所有候选邻居（至少共享1个节点）
        candidate_neighbors = set()
        for node in nodes:
            candidate_neighbors.update(node_to_elements[node])

        # 去掉自己
        candidate_neighbors.discard(elem_id)

        # 检查每个候选者是否满足边邻接条件（至少共享2个节点）
        for neighbor_id in candidate_neighbors:
            neighbor_nodes = set(element_connectivity[neighbor_id])
            shared_nodes = node_set & neighbor_nodes

            if len(shared_nodes) >= 2:  # 边邻接条件
                adjacency[elem_id].add(neighbor_id)

    # 统计信息
    neighbor_counts = [len(neighbors) for neighbors in adjacency.values()]
    avg_neighbors = sum(neighbor_counts) / len(neighbor_counts) if neighbor_counts else 0

    print(f"  - 平均邻居数: {avg_neighbors:.2f}")
    print(f"  - 邻居数范围: [{min(neighbor_counts)}, {max(neighbor_counts)}]")

    return adjacency


def generate_offset_sensors(
    current_sensors: List[int],
    adjacency_graph: Dict[int, Set[int]],
    offset_indices: List[int] = None,
    seed: int = 42,
    predefined_targets: Dict[int, int] = None,
) -> Tuple[List[int], Dict[int, int], List[str]]:
    """
    为当前传感器生成偏移传感器（按指定行索引偏移）

    Args:
        current_sensors: 当前传感器ID列表
        adjacency_graph: 邻接图 {elem_id: {neighbor_ids...}}
        offset_indices: 要偏移的测点行索引列表（例如 [0, 5, 10]），如果为None则默认偏移所有测点
        seed: 随机种子
        predefined_targets: 预定义的偏移目标 {channel_idx: target_element_id}。
            若提供，对应索引使用指定目标而非随机邻居选择。

    Returns:
        tuple: (offset_sensors, mapping, skip_reasons)
            - offset_sensors: 偏移传感器ID列表
            - mapping: 原传感器→偏移传感器映射 {orig_id: offset_id}
            - skip_reasons: 跳过的传感器及原因
    """
    print(f"\n[偏移生成] 生成偏移测点...")
    print(f"  - 原测点数: {len(current_sensors)}")

    # 如果未指定偏移索引，默认偏移所有测点
    if offset_indices is None:
        offset_indices = list(range(len(current_sensors)))

    print(f"  - 指定偏移行索引: {offset_indices if len(offset_indices) <= 10 else f'{offset_indices[:10]}... (共{len(offset_indices)}个)'}")
    if predefined_targets:
        print(f"  - 偏移策略: 预定义目标 (max stress diff, {len(predefined_targets)} 对)")
    else:
        print(f"  - 偏移策略: 按邻接表随机选择")

    random.seed(seed)

    offset_sensors = []
    mapping = {}
    skip_reasons = []

    for idx, sensor_id in enumerate(current_sensors):
        # 检查当前行索引是否在需要偏移的列表中
        if idx in offset_indices:
            if predefined_targets and idx in predefined_targets:
                # 使用预定义目标
                offset_id = int(predefined_targets[idx])
                offset_sensors.append(offset_id)
                mapping[sensor_id] = offset_id
                print(f"  - 已偏移: 行索引[{idx}] 测点 {sensor_id} → {offset_id} (predefined target)")
            elif sensor_id in adjacency_graph and adjacency_graph[sensor_id]:
                # 从邻接图中随机选择一个邻居
                neighbors = list(adjacency_graph[sensor_id])
                offset_id = random.choice(neighbors)
                offset_sensors.append(offset_id)
                mapping[sensor_id] = offset_id
                print(f"  - 已偏移: 行索引[{idx}] 测点 {sensor_id} → {offset_id} (从 {len(neighbors)} 个邻居中随机选择)")
            else:
                # 没有邻居，无法偏移
                offset_id = sensor_id
                offset_sensors.append(offset_id)
                mapping[sensor_id] = offset_id
                skip_reason = f"行索引[{idx}] 测点 {sensor_id} 无邻居，保持不变"
                skip_reasons.append(skip_reason)
                print(f"  - 警告: {skip_reason}")
        else:
            # 其余测点保持不变
            offset_id = sensor_id
            offset_sensors.append(offset_id)
            mapping[sensor_id] = offset_id

    actual_offset_count = sum(1 for idx in range(len(current_sensors))
                              if idx in offset_indices and
                              ((predefined_targets and idx in predefined_targets) or
                               (current_sensors[idx] in adjacency_graph and
                                adjacency_graph[current_sensors[idx]])))

    print(f"  - 成功生成: {len(offset_sensors)} 个偏移测点")
    print(f"  - 实际偏移: {actual_offset_count} 个测点")
    print(f"  - 保持不变: {len(offset_sensors) - actual_offset_count} 个测点")
    print(f"  - 跳过: {len(skip_reasons)} 个测点")

    return offset_sensors, mapping, skip_reasons


def save_adjacency_graph(adjacency: Dict[int, Set[int]], output_pkl: str, output_txt: str):
    """
    保存邻接图及统计信息

    Args:
        adjacency: 邻接图
        output_pkl: pickle文件路径
        output_txt: 统计信息文本路径
    """
    # 保存pickle
    with open(output_pkl, 'wb') as f:
        pickle.dump(adjacency, f)
    print(f"\n[邻接图] 已保存到: {output_pkl}")

    # 统计信息
    neighbor_counts = [len(neighbors) for neighbors in adjacency.values()]
    avg_neighbors = sum(neighbor_counts) / len(neighbor_counts)

    # 邻居数分布
    from collections import Counter
    count_distribution = Counter(neighbor_counts)

    # 保存统计信息
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("元素邻接关系图统计信息（边邻接：至少共享2个节点）\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"元素总数: {len(adjacency)}\n")
        f.write(f"平均邻接元素数: {avg_neighbors:.2f}\n")
        f.write(f"最少邻接元素数: {min(neighbor_counts)}\n")
        f.write(f"最多邻接元素数: {max(neighbor_counts)}\n\n")

        f.write("邻接元素数分布:\n")
        for count in sorted(count_distribution.keys()):
            f.write(f"  {count} 个邻接元素: {count_distribution[count]} 个元素\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("示例（前10个元素的邻接关系）:\n")
        f.write("=" * 70 + "\n\n")

        for i, (elem_id, neighbors) in enumerate(sorted(adjacency.items())[:10], 1):
            f.write(f"元素 {elem_id}:\n")
            f.write(f"  邻接元素数: {len(neighbors)}\n")
            f.write(f"  邻接元素ID: {sorted(neighbors)}\n\n")

    print(f"[统计信息] 已保存到: {output_txt}")


def save_offset_mapping(
    mapping: Dict[int, int],
    skip_reasons: List[str],
    output_csv: str,
    output_txt: str
):
    """
    保存偏移映射结果

    Args:
        mapping: 原测点→偏移测点映射
        skip_reasons: 跳过原因列表
        output_csv: 偏移测点CSV路径
        output_txt: 映射报告路径
    """
    # 保存偏移测点CSV
    # 注意：必须按原始ID顺序排列，以保持与measures_ID_auto.csv的维度对应关系
    # 错误做法：sorted(mapping.values()) 会按偏移后ID的数值排序，可能打乱维度顺序
    # 正确做法：按原始ID排序后，依次取对应的偏移ID
    sorted_orig_ids = sorted(mapping.keys())
    offset_sensors = [mapping[orig_id] for orig_id in sorted_orig_ids]
    df = pd.DataFrame({'offset_measures': offset_sensors})
    df.to_csv(output_csv, index=False)
    print(f"\n[偏移测点] 已保存到: {output_csv}")

    # 保存映射报告
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("偏移测点映射报告\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"原测点总数: {len(mapping) + len(skip_reasons)}\n")
        f.write(f"成功生成偏移测点数: {len(mapping)}\n")
        f.write(f"跳过的测点数: {len(skip_reasons)}\n\n")

        if skip_reasons:
            f.write("=" * 70 + "\n")
            f.write("跳过的测点:\n")
            f.write("=" * 70 + "\n\n")
            for reason in skip_reasons:
                f.write(f"  - {reason}\n")
            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write(f"偏移映射（前50个）:\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"{'序号':<8}{'原测点ID':<15}{'偏移测点ID':<15}\n")
        f.write("-" * 70 + "\n")

        sorted_mapping = sorted(mapping.items())
        for i, (orig_id, offset_id) in enumerate(sorted_mapping[:50], 1):
            f.write(f"{i:<8}{orig_id:<15}{offset_id:<15}\n")

        if len(sorted_mapping) > 50:
            f.write(f"\n... (共 {len(sorted_mapping)} 个映射)\n")

        # 统计信息
        id_diffs = [abs(offset_id - orig_id) for orig_id, offset_id in mapping.items()]
        f.write("\n" + "=" * 70 + "\n")
        f.write("统计信息:\n")
        f.write("=" * 70 + "\n\n")

        f.write("元素ID差值统计:\n")
        f.write(f"  最小差值: {min(id_diffs)}\n")
        f.write(f"  最大差值: {max(id_diffs)}\n")
        f.write(f"  平均差值: {sum(id_diffs) / len(id_diffs):.2f}\n\n")

        # 检查重复
        unique_offsets = len(set(mapping.values()))
        if unique_offsets == len(mapping):
            f.write("无重复：所有偏移测点唯一\n")
        else:
            f.write(f"警告：存在重复偏移测点（{len(mapping) - unique_offsets} 个重复）\n")

    print(f"[映射报告] 已保存到: {output_txt}")


# ========================================
# 网格解析函数
# ========================================

def parse_inp_manually(inp_file_path):
    """
    手动解析 Abaqus INP 文件的几何部分
    
    Args:
        inp_file_path: INP 文件路径
    
    Returns:
        mesh: meshio.Mesh 对象
    """
    print(f"\n[网格解析] 开始解析INP文件...")
    
    nodes = {}
    elements = {}
    element_types = {}
    
    current_section = None
    current_element_type = None
    in_part = False
    in_assembly = False
    
    with open(inp_file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            if not line or line.startswith('**'):
                continue
            
            if line.startswith('*'):
                line_upper = line.upper()
                
                if line_upper.startswith('*PART'):
                    in_part = True
                    in_assembly = False
                    continue
                
                elif line_upper.startswith('*END PART'):
                    in_part = False
                    continue
                
                elif line_upper.startswith('*ASSEMBLY'):
                    in_assembly = True
                    in_part = False
                    if VERBOSE:
                        print(f"  - 进入Assembly部分（行 {line_num}），停止几何解析")
                    break
                
                if not in_part:
                    continue
                
                if line_upper.startswith('*NODE'):
                    current_section = 'NODE'
                    current_element_type = None
                
                elif line_upper.startswith('*ELEMENT'):
                    current_section = 'ELEMENT'
                    parts = line.split(',')
                    for part in parts:
                        if 'TYPE=' in part.upper():
                            current_element_type = part.split('=')[1].strip()
                            break
                
                else:
                    if current_section in ['NODE', 'ELEMENT']:
                        current_section = None
                        current_element_type = None
                
                continue
            
            if current_section == 'NODE':
                parts = line.split(',')
                if len(parts) >= 4:
                    try:
                        node_id = int(parts[0].strip())
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        nodes[node_id] = [x, y, z]
                    except ValueError:
                        pass
            
            elif current_section == 'ELEMENT' and current_element_type:
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        element_id = int(parts[0].strip())
                        node_ids = [int(p.strip()) for p in parts[1:]]
                        elements[element_id] = node_ids
                        element_types[element_id] = current_element_type
                    except ValueError:
                        pass
    
    print(f"[网格解析] 解析完成")
    print(f"  - 节点数: {len(nodes)}, 元素数: {len(elements)}")
    
    if len(nodes) == 0 or len(elements) == 0:
        raise ValueError("未找到节点或元素")
    
    # 转换为 meshio 格式
    sorted_node_ids = sorted(nodes.keys())
    node_id_to_index = {nid: idx for idx, nid in enumerate(sorted_node_ids)}
    points = np.array([nodes[nid] for nid in sorted_node_ids], dtype=float)
    
    elements_by_type = {}
    for elem_id, node_ids in elements.items():
        elem_type = element_types[elem_id]
        if elem_type not in elements_by_type:
            elements_by_type[elem_type] = []
        
        node_indices = [node_id_to_index[nid] for nid in node_ids]
        elements_by_type[elem_type].append((elem_id, node_indices))
    
    for elem_type in elements_by_type:
        elements_by_type[elem_type].sort(key=lambda x: x[0])
    
    # Abaqus到VTK类型映射
    abaqus_to_vtk = {
        'C3D8': 'hexahedron', 'C3D8R': 'hexahedron',
        'C3D6': 'wedge', 'C3D4': 'tetra',
        'S4': 'quad', 'S4R': 'quad', 'S3': 'triangle',
    }
    
    cells = []
    abaqus_id_to_vtu_index = {}  # 记录Abaqus ID到VTU索引的映射
    vtu_index = 0

    for elem_type, elem_list in elements_by_type.items():
        vtk_type = abaqus_to_vtk.get(elem_type, elem_type.lower())
        connectivity = np.array([elem[1] for elem in elem_list], dtype=int)
        cells.append((vtk_type, connectivity))
        print(f"  - {elem_type} -> {vtk_type}: {len(elem_list)} 个元素")

        # 建立映射：Abaqus元素ID -> VTU索引
        for abaqus_id, _ in elem_list:
            abaqus_id_to_vtu_index[abaqus_id] = vtu_index
            vtu_index += 1

    mesh = meshio.Mesh(points=points, cells=cells)
    return mesh, abaqus_id_to_vtu_index


# ========================================
# 主程序函数
# ========================================

def main(
    inp_file_path: str = None,
    health_inp_path: str = None,
    output_dir: str = None,
    generate_offset: bool = None,
    offset_indices: list = None
) -> int:
    """
    主函数

    Args:
        inp_file_path: INP文件路径
        health_inp_path: 健康模型INP路径（用于偏移测点生成）
        output_dir: 输出目录
        generate_offset: 是否生成偏移测点
        offset_indices: 要偏移的测点行索引列表

    Returns:
        0 表示成功，非0 表示失败
    """
    # 使用传入参数或默认值
    _inp_file_path = inp_file_path or INP_FILE_PATH
    _health_inp_path = health_inp_path or HEALTH_INP_PATH
    _output_dir = output_dir or OUTPUT_DIR
    _generate_offset = generate_offset if generate_offset is not None else GENERATE_OFFSET
    _offset_indices = offset_indices if offset_indices is not None else OFFSET_INDICES

    # 生成偏移文件名（基于偏移配置）
    _offset_filename = get_offset_measures_filename(_offset_indices) if _generate_offset else None
    _offset_name = get_offset_name(_offset_indices) if _generate_offset else None

    # 检查关键输出文件是否已存在
    _base_files = ['measures_ID_original.csv', 'abaqus_id_to_vtu_index.csv']
    if _generate_offset:
        _base_files.append(get_offset_measures_filename(_offset_indices))
        _base_files.append('adjacency_graph.pkl')
    if all(os.path.exists(os.path.join(_output_dir, f)) for f in _base_files):
        print(f"[SKIP] AC output already exists: {_output_dir}")
        return 0

    print("\n" + "=" * 70)
    print("流程03：INP转VTU并自动提取元素集")
    if _generate_offset:
        print("模式：基础转换 + 偏移测点生成")
    else:
        print("模式：基础转换")
    print("=" * 70)

    try:
        # 创建输出目录
        os.makedirs(_output_dir, exist_ok=True)

        # 设置输出文件路径
        vtu_file = os.path.join(_output_dir, OUTPUT_VTU_NAME)
        measures_csv = os.path.join(_output_dir, OUTPUT_MEASURES_CSV)
        elsets_summary = os.path.join(_output_dir, OUTPUT_ELSETS_SUMMARY)
        id_mapping_csv = os.path.join(_output_dir, OUTPUT_ID_MAPPING_CSV)

        # 检查输入文件
        if not os.path.exists(_inp_file_path):
            raise FileNotFoundError(f"INP文件不存在: {_inp_file_path}")

        print(f"\n输入文件: {_inp_file_path}")
        print(f"输出目录: {_output_dir}")

        # 步骤 1: 解析元素集
        print("\n" + "=" * 70)
        print("步骤 1/5: 解析元素集")
        print("=" * 70)
        element_sets = parse_elsets_from_inp(_inp_file_path)

        # 步骤 2: 保存元素集汇总
        print("\n" + "=" * 70)
        print("步骤 2/5: 保存元素集汇总")
        print("=" * 70)
        save_elsets_summary(element_sets, elsets_summary)

        # 步骤 3: 生成测点CSV
        print("\n" + "=" * 70)
        print("步骤 3/5: 生成测点CSV")
        print("=" * 70)
        measure_ids = generate_measures_csv(element_sets, measures_csv, TARGET_ELSET_NAME)

        # 步骤 4: 解析网格
        print("\n" + "=" * 70)
        print("步骤 4/6: 解析网格")
        print("=" * 70)
        mesh, abaqus_id_to_vtu_index = parse_inp_manually(_inp_file_path)

        # 步骤 5: 保存ID映射
        print("\n" + "=" * 70)
        print("步骤 5/6: 保存ID映射文件")
        print("=" * 70)
        save_id_mapping(abaqus_id_to_vtu_index, id_mapping_csv)

        # 步骤 6: 保存VTU
        print("\n" + "=" * 70)
        print("步骤 6/6: 保存VTU文件")
        print("=" * 70)
        print(f"[转换] 保存VTU文件: {vtu_file}")
        mesh.write(vtu_file, file_format="vtu")
        print(f"[成功] VTU文件已保存")

        # 可选步骤：生成偏移测点（用于迁移学习实验）
        if _generate_offset:
            print("\n" + "=" * 70)
            print("可选步骤：生成偏移测点（迁移学习实验）")
            print(f"偏移配置: {_offset_indices} -> {_offset_name}")
            print("=" * 70)

            # 设置偏移文件路径（使用动态文件名）
            adjacency_pkl = os.path.join(_output_dir, OUTPUT_ADJACENCY_GRAPH_PKL)
            adjacency_txt = os.path.join(_output_dir, OUTPUT_ADJACENCY_INFO_TXT)
            offset_csv = os.path.join(_output_dir, _offset_filename)  # 动态文件名
            offset_txt = os.path.join(_output_dir, OUTPUT_OFFSET_MAPPING_TXT)

            # Step 1: 从health.inp解析potential_damage_region元素集
            print(f"\n[偏移-1/4] 从 health.inp 解析搜索区域...")
            print(f"  - 文件路径: {_health_inp_path}")

            if not os.path.exists(_health_inp_path):
                raise FileNotFoundError(f"health.inp 文件不存在: {_health_inp_path}")

            health_elsets = parse_elsets_from_inp(_health_inp_path)

            if OFFSET_REGION_ELSET not in health_elsets:
                raise ValueError(f"未找到元素集 '{OFFSET_REGION_ELSET}' in {_health_inp_path}")

            offset_region_elements = set(health_elsets[OFFSET_REGION_ELSET])
            print(f"  - 搜索区域元素数: {len(offset_region_elements)}")

            # Step 2: 解析元素连接性（从health.inp）
            print(f"\n[偏移-2/4] 解析元素连接性...")
            element_connectivity = parse_element_connectivity(_health_inp_path, offset_region_elements)

            # Step 3: 构建边邻接图
            print(f"\n[偏移-3/4] 构建边邻接图...")
            adjacency_graph = build_adjacency_graph_edge_based(element_connectivity)
            save_adjacency_graph(adjacency_graph, adjacency_pkl, adjacency_txt)

            # Step 4: 生成偏移测点
            print(f"\n[偏移-4/4] 生成偏移测点...")
            offset_sensors, mapping, skip_reasons = generate_offset_sensors(
                measure_ids, adjacency_graph, _offset_indices, OFFSET_RANDOM_SEED
            )
            save_offset_mapping(mapping, skip_reasons, offset_csv, offset_txt)

            print(f"\n[偏移完成] 成功生成 {len(offset_sensors)} 个偏移测点！")
            print(f"  - 成功率: {len(offset_sensors) / len(measure_ids) * 100:.1f}%")
            print(f"  - 跳过: {len(skip_reasons)} 个测点")

        # 总结
        print("\n" + "=" * 70)
        print("流程03完成！")
        print("=" * 70)
        print(f"\n输出文件：")
        print(f"  1. VTU文件:        {os.path.abspath(vtu_file)}")
        print(f"  2. 测点CSV:        {os.path.abspath(measures_csv)}")
        print(f"     >> 重要：此文件将被后续流程使用")
        print(f"  3. ID映射文件:     {os.path.abspath(id_mapping_csv)}")
        print(f"     >> 重要：流程11渲染需要此映射")
        print(f"  4. 元素集汇总:      {os.path.abspath(elsets_summary)}")

        if _generate_offset:
            print(f"\n偏移测点文件（迁移学习实验）：")
            print(f"  5. 邻接图:         {os.path.abspath(adjacency_pkl)}")
            print(f"  6. 邻接图统计:     {os.path.abspath(adjacency_txt)}")
            print(f"  7. 偏移测点CSV:    {os.path.abspath(offset_csv)}")
            print(f"  8. 偏移映射报告:   {os.path.abspath(offset_txt)}")

        if measure_ids:
            print(f"\n自动提取的测点信息：")
            print(f"  - 测点总数: {len(measure_ids)}")
            print(f"  - 前10个ID: {measure_ids[:10]}")
            print(f"  - 后10个ID: {measure_ids[-10:]}")

        print(f"\n关键优势：")
        print(f"  [+] 无需手动准备 measures_ID.csv！")
        print(f"  [+] 所有数据直接从INP文件提取")
        print(f"  [+] 后续流程请使用 '{OUTPUT_MEASURES_CSV}' (未偏移原始测点)")
        if _generate_offset:
            print(f"  [+] 偏移测点已生成（边邻接策略）")
        print()

        return 0

    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
        return 1


# ========================================
# External Config Interface (for AAA_oneclick_run.py)
# ========================================

def collect_all_offsets(config: dict) -> list:
    """
    收集所有case中配置的offset方案（不含offset_count）

    支持两种offset配置格式：
    - 单个方案: [0] 或 [1,2,3]
    - 批量方案: [[0],[1],[2],[3],[1,2],[3,4]]

    注意：offset_count 配置由 collect_all_offset_counts() 单独处理

    Args:
        config: 全局配置字典

    Returns:
        去重后的offset方案列表，每个元素是一个offset索引列表
        例如: [[0], [1], [2], [3], [1,2], [3,4]]
    """
    cases = config.get('cases', [])
    global_offset = config.get('offset')  # 兼容旧的全局offset配置

    offsets = []
    seen = set()

    # 收集case级别的offset
    for case in cases:
        offset = case.get('offset')
        if offset:
            if is_nested_offset(offset):
                # 批量offset方案，展开每个子方案
                for single_offset in offset:
                    key = tuple(sorted(single_offset))
                    if key not in seen:
                        seen.add(key)
                        offsets.append(single_offset)
            else:
                # 单个offset方案
                key = tuple(sorted(offset))
                if key not in seen:
                    seen.add(key)
                    offsets.append(offset)

    # 兼容：如果有全局offset且没有case级别offset，使用全局offset
    if global_offset and not offsets:
        if is_nested_offset(global_offset):
            offsets.extend(global_offset)
        else:
            offsets.append(global_offset)

    return offsets


def collect_all_offset_counts(config: dict) -> list:
    """
    收集所有case中配置的offset_count方案

    Args:
        config: 全局配置字典

    Returns:
        去重后的offset_count列表，如 [0, 1, 2, 3, 4, 5]
    """
    cases = config.get('cases', [])
    counts = set()

    for case in cases:
        offset_count = case.get('offset_count')
        if offset_count is not None:
            if isinstance(offset_count, list):
                for c in offset_count:
                    counts.add(int(c))
            else:
                counts.add(int(offset_count))

    return sorted(counts)


def run_from_config(config: dict) -> int:
    """
    从外部配置运行本脚本

    解析TL_settings.jsonc中的:
    - offset: 全局偏移测点行索引列表（兼容旧配置）
    - offset_count: 按偏移测点个数自动生成嵌套偏移方案（新配置）
    - cases: 检查每个case的offset/offset_count字段

    offset 和 offset_count 两种配置方式兼容共存：
    - 如果case中有 offset 字段，用旧逻辑（指定具体行索引）
    - 如果case中有 offset_count 字段，用新逻辑（按个数自动生成嵌套索引）

    Args:
        config: 从TL_settings.jsonc加载的配置字典

    Returns:
        0 表示成功，非0 表示失败
    """
    print("\n" + "=" * 70)
    print("[AC] 从外部配置运行格式转换和测点提取")
    print("=" * 70)

    # 1. 收集所有case配置的offset方案（旧方式）
    all_offsets = collect_all_offsets(config)

    print(f"[配置] 收集到 {len(all_offsets)} 个offset方案（旧方式）:")
    for offset_indices in all_offsets:
        offset_name = get_offset_name(offset_indices)
        offset_filename = get_offset_measures_filename(offset_indices)
        print(f"  - {offset_indices} -> {offset_name} -> {offset_filename}")

    # 2. 收集所有case配置的offset_count方案（新方式）
    all_offset_counts = collect_all_offset_counts(config)

    if all_offset_counts:
        print(f"\n[配置] 收集到 {len(all_offset_counts)} 档offset_count方案（新方式）:")
        for count in all_offset_counts:
            filename = get_offset_count_measures_filename(count)
            print(f"  - count={count} -> {filename}")

    # 3. 显示配置摘要
    has_any_offset = bool(all_offsets) or bool(all_offset_counts)
    print("\n[配置摘要]")
    print(f"  生成偏移测点: {'是' if has_any_offset else '否'}")
    print(f"  offset方案数（旧方式）: {len(all_offsets)}")
    print(f"  offset_count档数（新方式）: {len(all_offset_counts)}")

    # 4. 调用main函数生成基础输出 + 所有offset方案的测点文件
    print("\n[执行] 开始格式转换和测点提取...")

    # 确定是否需要生成邻接图（旧offset或新offset_count任一需要时都要生成）
    need_adjacency = has_any_offset

    if not has_any_offset:
        # 没有任何offset配置，只生成基础输出
        return main(generate_offset=False, offset_indices=None)

    # --- 处理旧方式的 offset ---
    result = 0
    adjacency_built = False

    if all_offsets:
        for i, offset_indices in enumerate(all_offsets):
            offset_name = get_offset_name(offset_indices)
            print(f"\n[执行-offset {i+1}/{len(all_offsets)}] 生成offset方案: {offset_name}")

            # 第一个方案执行完整流程（含邻接图构建），后续方案只生成偏移测点
            if i == 0:
                result = main(generate_offset=True, offset_indices=offset_indices)
                adjacency_built = True
                if result != 0:
                    return result
            else:
                result = generate_additional_offset(offset_indices)
                if result != 0:
                    return result

    # --- 处理新方式的 offset_count ---
    if all_offset_counts:
        non_zero_counts = [c for c in all_offset_counts if c > 0]

        if not adjacency_built:
            if non_zero_counts:
                # 邻接图还没构建，需要先通过 main() 构建基础输出和邻接图
                # 先运行 main() 生成 measures_ID_original.csv 和邻接图
                # 使用一个临时的单元素索引列表触发邻接图构建
                print(f"\n[执行-offset_count] 构建基础输出和邻接图...")
                result = main(generate_offset=True, offset_indices=[0])
                adjacency_built = True
                if result != 0:
                    return result
            else:
                # 全部为 count=0，只需基础输出
                result = main(generate_offset=False, offset_indices=None)
                if result != 0:
                    return result

        # 现在邻接图已就绪（如果有非零count），生成所有 offset_count 方案的测点文件
        result = generate_offset_count_files(all_offset_counts, seed=OFFSET_RANDOM_SEED)
        if result != 0:
            return result

    return result


def generate_offset_count_files(
    count_list: List[int],
    seed: int = 42,
    targets_json: str = None,
    overwrite: bool = False,
) -> int:
    """
    根据 offset_count 配置，生成所有档位的偏移测点文件

    两种模式：
      1. 默认随机模式（targets_json=None）：使用 generate_nested_offset_sets()
         生成嵌套索引，对每个 count 调用 generate_offset_sensors() 随机选邻居。
      2. 预定义目标模式（targets_json 不为 None）：从 JSON 加载预定义的
         (channel_idx, target_id) 对，按 n_o_step 顺序构造嵌套集合。

    命名规则：measures_ID_offset_count_N.csv

    Args:
        count_list: 偏移个数列表，如 [0, 1, 2, 3, 4, 5]
        seed: 随机种子（仅在随机模式生效）
        targets_json: 预定义目标 JSON 文件路径（如 offset_targets.json）
        overwrite: 是否覆盖已存在的文件

    Returns:
        0 表示成功，非0 表示失败
    """
    try:
        print(f"\n" + "=" * 70)
        print(f"[offset_count] 生成嵌套偏移测点文件")
        print(f"  档位列表: {count_list}")
        print(f"  模式: {'预定义目标 (' + targets_json + ')' if targets_json else '随机邻居'}")
        print("=" * 70)

        # 加载原始测点
        measures_csv = os.path.join(OUTPUT_DIR, OUTPUT_MEASURES_CSV)
        if not os.path.exists(measures_csv):
            print(f"  [错误] 原始测点文件不存在: {measures_csv}")
            return 1

        measures_df = pd.read_csv(measures_csv)
        measure_ids = measures_df['all_measures'].tolist()
        n_sensors = len(measure_ids)
        print(f"  传感器总数: {n_sensors}")

        # 嵌套索引集 + 预定义目标
        predefined_targets: Dict[int, int] = None
        if targets_json:
            import json as _json
            with open(targets_json, "r", encoding="utf-8") as f:
                cfg = _json.load(f)
            ordered_pairs = sorted(
                cfg["pairs_ordered_by_difficulty"], key=lambda p: p["n_o_step"]
            )
            predefined_targets = {
                int(p["channel_idx"]): int(p["target_id"]) for p in ordered_pairs
            }
            ordered_channel_idx = [int(p["channel_idx"]) for p in ordered_pairs]
            nested_sets = {0: []}
            for k in count_list:
                if k == 0:
                    continue
                if k > len(ordered_channel_idx):
                    print(f"  [错误] count={k} 超过 JSON 中定义的对数 {len(ordered_channel_idx)}")
                    return 1
                nested_sets[k] = ordered_channel_idx[:k]
            print(f"  预定义嵌套集合:")
            for k in sorted(nested_sets.keys()):
                print(f"    count={k}: {nested_sets[k]}")
        else:
            nested_sets = generate_nested_offset_sets(n_sensors, count_list, seed=seed)

        # 预加载邻接图（如果有 count>0 的档位需要生成且使用随机模式）
        adjacency_graph = None
        non_zero_counts = [c for c in count_list if c > 0]
        if non_zero_counts and not predefined_targets:
            adjacency_pkl = os.path.join(OUTPUT_DIR, OUTPUT_ADJACENCY_GRAPH_PKL)
            if not os.path.exists(adjacency_pkl):
                print(f"  [错误] 邻接图不存在: {adjacency_pkl}")
                return 1

            with open(adjacency_pkl, 'rb') as f:
                adjacency_graph = pickle.load(f)
            print(f"  邻接图已加载: {len(adjacency_graph)} 个元素")
        elif predefined_targets:
            adjacency_graph = {}  # 空字典，用 predefined_targets 直接选

        # 对每个 count 生成偏移测点文件
        for count in sorted(count_list):
            target_filename = get_offset_count_measures_filename(count)
            target_path = os.path.join(OUTPUT_DIR, target_filename)

            if os.path.exists(target_path) and not overwrite:
                print(f"\n  [跳过] offset_count={count}: 文件已存在 {target_filename}")
                continue

            offset_indices = nested_sets[count]

            if count == 0:
                # count=0: 与 original 完全相同
                df = pd.DataFrame({'offset_measures': measure_ids})
                df.to_csv(target_path, index=False)
                print(f"\n  [成功] offset_count=0: 生成 {target_filename}（与original相同）")
            else:
                # count>0: 调用 generate_offset_sensors
                offset_sensors, mapping, skip_reasons = generate_offset_sensors(
                    measure_ids, adjacency_graph, offset_indices,
                    OFFSET_RANDOM_SEED, predefined_targets=predefined_targets,
                )

                # 保存偏移测点CSV
                offset_txt = os.path.join(OUTPUT_DIR, f"offset_mapping_count_{count}.txt")
                save_offset_mapping(mapping, skip_reasons, target_path, offset_txt)

                print(f"  [成功] offset_count={count}: 生成 {target_filename}")

        print(f"\n[offset_count] 全部 {len(count_list)} 档偏移测点文件生成完成")
        return 0

    except Exception as e:
        print(f"  [错误] 生成offset_count偏移测点失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


def generate_additional_offset(offset_indices: list) -> int:
    """
    生成额外的偏移测点文件（邻接图已存在）

    Args:
        offset_indices: 偏移测点行索引列表

    Returns:
        0 表示成功，非0 表示失败
    """
    try:
        offset_name = get_offset_name(offset_indices)
        offset_filename = get_offset_measures_filename(offset_indices)

        # 检查是否已存在
        offset_csv_path = os.path.join(OUTPUT_DIR, offset_filename)
        if os.path.exists(offset_csv_path):
            print(f"  [跳过] 偏移测点文件已存在: {offset_csv_path}")
            return 0

        # 加载邻接图
        adjacency_pkl = os.path.join(OUTPUT_DIR, OUTPUT_ADJACENCY_GRAPH_PKL)
        if not os.path.exists(adjacency_pkl):
            print(f"  [错误] 邻接图不存在: {adjacency_pkl}")
            return 1

        with open(adjacency_pkl, 'rb') as f:
            adjacency_graph = pickle.load(f)

        # 加载原始测点
        measures_csv = os.path.join(OUTPUT_DIR, OUTPUT_MEASURES_CSV)
        if not os.path.exists(measures_csv):
            print(f"  [错误] 原始测点文件不存在: {measures_csv}")
            return 1

        measures_df = pd.read_csv(measures_csv)
        measure_ids = measures_df['all_measures'].tolist()

        # 生成偏移测点
        offset_sensors, mapping, skip_reasons = generate_offset_sensors(
            measure_ids, adjacency_graph, offset_indices, OFFSET_RANDOM_SEED
        )

        # 保存偏移测点
        offset_txt = os.path.join(OUTPUT_DIR, f"offset_mapping_{offset_name}.txt")
        save_offset_mapping(mapping, skip_reasons, offset_csv_path, offset_txt)

        print(f"  [成功] 生成偏移测点: {offset_filename}")
        return 0

    except Exception as e:
        print(f"  [错误] 生成额外偏移测点失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


# ========================================
# 直接执行入口
# ========================================

def load_jsonc(filepath: str) -> dict:
    """加载JSONC文件（支持//注释和尾随逗号）"""
    import re
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    # 去除 // 注释
    lines = content.split('\n')
    cleaned_lines = []
    for line in lines:
        in_string = False
        result = []
        i = 0
        while i < len(line):
            char = line[i]
            if char == '"' and (i == 0 or line[i-1] != '\\'):
                in_string = not in_string
                result.append(char)
            elif char == '/' and i + 1 < len(line) and line[i+1] == '/' and not in_string:
                break
            else:
                result.append(char)
            i += 1
        cleaned_lines.append(''.join(result))
    content = '\n'.join(cleaned_lines)
    content = re.sub(r',\s*([\]}])', r'\1', content)
    return json.loads(content)


if __name__ == "__main__":
    # 独立运行时，从 TL_settings.jsonc 加载配置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "TL_settings.jsonc")

    if os.path.exists(config_path):
        print(f"[配置] 从 {config_path} 加载配置")
        config = load_jsonc(config_path)
        sys.exit(run_from_config(config))
    else:
        print(f"[错误] 配置文件不存在: {config_path}")
        print("[提示] 请确保 TL_settings.jsonc 存在，或通过 AAA_oneclick_run.py 运行")
        sys.exit(1)
