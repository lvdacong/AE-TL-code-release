"""
A0_generate_repair_regions.py
==============================
BFS growth from seed element to generate nested repair region definitions.
Outputs element ID lists for N_r in {4, 8, 12, 16, 20}.

Usage:
    cd script && python A0_generate_repair_regions.py
"""

import json
import os
import pickle
import sys
from collections import deque
from typing import Dict, List, Set, Tuple

import numpy as np


# ========================================
# 参数配置区
# ========================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

# --- 输入文件 ---
ADJACENCY_GRAPH_PKL = os.path.join(SCRIPT_DIR, "AC_convert_and_extract_output", "adjacency_graph.pkl")
HEALTH_INP_PATH = r"C:\SHM_abaqus_models\health.inp"

# --- 输出文件 ---
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "A0_repair_regions_output.json")

# --- BFS 参数 ---
SEED_ELEMENT = 72048                  # region_2 精确中心，坐标 (11250, 4000, 22750)
Y_TARGET = 4000.0                     # 目标平面 Y 坐标
Y_TOLERANCE = 100.0                   # Y 坐标容差 (mm)
REGION_SIZES = [4, 8, 12, 16, 20]     # 嵌套区域尺寸


# ========================================
# INP 解析函数（参考 AC_convert_and_extract.py）
# ========================================

def parse_nodes_and_elements(inp_file_path: str) -> Tuple[Dict[int, List[float]], Dict[int, List[int]]]:
    """
    从 INP 文件解析所有节点坐标和元素连接性（仅 Part 内的几何数据）

    Args:
        inp_file_path: INP 文件路径

    Returns:
        (nodes, elements)
        - nodes: {node_id: [x, y, z]}
        - elements: {element_id: [node_id1, node_id2, ...]}
    """
    print(f"\n[INP解析] 开始解析: {inp_file_path}")

    nodes = {}
    elements = {}

    current_section = None
    in_part = False

    with open(inp_file_path, 'r', encoding='latin1') as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith('**'):
                continue

            if line.startswith('*'):
                line_upper = line.upper()

                if line_upper.startswith('*PART'):
                    in_part = True
                    continue
                elif line_upper.startswith('*END PART'):
                    in_part = False
                    continue
                elif line_upper.startswith('*ASSEMBLY'):
                    break  # Assembly 之后不再解析

                if not in_part:
                    continue

                if line_upper.startswith('*NODE'):
                    current_section = 'NODE'
                elif line_upper.startswith('*ELEMENT'):
                    current_section = 'ELEMENT'
                else:
                    if current_section in ['NODE', 'ELEMENT']:
                        current_section = None

                continue

            # 解析节点
            if current_section == 'NODE':
                parts = line.split(',')
                if len(parts) >= 4:
                    try:
                        node_id = int(parts[0].strip())
                        x = float(parts[1].strip())
                        y = float(parts[2].strip())
                        z = float(parts[3].strip())
                        nodes[node_id] = [x, y, z]
                    except ValueError:
                        pass

            # 解析元素
            elif current_section == 'ELEMENT':
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        elem_id = int(parts[0].strip())
                        node_ids = [int(p.strip()) for p in parts[1:] if p.strip()]
                        elements[elem_id] = node_ids
                    except ValueError:
                        pass

    print(f"  - 节点数: {len(nodes)}")
    print(f"  - 元素数: {len(elements)}")

    return nodes, elements


# ========================================
# 形心计算
# ========================================

def compute_centroids(
    nodes: Dict[int, List[float]],
    elements: Dict[int, List[int]]
) -> Dict[int, np.ndarray]:
    """
    计算每个元素的形心坐标（节点坐标均值）

    Args:
        nodes: 节点坐标字典
        elements: 元素连接性字典

    Returns:
        {element_id: np.array([cx, cy, cz])}
    """
    print(f"\n[形心计算] 计算 {len(elements)} 个元素的形心...")

    centroids = {}
    missing_count = 0

    for elem_id, node_ids in elements.items():
        coords = []
        for nid in node_ids:
            if nid in nodes:
                coords.append(nodes[nid])
        if coords:
            centroids[elem_id] = np.mean(coords, axis=0)
        else:
            missing_count += 1

    print(f"  - 成功计算: {len(centroids)} 个")
    if missing_count > 0:
        print(f"  - 缺失节点跳过: {missing_count} 个")

    return centroids


# ========================================
# Y=4000 平面过滤
# ========================================

def filter_y_plane(
    adjacency: Dict[int, Set[int]],
    centroids: Dict[int, np.ndarray],
    y_target: float,
    y_tolerance: float
) -> Tuple[Dict[int, Set[int]], Set[int]]:
    """
    从邻接图中过滤出 Y=y_target 平面上的元素子图

    Args:
        adjacency: 完整邻接图
        centroids: 元素形心字典
        y_target: 目标 Y 坐标
        y_tolerance: Y 坐标容差

    Returns:
        (sub_adjacency, plane_elements)
        - sub_adjacency: 过滤后的子图
        - plane_elements: 平面上的元素 ID 集合
    """
    print(f"\n[平面过滤] 过滤 Y={y_target} 平面（容差 ±{y_tolerance}mm）...")

    # 找出所有在 Y 平面上的元素
    plane_elements = set()
    for elem_id, centroid in centroids.items():
        if abs(centroid[1] - y_target) <= y_tolerance:
            if elem_id in adjacency:
                plane_elements.add(elem_id)

    print(f"  - 邻接图元素总数: {len(adjacency)}")
    print(f"  - Y={y_target} 平面元素数: {len(plane_elements)}")

    # 构建子图（只保留平面上元素之间的邻接关系）
    sub_adjacency = {}
    for elem_id in plane_elements:
        neighbors_on_plane = adjacency[elem_id] & plane_elements
        sub_adjacency[elem_id] = neighbors_on_plane

    # 统计子图信息
    neighbor_counts = [len(n) for n in sub_adjacency.values()]
    if neighbor_counts:
        print(f"  - 子图平均邻居数: {np.mean(neighbor_counts):.2f}")
        print(f"  - 子图邻居数范围: [{min(neighbor_counts)}, {max(neighbor_counts)}]")

    return sub_adjacency, plane_elements


# ========================================
# BFS 生长
# ========================================

def bfs_grow(
    sub_adjacency: Dict[int, Set[int]],
    centroids: Dict[int, np.ndarray],
    seed: int,
    max_size: int
) -> List[int]:
    """
    从种子元素出发做 BFS 生长，每层按与种子形心距离排序

    Args:
        sub_adjacency: Y=4000 子图
        centroids: 元素形心
        seed: 种子元素 ID
        max_size: 最大生长数量

    Returns:
        按 BFS 顺序排列的元素 ID 列表（包含种子）
    """
    print(f"\n[BFS生长] 从种子元素 {seed} 开始，目标 {max_size} 个元素...")

    if seed not in sub_adjacency:
        print(f"  [错误] 种子元素 {seed} 不在 Y=4000 子图中！")
        if seed in centroids:
            print(f"  种子元素形心: {centroids[seed]}")
        return []

    seed_centroid = centroids[seed]
    print(f"  - 种子形心: ({seed_centroid[0]:.1f}, {seed_centroid[1]:.1f}, {seed_centroid[2]:.1f})")

    # BFS
    visited = {seed}
    result = [seed]
    queue = deque()

    # 将种子的邻居加入队列
    neighbors = sub_adjacency.get(seed, set()) - visited
    # 按与种子距离排序
    sorted_neighbors = sorted(
        neighbors,
        key=lambda eid: np.linalg.norm(centroids[eid] - seed_centroid)
    )
    for n in sorted_neighbors:
        queue.append(n)

    while queue and len(result) < max_size:
        elem_id = queue.popleft()

        if elem_id in visited:
            continue

        visited.add(elem_id)
        result.append(elem_id)

        if len(result) >= max_size:
            break

        # 将新邻居加入队列（按距离排序）
        new_neighbors = sub_adjacency.get(elem_id, set()) - visited
        sorted_new = sorted(
            new_neighbors,
            key=lambda eid: np.linalg.norm(centroids[eid] - seed_centroid)
        )
        for n in sorted_new:
            if n not in visited:
                queue.append(n)

    print(f"  - 实际生长: {len(result)} 个元素")

    return result


# ========================================
# 验证函数
# ========================================

def validate_regions(
    regions: Dict[int, List[int]],
    centroids: Dict[int, np.ndarray],
    y_target: float,
    y_tolerance: float
):
    """
    验证生成的嵌套区域

    检查项：
    1. 所有元素 Y 坐标是否为 y_target
    2. 嵌套关系是否成立
    3. 每个区域的空间范围

    Args:
        regions: {size: [elem_ids]}
        centroids: 元素形心
        y_target: 目标 Y 坐标
        y_tolerance: Y 容差
    """
    print("\n" + "=" * 70)
    print("验证结果")
    print("=" * 70)

    sizes = sorted(regions.keys())
    all_pass = True

    # 1. Y 坐标检查
    print("\n[验证1] Y 坐标检查:")
    for size in sizes:
        elem_ids = regions[size]
        y_values = [centroids[eid][1] for eid in elem_ids]
        y_ok = all(abs(y - y_target) <= y_tolerance for y in y_values)
        status = "PASS" if y_ok else "FAIL"
        if not y_ok:
            all_pass = False
        print(f"  N={size:2d}: {status}  Y范围=[{min(y_values):.1f}, {max(y_values):.1f}]")

    # 2. 嵌套关系检查
    print("\n[验证2] 嵌套关系检查:")
    for i in range(len(sizes) - 1):
        small = set(regions[sizes[i]])
        large = set(regions[sizes[i + 1]])
        is_subset = small.issubset(large)
        status = "PASS" if is_subset else "FAIL"
        if not is_subset:
            all_pass = False
        print(f"  {sizes[i]} c= {sizes[i+1]}: {status}")

    # 3. 空间范围
    print("\n[验证3] 各区域空间范围:")
    for size in sizes:
        elem_ids = regions[size]
        coords = np.array([centroids[eid] for eid in elem_ids])
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
        dx = x_max - x_min
        dz = z_max - z_min
        print(f"  N={size:2d}: X=[{x_min:.0f}, {x_max:.0f}] (dx={dx:.0f})  "
              f"Z=[{z_min:.0f}, {z_max:.0f}] (dz={dz:.0f})")

    print(f"\n{'='*70}")
    if all_pass:
        print("所有验证通过！")
    else:
        print("存在验证失败项，请检查！")
    print("=" * 70)


# ========================================
# 主函数
# ========================================

def main() -> int:
    """
    主函数

    Returns:
        0 表示成功，非0 表示失败
    """
    print("=" * 70)
    print("A0: 生成嵌套修补区域定义（BFS 生长）")
    print("=" * 70)
    print(f"种子元素: {SEED_ELEMENT}")
    print(f"目标平面: Y={Y_TARGET} (容差 ±{Y_TOLERANCE}mm)")
    print(f"区域尺寸: {REGION_SIZES}")

    try:
        # ----------------------------------------
        # 步骤 1: 加载邻接图
        # ----------------------------------------
        print("\n" + "=" * 70)
        print("步骤 1/5: 加载邻接图")
        print("=" * 70)

        if not os.path.exists(ADJACENCY_GRAPH_PKL):
            raise FileNotFoundError(f"邻接图文件不存在: {ADJACENCY_GRAPH_PKL}")

        with open(ADJACENCY_GRAPH_PKL, 'rb') as f:
            adjacency = pickle.load(f)

        print(f"  - 邻接图元素数: {len(adjacency)}")
        neighbor_counts = [len(n) for n in adjacency.values()]
        print(f"  - 平均邻居数: {np.mean(neighbor_counts):.2f}")

        # ----------------------------------------
        # 步骤 2: 解析元素坐标并计算形心
        # ----------------------------------------
        print("\n" + "=" * 70)
        print("步骤 2/5: 解析元素坐标并计算形心")
        print("=" * 70)

        if not os.path.exists(HEALTH_INP_PATH):
            raise FileNotFoundError(f"health.inp 不存在: {HEALTH_INP_PATH}")

        nodes, elements = parse_nodes_and_elements(HEALTH_INP_PATH)
        centroids = compute_centroids(nodes, elements)

        # 验证种子元素
        if SEED_ELEMENT not in centroids:
            raise ValueError(f"种子元素 {SEED_ELEMENT} 不在解析到的元素中！")

        seed_c = centroids[SEED_ELEMENT]
        print(f"\n  种子元素 {SEED_ELEMENT} 形心: "
              f"({seed_c[0]:.1f}, {seed_c[1]:.1f}, {seed_c[2]:.1f})")

        # ----------------------------------------
        # 步骤 3: 过滤 Y=4000 子图
        # ----------------------------------------
        print("\n" + "=" * 70)
        print("步骤 3/5: 过滤 Y=4000 子图")
        print("=" * 70)

        sub_adjacency, plane_elements = filter_y_plane(
            adjacency, centroids, Y_TARGET, Y_TOLERANCE
        )

        if SEED_ELEMENT not in sub_adjacency:
            raise ValueError(
                f"种子元素 {SEED_ELEMENT} 不在 Y={Y_TARGET} 平面子图中！"
                f" 其 Y 坐标为 {centroids[SEED_ELEMENT][1]:.1f}"
            )

        # ----------------------------------------
        # 步骤 4: BFS 生长
        # ----------------------------------------
        print("\n" + "=" * 70)
        print("步骤 4/5: BFS 生长")
        print("=" * 70)

        max_size = max(REGION_SIZES)
        bfs_order = bfs_grow(sub_adjacency, centroids, SEED_ELEMENT, max_size)

        if len(bfs_order) < max_size:
            print(f"\n  [警告] BFS 只生长到 {len(bfs_order)} 个元素，"
                  f"不足目标 {max_size} 个！")

        # 截取嵌套区域
        regions = {}
        for size in REGION_SIZES:
            if size <= len(bfs_order):
                regions[size] = bfs_order[:size]
            else:
                regions[size] = bfs_order[:]
                print(f"  [警告] N={size} 只有 {len(bfs_order)} 个元素可用")

        # ----------------------------------------
        # 步骤 5: 输出与验证
        # ----------------------------------------
        print("\n" + "=" * 70)
        print("步骤 5/5: 输出与验证")
        print("=" * 70)

        # 打印每个区域的元素ID和形心
        output_data = {}
        for size in REGION_SIZES:
            elem_ids = regions.get(size, [])
            print(f"\n--- N={size} 区域（{len(elem_ids)} 个元素）---")

            # JSON 格式输出（可直接粘贴到 TL_settings.jsonc）
            print(f"  元素ID (JSON): {json.dumps(elem_ids)}")

            # 每个元素的形心
            print(f"  各元素形心:")
            elem_details = []
            for eid in elem_ids:
                c = centroids[eid]
                print(f"    {eid}: ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})")
                elem_details.append({
                    "element_id": eid,
                    "centroid": [round(c[0], 1), round(c[1], 1), round(c[2], 1)]
                })

            output_data[f"N{size}"] = {
                "element_ids": elem_ids,
                "details": elem_details
            }

        # 验证
        validate_regions(regions, centroids, Y_TARGET, Y_TOLERANCE)

        # 保存到 JSON
        save_data = {
            "seed_element": SEED_ELEMENT,
            "seed_centroid": [
                round(seed_c[0], 1),
                round(seed_c[1], 1),
                round(seed_c[2], 1)
            ],
            "y_plane": Y_TARGET,
            "y_tolerance": Y_TOLERANCE,
            "regions": {}
        }
        for size in REGION_SIZES:
            elem_ids = regions.get(size, [])
            save_data["regions"][f"N{size}"] = {
                "element_ids": elem_ids,
                "count": len(elem_ids)
            }

        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"\n[保存] 结果已保存到: {OUTPUT_JSON}")

        # 最终汇总：可直接粘贴到 TL_settings.jsonc 的格式
        print("\n" + "=" * 70)
        print("可粘贴到 TL_settings.jsonc 的 region_by_elem_ID 配置:")
        print("=" * 70)
        for size in REGION_SIZES:
            elem_ids = regions.get(size, [])
            label = f"repair_{size}"
            print(f'  "{label}": {json.dumps(elem_ids)},')

        print("\n" + "=" * 70)
        print("A0 完成！")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
        return 1


# ========================================
# 直接执行入口
# ========================================

if __name__ == "__main__":
    sys.exit(main())
