# -*- coding: utf-8 -*-
"""
AB_generate_simulation_data.py
===============================
Batch FEM simulation runner. Generates all SHM datasets (healthy, damaged,
repaired) via CAE method with auto-cleanup of intermediate files.

Usage:
    cd script && python AB_generate_simulation_data.py
"""

import os
import sys
import time
import random
import shutil
import subprocess
import argparse
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# ========================================
# 数据集配置
# ========================================
# 【配置来源说明】
#   - 通过 AAA_oneclick_run.py 调用时：从 TL_settings.jsonc 读取
#     (FEM_models + simulation_counts)
#   - 独立运行本脚本时：使用下方默认配置
#
# ========================================

# 独立运行时的默认配置（通过 run_from_config 调用时会被覆盖）
DATASET_CONFIGS = {
    # 此配置仅供独立运行使用，实际运行时应通过 TL_settings.jsonc 配置
    # 详见 TL_settings.jsonc 的 FEM_models 和 simulation_counts 区
}

# ========================================
# 全局配置
# ========================================

# 路径配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(SCRIPT_DIR)
CAE_MODEL_DIR = r"C:\SHM_abaqus_models"  # CAE模型文件目录
AUXILIARY_SCRIPT = os.path.join(SCRIPT_DIR, "AB_generate_simulation_data_auxiliary.py")
POSTPROCESS_SCRIPT = os.path.join(SCRIPT_DIR, "AB_generate_simulation_data_postprocess_auxiliary.py")

# Abaqus命令
ABAQUS_CMD = "abq2025hf1.bat"

# 并行配置
MAX_WORKERS = 4     # 并行workers
CPUS_PER_JOB = 1    # 每个Abaqus作业的CPU数

# 文件清理策略：前10次迭代保留完整记录，之后清理
KEEP_FULL_ITERATIONS = 10  # 保留前10次迭代的所有文件

# ========================================
# Helper Functions
# ========================================

def generate_random_params():
    """生成随机载荷参数（与01脚本一致）"""
    params = {
        'h1': random.uniform(4000, 15000),
        'h2': random.uniform(4000, 10000),
        'h3': random.uniform(4000, 15000),
        'draft': random.uniform(10000, 30000),
        'mx1': random.uniform(1e8, 1e9) * random.choice([-1, 1]),
        'my1': random.uniform(1e8, 1e9) * random.choice([-1, 1]),
        'mz1': random.uniform(1e8, 1e9) * random.choice([-1, 1]),
        'mx2': random.uniform(1e8, 1e9) * random.choice([-1, 1]),
        'my2': random.uniform(1e8, 1e9) * random.choice([-1, 1]),
        'mz2': random.uniform(1e8, 1e9) * random.choice([-1, 1]),
    }
    return params


def save_loading_conditions(output_dir, params):
    """保存载荷条件到CSV"""
    csv_path = os.path.join(output_dir, "loading_conditions.csv")
    with open(csv_path, 'w') as f:
        f.write("parameter,value\n")
        for key, value in params.items():
            f.write(f"{key},{value}\n")


def run_postprocess(output_dir, iteration_num, abaqus_cmd=None, postprocess_script=None, delete_odb=True):
    """运行后处理"""
    # 使用传入的参数，避免依赖全局变量
    if abaqus_cmd is None:
        abaqus_cmd = ABAQUS_CMD
    if postprocess_script is None:
        postprocess_script = POSTPROCESS_SCRIPT

    # 设置环境变量
    env = os.environ.copy()
    env['SHM_DELETE_ODB'] = '0' if not delete_odb else '1'

    # 运行后处理
    odb_path = os.path.join(output_dir, "iteration.odb")
    cmd = f'"{abaqus_cmd}" python "{postprocess_script}" -- "{odb_path}" "{output_dir}" {iteration_num}'

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600,
            env=env,
            encoding='utf-8',
            errors='ignore',
            cwd=output_dir  # 设置工作目录为迭代文件夹，确保.rpy文件生成在正确位置
        )

        if result.returncode == 0:
            # 验证NPY文件是否创建
            npy_path = os.path.join(output_dir, "iteration.npy")
            if os.path.exists(npy_path):
                return {'success': True}
            else:
                return {'success': False, 'error': f"NPY file not created. Stdout: {result.stdout[:500]}, Stderr: {result.stderr[:500]}"}
        else:
            return {'success': False, 'error': f"Postprocess failed (rc={result.returncode}): Stderr: {result.stderr[:500]}, Stdout: {result.stdout[:500]}"}

    except subprocess.TimeoutExpired:
        return {'success': False, 'error': 'Postprocessing timeout'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def run_iteration(iteration_num, template_cae, output_base_dir, dataset_name, abaqus_cmd=None, auxiliary_script=None, postprocess_script=None, delete_odb=True, delete_cae=True, delete_inp=True):
    """执行单个迭代的完整流程"""
    # 使用传入的参数，避免依赖全局变量
    if abaqus_cmd is None:
        abaqus_cmd = ABAQUS_CMD
    if auxiliary_script is None:
        auxiliary_script = AUXILIARY_SCRIPT
    if postprocess_script is None:
        postprocess_script = POSTPROCESS_SCRIPT

    t_start = time.time()

    try:
        # 创建迭代目录
        output_dir = os.path.join(output_base_dir, str(iteration_num))
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"[{dataset_name}] 迭代 {iteration_num} 开始 (时间: {datetime.now().strftime('%H:%M:%S')})")
        print(f"{'='*60}")

        # Step 1: 复制CAE文件
        print(f"[{dataset_name}/{iteration_num}] Step 1/4: 复制CAE文件...")
        work_cae = os.path.join(output_dir, "model_work.cae")

        if not os.path.exists(template_cae):
            return {
                'success': False,
                'iteration': iteration_num,
                'dataset': dataset_name,
                'error': f'Template CAE not found: {template_cae}',
                'stage': 'CAE_COPY'
            }

        shutil.copy2(template_cae, work_cae)
        print(f"[{dataset_name}/{iteration_num}]   CAE复制完成")

        # Step 2: 生成随机参数
        print(f"[{dataset_name}/{iteration_num}] Step 2/4: 生成随机参数...")
        params = generate_random_params()
        save_loading_conditions(output_dir, params)
        print(f"[{dataset_name}/{iteration_num}]   参数生成完成: h1={params['h1']:.1f}, h2={params['h2']:.1f}, h3={params['h3']:.1f}")

        # Step 3: 调用辅助脚本修改CAE并运行求解器
        print(f"[{dataset_name}/{iteration_num}] Step 3/4: 调用Abaqus修改CAE并计算...")
        cmd = (
            f'"{abaqus_cmd}" cae noGUI="{auxiliary_script}" -- '
            f'"{work_cae}" "{output_dir}" '
            f'{params["h1"]} {params["h2"]} {params["h3"]} {params["draft"]} '
            f'{params["mx1"]} {params["my1"]} {params["mz1"]} '
            f'{params["mx2"]} {params["my2"]} {params["mz2"]} '
            f'{iteration_num}'
        )

        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=3600,  # 1小时超时
            cwd=output_dir  # 设置工作目录为迭代文件夹，确保.rpy文件生成在正确位置
        )

        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            return {
                'success': False,
                'iteration': iteration_num,
                'dataset': dataset_name,
                'error': f'Auxiliary script failed: {error_msg[:500]}',
                'stage': 'CAE_MODIFY'
            }

        print(f"[{dataset_name}/{iteration_num}]   Abaqus计算完成")

        # Step 4: 后处理ODB
        print(f"[{dataset_name}/{iteration_num}] Step 4/4: 后处理ODB...")
        odb_path = os.path.join(output_dir, "iteration.odb")

        if not os.path.exists(odb_path):
            return {
                'success': False,
                'iteration': iteration_num,
                'dataset': dataset_name,
                'error': 'ODB file not generated',
                'stage': 'SOLVE'
            }

        post_result = run_postprocess(output_dir, iteration_num, abaqus_cmd, postprocess_script, delete_odb)

        if not post_result['success']:
            return {
                'success': False,
                'iteration': iteration_num,
                'dataset': dataset_name,
                'error': post_result.get('error', 'Unknown postprocess error'),
                'stage': 'POSTPROCESS'
            }

        print(f"[{dataset_name}/{iteration_num}]   后处理完成")

        # 验证输出文件
        npy_path = os.path.join(output_dir, "iteration.npy")
        info_path = os.path.join(output_dir, "iteration_info.txt")

        if not os.path.exists(npy_path):
            return {
                'success': False,
                'iteration': iteration_num,
                'dataset': dataset_name,
                'error': 'NPY file not generated',
                'stage': 'POSTPROCESS'
            }

        # 可选清理
        if delete_cae:
            try:
                os.remove(work_cae)
                # 删除jnl文件
                jnl_file = os.path.join(output_dir, "model_work.jnl")
                if os.path.exists(jnl_file):
                    os.remove(jnl_file)
            except:
                pass

        if delete_inp:
            try:
                inp_file = os.path.join(output_dir, "iteration.inp")
                if os.path.exists(inp_file):
                    os.remove(inp_file)
            except:
                pass

        t_total = time.time() - t_start

        print(f"[{dataset_name}/{iteration_num}] 完成！总耗时: {t_total:.2f} 秒")
        print(f"{'='*60}\n")

        return {
            'success': True,
            'iteration': iteration_num,
            'dataset': dataset_name,
            't_total': t_total
        }

    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'iteration': iteration_num,
            'dataset': dataset_name,
            'error': 'Timeout (3600s)',
            'stage': 'TIMEOUT'
        }
    except Exception as e:
        return {
            'success': False,
            'iteration': iteration_num,
            'dataset': dataset_name,
            'error': f'{str(e)}\n{traceback.format_exc()}',
            'stage': 'UNKNOWN'
        }


def check_iteration_completeness(iteration_dir, min_size_kb=1400, max_size_kb=1600):
    """
    检查单个迭代文件夹是否完整

    Args:
        iteration_dir: 迭代文件夹路径
        min_size_kb: NPY文件最小大小（KB），默认1400
        max_size_kb: NPY文件最大大小（KB），默认1600

    Returns:
        (is_complete, message): (True/False, 状态消息)
    """
    npy_file = os.path.join(iteration_dir, "iteration.npy")

    if not os.path.exists(npy_file):
        return False, "iteration.npy not found"

    size_bytes = os.path.getsize(npy_file)
    size_kb = size_bytes / 1024.0

    if size_kb < min_size_kb:
        return False, f"iteration.npy too small ({size_kb:.1f} KB < {min_size_kb} KB)"
    if size_kb > max_size_kb:
        return False, f"iteration.npy too large ({size_kb:.1f} KB > {max_size_kb} KB)"

    return True, f"OK ({size_kb:.1f} KB)"


def find_incomplete_iterations(output_base_dir, lookback=4):
    """
    查找不完整的迭代，回溯检查最后N个迭代

    Args:
        output_base_dir: 输出基础目录
        lookback: 回溯检查的迭代数量

    Returns:
        (start_iteration, incomplete_list):
            - start_iteration: 应该开始的迭代编号（第一个不完整的位置）
            - incomplete_list: 不完整的迭代编号列表
    """
    try:
        if not os.path.exists(output_base_dir):
            return 1, []

        # 获取所有数字编号的文件夹
        all_iterations = []
        for name in os.listdir(output_base_dir):
            full_path = os.path.join(output_base_dir, name)
            if os.path.isdir(full_path) and name.isdigit():
                all_iterations.append(int(name))

        if not all_iterations:
            return 1, []

        all_iterations.sort()
        max_iteration = max(all_iterations)

        # 回溯检查最后N个迭代
        check_start = max(1, max_iteration - lookback + 1)
        incomplete_list = []

        for i in range(check_start, max_iteration + 1):
            iteration_dir = os.path.join(output_base_dir, str(i))
            if os.path.exists(iteration_dir):
                is_complete, msg = check_iteration_completeness(iteration_dir)
                if not is_complete:
                    incomplete_list.append(i)

        # 如果有不完整的迭代，从第一个不完整的位置开始
        if incomplete_list:
            return min(incomplete_list), incomplete_list
        else:
            # 所有检查的迭代都完整，从max+1开始
            return max_iteration + 1, []

    except Exception as e:
        print(f"[WARNING] Error finding incomplete iterations: {e}")
        return 1, []


def delete_incomplete_iterations(output_base_dir, incomplete_list):
    """
    删除不完整的迭代文件夹

    Args:
        output_base_dir: 输出基础目录
        incomplete_list: 不完整的迭代编号列表

    Returns:
        deleted_count: 成功删除的文件夹数量
    """
    deleted_count = 0
    for iteration_num in incomplete_list:
        iteration_dir = os.path.join(output_base_dir, str(iteration_num))
        if os.path.exists(iteration_dir):
            try:
                shutil.rmtree(iteration_dir)
                print(f"[CLEANUP] Deleted incomplete iteration folder: {iteration_dir}")
                deleted_count += 1
            except Exception as e:
                print(f"[ERROR] Failed to delete {iteration_dir}: {e}")

    return deleted_count


def verify_all_iterations(output_base_dir, expected_count):
    """
    全盘验证所有迭代的完整性

    Args:
        output_base_dir: 输出基础目录
        expected_count: 期望的迭代总数

    Returns:
        (missing_list, incomplete_list):
            - missing_list: 缺失的迭代编号列表
            - incomplete_list: 不完整的迭代编号列表
    """
    missing_list = []
    incomplete_list = []

    for i in range(1, expected_count + 1):
        iteration_dir = os.path.join(output_base_dir, str(i))

        if not os.path.exists(iteration_dir):
            missing_list.append(i)
        else:
            is_complete, msg = check_iteration_completeness(iteration_dir)
            if not is_complete:
                incomplete_list.append(i)

    return missing_list, incomplete_list


def _find_max_suffix(path):
    """查找目录中最大的数字编号（已废弃，使用find_incomplete_iterations代替）"""
    try:
        max_id = 0
        if not os.path.exists(path):
            return 0
        for name in os.listdir(path):
            full = os.path.join(path, name)
            if not os.path.isdir(full):
                continue
            if name.isdigit():
                try:
                    n = int(name)
                    if n > max_id:
                        max_id = n
                except Exception:
                    pass
        return max_id
    except Exception:
        return 0


# ========================================
# 主执行函数
# ========================================

def main():
    """主执行函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='02 统一数据生成脚本（CAE方法）- 支持断点续传和查缺补漏',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法（支持自动断点续传）
  python AB_generate_simulation_data.py

  # 强制重新生成所有迭代
  python AB_generate_simulation_data.py --force

  # 设置最大重试次数为5
  python AB_generate_simulation_data.py --max-retries 5

  # 跳过最终验证（不推荐）
  python AB_generate_simulation_data.py --skip-verify

功能说明:
  - 自动断点续传: 检测已完成的迭代，从中断处继续
  - 回溯检查: 检查最后4个迭代的完整性
  - 查缺补漏: 完成后全盘验证，自动重试失败的迭代
  - 完整性判定: 检查NPY文件存在且大小约1485KB
        """
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重新生成所有迭代（忽略已存在的迭代）'
    )

    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='失败迭代的最大重试次数（默认: 3）'
    )

    parser.add_argument(
        '--skip-verify',
        action='store_true',
        help='跳过最终验证步骤（不推荐）'
    )

    parser.add_argument(
        '--lookback',
        type=int,
        default=4,
        help='回溯检查的迭代数量（默认: 4）'
    )

    args = parser.parse_args()

    print("="*80)
    print("AB 统一数据生成脚本（CAE方法）- 生产模式")
    print("="*80)
    print(f"CAE模型目录: {CAE_MODEL_DIR}")
    print(f"CAE修改脚本: {AUXILIARY_SCRIPT}")
    print(f"后处理脚本: {POSTPROCESS_SCRIPT}")
    print(f"最大并发: {MAX_WORKERS}")
    print(f"Abaqus命令: {ABAQUS_CMD}")
    print(f"文件清理策略: 前{KEEP_FULL_ITERATIONS}次迭代保留完整记录，之后清理ODB/CAE/INP文件")
    print(f"\n运行模式:")
    print(f"  - 强制重新生成: {'是' if args.force else '否'}")
    print(f"  - 回溯检查迭代数: {args.lookback}")
    print(f"  - 最大重试次数: {args.max_retries}")
    print(f"  - 跳过验证: {'是' if args.skip_verify else '否'}")
    print("="*80)

    # 验证辅助脚本存在
    if not os.path.exists(AUXILIARY_SCRIPT):
        print(f"\nERROR: 辅助脚本不存在: {AUXILIARY_SCRIPT}")
        return 1

    # 验证后处理脚本存在
    if not os.path.exists(POSTPROCESS_SCRIPT):
        print(f"\nERROR: 后处理脚本不存在: {POSTPROCESS_SCRIPT}")
        return 1

    # 显示数据集配置
    print("\n数据集配置:")
    total_iterations = 0
    for idx, (dataset_name, config) in enumerate(DATASET_CONFIGS.items(), 1):
        count = config['count']
        if count == 0:
            continue  # 跳过count=0的数据集
        cae_file = os.path.join(CAE_MODEL_DIR, config['cae_template'])
        exists = "OK" if os.path.exists(cae_file) else "!!"
        print(f"  {idx}. [{exists}] {config['name']:15s} - {count:4d} 个样本 ({config['cae_template']})")
        total_iterations += count
    print(f"\n  总计: {total_iterations} 个样本")
    print("="*80)

    # 检查CAE文件
    missing_files = []
    for dataset_name, config in DATASET_CONFIGS.items():
        if config['count'] == 0:
            continue  # 跳过count=0的数据集
        cae_file = os.path.join(CAE_MODEL_DIR, config['cae_template'])
        if not os.path.exists(cae_file):
            missing_files.append(cae_file)

    if missing_files:
        print("\nERROR: 以下CAE文件缺失:")
        for f in missing_files:
            print(f"  - {f}")
        return 1

    # 构建任务队列（支持断点续传和回溯检查）
    print("\n" + "="*80)
    print("STEP 1: 扫描输出目录并执行断点续传检查")
    print("="*80)
    all_tasks = []
    dataset_resume_info = {}  # 记录每个数据集的断点续传信息

    for dataset_name, config in DATASET_CONFIGS.items():
        num_iterations = config['count']
        if num_iterations == 0:
            continue  # 跳过count=0的数据集

        template_cae = os.path.join(CAE_MODEL_DIR, config['cae_template'])
        output_base_dir = config['output_dir']

        # 确保输出目录存在
        os.makedirs(output_base_dir, exist_ok=True)

        print(f"\n[{dataset_name}] 检查输出目录: {output_base_dir}")

        # 使用新的断点续传逻辑或强制重新生成
        if args.force:
            print(f"[{dataset_name}] 强制模式: 将重新生成所有 {num_iterations} 个迭代")
            start_iteration = 1
            incomplete_list = []

            # 如果强制模式，删除所有现有迭代
            if os.path.exists(output_base_dir):
                existing_iterations = [int(name) for name in os.listdir(output_base_dir)
                                       if os.path.isdir(os.path.join(output_base_dir, name)) and name.isdigit()]
                if existing_iterations:
                    print(f"[{dataset_name}] 删除现有的 {len(existing_iterations)} 个迭代文件夹...")
                    for iter_num in existing_iterations:
                        iter_dir = os.path.join(output_base_dir, str(iter_num))
                        try:
                            shutil.rmtree(iter_dir)
                        except Exception as e:
                            print(f"[{dataset_name}] 警告: 无法删除 {iter_dir}: {e}")
        else:
            # 使用断点续传逻辑（回溯检查）
            start_iteration, incomplete_list = find_incomplete_iterations(output_base_dir, lookback=args.lookback)

            # 删除不完整的迭代
            if incomplete_list:
                print(f"[{dataset_name}] 发现 {len(incomplete_list)} 个不完整的迭代: {incomplete_list}")
                deleted = delete_incomplete_iterations(output_base_dir, incomplete_list)
                print(f"[{dataset_name}] 已删除 {deleted} 个不完整的文件夹")

            # 显示断点续传信息
            if start_iteration > 1:
                print(f"[{dataset_name}] 断点续传: 从迭代 {start_iteration} 开始")
                print(f"[{dataset_name}] 已完成: {start_iteration - 1} / {num_iterations}")
            else:
                print(f"[{dataset_name}] 首次运行: 将生成全部 {num_iterations} 个迭代")

        # 记录断点续传信息
        dataset_resume_info[dataset_name] = {
            'start_iteration': start_iteration,
            'incomplete_deleted': len(incomplete_list),
            'total_iterations': num_iterations
        }

        # 创建任务：只生成从start_iteration到num_iterations的部分
        tasks_to_generate = num_iterations - start_iteration + 1
        if tasks_to_generate > 0:
            for i in range(tasks_to_generate):
                iteration_num = start_iteration + i
                all_tasks.append((iteration_num, template_cae, output_base_dir, dataset_name))
        else:
            print(f"[{dataset_name}] 所有迭代已完成，无需生成新任务")

    print(f"\n断点续传汇总:")
    for dataset_name, info in dataset_resume_info.items():
        if info['start_iteration'] > 1:
            print(f"  [{dataset_name}] 从迭代 {info['start_iteration']} 继续 "
                  f"(已完成 {info['start_iteration']-1}/{info['total_iterations']}, "
                  f"删除不完整 {info['incomplete_deleted']})")
        else:
            print(f"  [{dataset_name}] 首次运行 (总计 {info['total_iterations']} 个迭代)")

    print(f"\n  总任务数: {len(all_tasks)}")
    print(f"  并发数: {MAX_WORKERS}")
    print("="*80)

    # 如果没有任务需要执行，跳过执行阶段
    if len(all_tasks) == 0:
        print("\n所有迭代已完成，无需生成新任务")
        print("进入全盘验证阶段...")
    else:
        # 执行任务
        print("\n" + "="*80)
        print(f"STEP 2: 并行执行 {len(all_tasks)} 个任务")
        print("="*80 + "\n")

    t_start = time.time()
    results = []
    completed = 0
    failed = 0

    # 按数据集统计
    dataset_stats = {name: {'completed': 0, 'failed': 0} for name in DATASET_CONFIGS.keys()}

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for task in all_tasks:
            iteration_num = task[0]
            # 文件清理策略：前KEEP_FULL_ITERATIONS次迭代保留所有文件，之后清理
            should_clean = iteration_num > KEEP_FULL_ITERATIONS

            future = executor.submit(
                run_iteration,
                task[0],  # iteration_num
                task[1],  # template_cae
                task[2],  # output_base_dir
                task[3],  # dataset_name
                ABAQUS_CMD,  # abaqus_cmd
                AUXILIARY_SCRIPT,  # auxiliary_script
                POSTPROCESS_SCRIPT,  # postprocess_script
                should_clean,  # delete_odb
                should_clean,  # delete_cae
                should_clean   # delete_inp
            )
            futures[future] = task

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            if result['success']:
                completed += 1
                dataset_stats[result['dataset']]['completed'] += 1
                print(f"\n[OK] 进度: {completed + failed}/{len(all_tasks)} 完成 "
                      f"(成功={completed}, 失败={failed})")
            else:
                failed += 1
                dataset_stats[result['dataset']]['failed'] += 1
                print(f"\n[FAIL] [{result['dataset']}] 迭代 {result['iteration']} 失败: {result.get('error', 'Unknown')[:100]}")
                print(f"   阶段: {result.get('stage', 'Unknown')}")
                print(f"   进度: {completed + failed}/{len(all_tasks)} 完成 "
                      f"(成功={completed}, 失败={failed})")

    t_total = time.time() - t_start

    # ========================================
    # STEP 3: 全盘验证和查缺补漏
    # ========================================
    if args.skip_verify:
        print("\n" + "="*80)
        print("STEP 3: 跳过验证（用户指定 --skip-verify）")
        print("="*80)
        all_verified = False  # 未验证，需要在最终汇总中检查
    else:
        print("\n" + "="*80)
        print("STEP 3: 全盘验证和查缺补漏")
        print("="*80)

        max_retries = args.max_retries
        retry_count = 0
        all_verified = False

        while retry_count < max_retries and not all_verified:
            if retry_count > 0:
                print(f"\n{'='*80}")
                print(f"重试 {retry_count}/{max_retries}: 检测到失败的迭代")
                print(f"{'='*80}")

            # 验证所有数据集
            retry_tasks = []
            verification_summary = {}

            for dataset_name, config in DATASET_CONFIGS.items():
                if config['count'] == 0:
                    continue

                output_base_dir = config['output_dir']
                template_cae = os.path.join(CAE_MODEL_DIR, config['cae_template'])
                expected_count = config['count']

                # 全盘验证
                missing_list, incomplete_list = verify_all_iterations(output_base_dir, expected_count)

                failed_list = missing_list + incomplete_list
                verification_summary[dataset_name] = {
                    'missing': len(missing_list),
                    'incomplete': len(incomplete_list),
                    'total_failed': len(failed_list)
                }

                # 如果有失败的迭代，添加到重试任务
                if failed_list:
                    print(f"\n[{dataset_name}] 发现 {len(failed_list)} 个失败的迭代:")
                    print(f"  - 缺失: {len(missing_list)} 个 {missing_list[:10]}")
                    print(f"  - 不完整: {len(incomplete_list)} 个 {incomplete_list[:10]}")

                    # 删除不完整的迭代文件夹
                    if incomplete_list:
                        deleted = delete_incomplete_iterations(output_base_dir, incomplete_list)
                        print(f"  - 已删除 {deleted} 个不完整的文件夹")

                    # 添加到重试任务
                    for iteration_num in failed_list:
                        retry_tasks.append((iteration_num, template_cae, output_base_dir, dataset_name))

            # 检查是否所有迭代都通过验证
            total_failed = sum(info['total_failed'] for info in verification_summary.values())

            if total_failed == 0:
                print("\n" + "="*80)
                print("验证成功：所有迭代都已完成！")
                print("="*80)
                all_verified = True
                break
            else:
                print(f"\n发现 {total_failed} 个失败的迭代，准备重试...")

            # 重试失败的任务
            if retry_tasks:
                retry_count += 1
                print(f"\n开始重试 {len(retry_tasks)} 个失败的迭代...")

                # 执行重试任务（使用相同的并行逻辑）
                retry_results = []
                retry_completed = 0
                retry_failed = 0

                with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = {}
                    for task in retry_tasks:
                        iteration_num = task[0]
                        should_clean = iteration_num > KEEP_FULL_ITERATIONS

                        future = executor.submit(
                            run_iteration,
                            task[0],  # iteration_num
                            task[1],  # template_cae
                            task[2],  # output_base_dir
                            task[3],  # dataset_name
                            ABAQUS_CMD,
                            AUXILIARY_SCRIPT,
                            POSTPROCESS_SCRIPT,
                            should_clean,
                            should_clean,
                            should_clean
                        )
                        futures[future] = task

                    for future in as_completed(futures):
                        result = future.result()
                        retry_results.append(result)

                        if result['success']:
                            retry_completed += 1
                        else:
                            retry_failed += 1
                            print(f"\n[RETRY FAIL] [{result['dataset']}] 迭代 {result['iteration']} 重试失败: {result.get('error', 'Unknown')[:100]}")

                print(f"\n重试完成: 成功 {retry_completed} / 失败 {retry_failed}")

            # 如果达到最大重试次数，退出循环
            if retry_count >= max_retries:
                print(f"\n{'='*80}")
                print(f"警告：已达到最大重试次数 ({max_retries})")
                print(f"{'='*80}")
                break

    # ========================================
    # STEP 4: 最终验证和汇总
    # ========================================
    print("\n" + "="*80)
    print("STEP 4: 最终验证和执行汇总")
    print("="*80)

    # 最终验证所有数据集
    final_verification = {}
    all_datasets_complete = True

    for dataset_name, config in DATASET_CONFIGS.items():
        if config['count'] == 0:
            continue

        output_base_dir = config['output_dir']
        expected_count = config['count']

        missing_list, incomplete_list = verify_all_iterations(output_base_dir, expected_count)
        total_complete = expected_count - len(missing_list) - len(incomplete_list)

        final_verification[dataset_name] = {
            'expected': expected_count,
            'complete': total_complete,
            'missing': len(missing_list),
            'incomplete': len(incomplete_list)
        }

        if missing_list or incomplete_list:
            all_datasets_complete = False

    # 显示最终验证结果
    print("\n最终验证结果:")
    print("-"*80)
    for dataset_name, info in final_verification.items():
        status_icon = "✓" if info['missing'] == 0 and info['incomplete'] == 0 else "✗"
        print(f"  [{status_icon}] {dataset_name:20s}: {info['complete']}/{info['expected']} 完成", end="")
        if info['missing'] > 0 or info['incomplete'] > 0:
            print(f" (缺失:{info['missing']}, 不完整:{info['incomplete']})")
        else:
            print()
    print("="*80)

    # 执行统计
    print("\n执行统计:")
    print(f"  初始任务数: {len(all_tasks)}")
    print(f"  成功: {completed}")
    print(f"  失败: {failed}")
    print(f"  总耗时: {t_total:.2f} 秒 ({t_total/60:.2f} 分钟)")
    if completed > 0:
        print(f"  平均耗时: {t_total/completed:.2f} 秒/任务")
    print("="*80)

    # 按数据集统计
    print("\n按数据集统计:")
    print("-"*80)
    for dataset_name in DATASET_CONFIGS.keys():
        stats = dataset_stats[dataset_name]
        total = stats['completed'] + stats['failed']
        if total > 0:
            print(f"  {dataset_name:20s}: 成功 {stats['completed']:4d} / 失败 {stats['failed']:2d} / 总计 {total:4d}")
    print("="*80)

    # 保存汇总
    summary_dir = r"C:\SHM_abaqus_data"
    os.makedirs(summary_dir, exist_ok=True)
    summary_file = os.path.join(summary_dir, "execution_summary_cae_method.txt")

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("AB CAE Method - Execution Summary\n")
        f.write("="*80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Mode: Production\n")
        f.write(f"Cleanup Strategy: Keep first {KEEP_FULL_ITERATIONS} iterations fully, clean others\n")
        f.write(f"Total tasks: {len(all_tasks)}\n")
        f.write(f"Successful: {completed}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Total time: {t_total:.2f} seconds ({t_total/60:.2f} minutes)\n")
        if completed > 0:
            f.write(f"Average time: {t_total/completed:.2f} seconds/task\n")
        f.write("="*80 + "\n\n")

        f.write("Dataset Statistics:\n")
        f.write("-"*80 + "\n")
        for dataset_name in DATASET_CONFIGS.keys():
            stats = dataset_stats[dataset_name]
            total = stats['completed'] + stats['failed']
            if total > 0:
                f.write(f"  {dataset_name:20s}: Success {stats['completed']:4d} / Failed {stats['failed']:2d} / Total {total:4d}\n")
        f.write("="*80 + "\n\n")

        f.write("Individual Results:\n")
        f.write("-"*80 + "\n")
        for r in sorted(results, key=lambda x: (x['dataset'], x['iteration'])):
            if r['success']:
                f.write(f"[{r['dataset']:20s}] Iteration {r['iteration']:4d}: SUCCESS ({r['t_total']:.2f}s)\n")
            else:
                f.write(f"[{r['dataset']:20s}] Iteration {r['iteration']:4d}: FAILED - {r.get('error', 'Unknown')[:100]}\n")

    print(f"\n汇总已保存到: {summary_file}")

    # 基于最终验证结果确定返回值
    if all_datasets_complete:
        print("\n" + "="*80)
        print("[SUCCESS] 所有数据集的所有迭代都已成功完成！")
        print("="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("[ERROR] 仍有部分迭代未完成")
        print("详细信息请查看上方的最终验证结果")
        print("="*80)
        return 1


# ========================================
# External Config Interface (for AAA_oneclick_run.py)
# ========================================

def run_from_config(config: dict) -> int:
    """
    从外部配置运行本脚本

    解析TL_settings.jsonc中的:
    - FEM_models: 模型列表
    - simulation_counts: 各模型的仿真数据量

    Args:
        config: 从TL_settings.jsonc加载的配置字典

    Returns:
        0 表示成功，非0 表示失败
    """
    global DATASET_CONFIGS

    print("\n" + "=" * 70)
    print("[AB] 从外部配置运行仿真数据生成")
    print("=" * 70)

    # 1. 解析FEM模型和仿真数据量
    fem_models = config.get('FEM_models', {})
    simulation_counts = config.get('simulation_counts', {})

    if not fem_models:
        print("[错误] 未找到FEM_models配置")
        return 1

    if not simulation_counts:
        print("[错误] 未找到simulation_counts配置")
        return 1

    # 2. 将TL_settings格式转换为原脚本格式
    converted_configs = {}

    for model_name in fem_models.keys():
        count = simulation_counts.get(model_name, 0)
        if count == 0:
            print(f"[跳过] 模型 {model_name} 的仿真数量为0")
            continue

        # 模型名称映射
        name_map = {
            'health': '原始健康结构',
            'first_damage': '首次随机形状损伤结构',
            'damage_repaired': '损伤修补后结构',
            'second_damage': '修补后再次随机形状损伤结构'
        }

        converted_configs[model_name] = {
            'name': name_map.get(model_name, model_name),
            'count': count,
            'cae_template': f'{model_name}.cae',
            'output_dir': rf'C:\SHM_abaqus_data\{model_name}'
        }

    if not converted_configs:
        print("[错误] 没有有效的数据集配置")
        return 1

    DATASET_CONFIGS.clear()
    DATASET_CONFIGS.update(converted_configs)
    print(f"[配置] 加载数据集配置: {list(DATASET_CONFIGS.keys())}")

    # 2. 显示配置摘要
    print("\n[配置摘要]")
    total_count = 0
    for model_name, model_config in DATASET_CONFIGS.items():
        count = model_config.get('count', 0)
        total_count += count
        print(f"  - {model_name}: {count} 个样本")
    print(f"  总计: {total_count} 个样本")

    # 3. 调用原有的main函数逻辑
    print("\n[执行] 开始生成仿真数据...")
    return main()


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
    config_path = os.path.join(SCRIPT_DIR, "TL_settings.jsonc")
    if os.path.exists(config_path):
        print(f"[配置] 从 {config_path} 加载配置")
        config = load_jsonc(config_path)
        sys.exit(run_from_config(config))
    else:
        print(f"[错误] 配置文件不存在: {config_path}")
        print("[提示] 请确保 TL_settings.jsonc 存在，或通过 AAA_oneclick_run.py 运行")
        sys.exit(1)
