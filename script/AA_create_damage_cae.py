"""
AA_create_damage_cae.py
========================
Batch creation of damage/repair CAE models from element ID lists.
Supports checkpoint resume and automatic retry of failed scenarios.

Usage:
    cd script && python AA_create_damage_cae.py

    # 强制重新生成所有场景
    python AA_create_damage_cae.py --force

    # 自定义输出目录
    python AA_create_damage_cae.py --output C:\custom_output

    # 设置最大重试次数
    python AA_create_damage_cae.py --max-retries 5
"""

import os
import sys
import subprocess
import argparse
import json
from datetime import datetime

# ========================================
# Default Configuration
# ========================================

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(SCRIPT_DIR)
TEMPLATE_CAE = os.path.join(WORKSPACE_DIR, "FEM_model", "model_1214.cae")
DEFAULT_OUTPUT_DIR = r"C:\SHM_abaqus_models"
AUXILIARY_SCRIPT = os.path.join(SCRIPT_DIR, "AA_create_damage_cae_auxiliary.py")

# Abaqus command
ABAQUS_CMD = "abq2025hf1.bat"

# ========================================
# Adjacency Method Configuration (Direct Element ID List)
# ========================================

# ========================================
# 区域定义和模型配置
# ========================================
# 【配置来源说明】
#   - 通过 AAA_oneclick_run.py 调用时：从 TL_settings.jsonc 读取
#     (region_by_elem_ID + FEM_models)
#   - 独立运行本脚本时：使用下方默认配置
#
# ========================================

# 预定义区域（独立运行时的默认配置）
# 通过 run_from_config 调用时会被 TL_settings.jsonc 的 region_by_elem_ID 覆盖
PREDEFINED_REGIONS = {
    # 此配置仅供独立运行使用，实际运行时应通过 TL_settings.jsonc 配置
}

# Global thickness for healthy model (健康模型整体板厚)
GLOBAL_THICKNESS = 10.0  # mm

# 模型配置（独立运行时的默认配置）
# 通过 run_from_config 调用时会被 TL_settings.jsonc 的 FEM_models 覆盖
MODELS = {
    # 此配置仅供独立运行使用，实际运行时应通过 TL_settings.jsonc 配置
}

def check_scenario_completeness(output_dir, scenario_name, min_size_mb=5):
    """
    Check if a scenario CAE and INP files are complete and valid

    Args:
        output_dir: Output directory path
        scenario_name: Scenario name
        min_size_mb: Minimum expected file size in MB

    Returns:
        (is_complete, message): Tuple of completion status and message
    """
    cae_file = os.path.join(output_dir, "%s.cae" % scenario_name)
    inp_file = os.path.join(output_dir, "%s.inp" % scenario_name)

    if not os.path.exists(cae_file):
        return False, "CAE file missing"
    if not os.path.exists(inp_file):
        return False, "INP file missing"

    cae_size_mb = os.path.getsize(cae_file) / 1024.0 / 1024.0
    inp_size_kb = os.path.getsize(inp_file) / 1024.0

    if cae_size_mb < min_size_mb:
        return False, "CAE file too small (%.1f MB < %d MB)" % (cae_size_mb, min_size_mb)
    if inp_size_kb < 10:
        return False, "INP file too small (%.1f KB)" % inp_size_kb

    return True, "OK (CAE: %.1f MB, INP: %.1f KB)" % (cae_size_mb, inp_size_kb)


def scan_output_directory(output_dir, scenarios):
    """
    Scan output directory to determine completion status of all scenarios

    Args:
        output_dir: Output directory path
        scenarios: List of scenario names

    Returns:
        Dictionary mapping scenario names to status info
    """
    status = {}
    for scenario_name in scenarios:
        is_complete, msg = check_scenario_completeness(output_dir, scenario_name)
        status[scenario_name] = {
            'complete': is_complete,
            'message': msg
        }
    return status


def print_status_summary(status):
    """Print a summary of scenario completion status"""
    print("\n" + "=" * 70)
    print("Output Directory Status Check")
    print("=" * 70)

    complete_count = sum(1 for s in status.values() if s['complete'])
    total_count = len(status)

    print("Overall: %d/%d scenarios complete\n" % (complete_count, total_count))

    for idx, (name, info) in enumerate(status.items(), 1):
        status_symbol = "OK" if info['complete'] else "X"
        print("  %d. [%s] %-20s - %s" % (idx, status_symbol, name, info['message']))

    print("=" * 70)


def get_incomplete_scenarios(scenarios, status):
    """
    Get list of incomplete scenarios that need to be generated

    Args:
        scenarios: List of all scenario names
        status: Dictionary of scenario status from scan_output_directory()

    Returns:
        List of incomplete scenario names
    """
    incomplete = []
    for name in scenarios:
        if not status[name]['complete']:
            incomplete.append(name)
    return incomplete


def verify_all_scenarios(output_dir, scenarios):
    """
    Perform full verification of all scenarios

    Args:
        output_dir: Output directory path
        scenarios: List of scenario names

    Returns:
        List of (scenario_name, error_message) tuples for failed scenarios
    """
    status = scan_output_directory(output_dir, scenarios)
    failed = []
    for name, info in status.items():
        if not info['complete']:
            failed.append((name, info['message']))
    return failed


def execute_models(output_dir, models_to_generate):
    """Execute Abaqus to generate damage models"""
    print("\n" + "=" * 70)
    print("Batch Damage Model Creation - Direct Element ID Method")
    print("=" * 70)
    print("Approach: 直接指定单元ID列表作为损伤/修补区域")
    print("Advantages:")
    print("  - 精确控制：直接指定每个单元ID")
    print("  - 灵活配置：每个模型可包含多个不同板厚的区域")
    print("  - 逻辑清晰：健康模型为基准，叠加特定区域的新板厚")
    print("=" * 70)

    # Serialize configuration
    config = {
        'regions': PREDEFINED_REGIONS,
        'models': {name: MODELS[name] for name in models_to_generate if name != 'health'},
        'global_thickness': GLOBAL_THICKNESS,
        'generate_health': 'health' in models_to_generate
    }
    config_json = json.dumps(config, indent=2)

    # Write config to temporary file
    config_file = os.path.join(output_dir, "adjacency_config_temp.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_json)

    print("\nConfiguration saved to: %s" % config_file)

    # Print configuration summary
    print("\nPredefined Regions:")
    for region_name, element_ids in PREDEFINED_REGIONS.items():
        print("  - %s: %d elements" % (region_name, len(element_ids)))

    print("\nModels to Generate:")
    for model_name in models_to_generate:
        if model_name == 'health':
            print("  - health: 健康模型 (全局板厚 %.1fmm)" % GLOBAL_THICKNESS)
        else:
            model_config = MODELS[model_name]
            print("  - %s: %s" % (model_name, model_config['description']))
            for region in model_config['regions']:
                print("    * %s: %.1fmm" % (region['name'], region['thickness']))

    # Build Abaqus command
    cmd = '"%s" cae noGUI="%s" -- adjacency "%s" "%s" "%s"' % (
        ABAQUS_CMD,
        AUXILIARY_SCRIPT,
        TEMPLATE_CAE,
        output_dir,
        config_file
    )

    return run_abaqus_command(cmd, output_dir, config_file, models_to_generate)


def run_abaqus_command(cmd, output_dir, config_file, expected_models):
    """Execute Abaqus command and handle results"""
    print("\nExecuting Abaqus command:")
    print("  %s" % cmd)
    print("")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minutes timeout
            encoding='utf-8',
            errors='ignore',
            cwd=output_dir  # Run Abaqus in output directory to keep temp files there
        )

        # Print output
        if result.stdout:
            print("\n=== Abaqus Output ===")
            try:
                print(result.stdout)
            except UnicodeEncodeError:
                print("(Output contains characters that cannot be displayed)")

        if result.stderr:
            print("\n=== Abaqus Messages ===")
            try:
                print(result.stderr)
            except UnicodeEncodeError:
                print("(Messages contain characters that cannot be displayed)")

        # Check result
        if result.returncode != 0:
            print("\nERROR: Abaqus execution failed (return code: %d)" % result.returncode)
            return 1

        # Clean up config file if provided
        if config_file and os.path.exists(config_file):
            try:
                os.remove(config_file)
                print("\nTemporary config file removed")
            except:
                pass

        print("\n" + "=" * 70)
        print("SUCCESS: All damage models generated!")
        print("=" * 70)
        print("Output directory: %s" % output_dir)
        print("\nGenerated models:")
        for name in expected_models:
            cae_file = os.path.join(output_dir, "%s.cae" % name)
            inp_file = os.path.join(output_dir, "%s.inp" % name)

            cae_exists = os.path.exists(cae_file)
            inp_exists = os.path.exists(inp_file)

            if cae_exists and inp_exists:
                cae_size = os.path.getsize(cae_file) / 1024.0 / 1024.0  # MB
                inp_size = os.path.getsize(inp_file) / 1024.0  # KB
                print("  [OK] %s - CAE: %.1f MB, INP: %.1f KB" % (name, cae_size, inp_size))
            elif cae_exists:
                cae_size = os.path.getsize(cae_file) / 1024.0 / 1024.0
                print("  [!]  %s - CAE: %.1f MB (INP MISSING)" % (name, cae_size))
            else:
                print("  [X]  %s - NOT FOUND" % name)
        print("=" * 70)

        return 0

    except subprocess.TimeoutExpired:
        print("\nERROR: Abaqus execution timeout (exceeded 30 minutes)")
        return 1
    except Exception as e:
        print("\nERROR: Execution failed: %s" % str(e))
        import traceback
        print(traceback.format_exc())
        return 1


# ========================================
# Main Function
# ========================================

def main():
    """Main execution function"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='批量损伤模型创建工具（支持断点续传和查缺补漏）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 基本用法：自动断点续传
  python %(prog)s

  # 强制重新生成所有场景
  python %(prog)s --force

  # 设置最大重试次数为5
  python %(prog)s --max-retries 5

  # 自定义输出目录
  python %(prog)s --output C:\\custom_output

Features:
  - 断点续传：脚本会自动检测已完成的场景，只生成缺失的部分
  - 查缺补漏：生成结束后会进行全盘验证，自动重试失败的场景
  - 智能检测：通过CAE文件大小判断是否完整生成（默认最小5MB）
  - 可靠性高：最多自动重试3次（可通过--max-retries调整）
        """
    )

    parser.add_argument(
        '--output',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help='输出目录 (默认: %s)' % DEFAULT_OUTPUT_DIR
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重新生成所有场景（忽略已存在的文件）'
    )

    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='失败场景的最大重试次数（默认: 3）'
    )

    parser.add_argument(
        '--skip-verify',
        action='store_true',
        help='跳过最终验证步骤（不推荐）'
    )

    args = parser.parse_args()

    # Print header
    print("=" * 70)
    print("Batch Damage Model Creation")
    print("批量损伤模型创建工具")
    print("=" * 70)
    print("Start time: %s" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("Template CAE: %s" % TEMPLATE_CAE)
    print("Output directory: %s" % args.output)
    print("=" * 70)

    # Verify template CAE exists
    if not os.path.exists(TEMPLATE_CAE):
        print("\nERROR: Template CAE not found: %s" % TEMPLATE_CAE)
        return 1

    # Verify auxiliary script exists
    if not os.path.exists(AUXILIARY_SCRIPT):
        print("\nERROR: Auxiliary script not found: %s" % AUXILIARY_SCRIPT)
        return 1

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # All model names (health + damage models)
    all_models = ['health'] + list(MODELS.keys())

    print("\nTotal models configured: %d" % len(all_models))
    print("  1. health: 健康模型 (全局板厚 %.1fmm)" % GLOBAL_THICKNESS)
    for idx, (name, config) in enumerate(MODELS.items(), 2):
        print("  %d. %s: %s" % (idx, name, config['description']))
        print("     Regions: %d" % len(config['regions']))

    # ========================================
    # Step 1: Scan output directory for existing files
    # ========================================
    if not args.force:
        print("\n" + "=" * 70)
        print("STEP 1: Scanning output directory for existing files")
        print("=" * 70)
        status = scan_output_directory(args.output, all_models)
        print_status_summary(status)

        # Get incomplete scenarios
        models_to_generate = get_incomplete_scenarios(all_models, status)

        if not models_to_generate:
            print("\n" + "=" * 70)
            print("All models already complete!")
            print("Use --force to regenerate all models")
            print("=" * 70)
            return 0
        else:
            print("\nModels to generate: %d" % len(models_to_generate))
            for name in models_to_generate:
                print("  - %s" % name)
    else:
        print("\n--force flag detected: Regenerating all models")
        models_to_generate = all_models

    # ========================================
    # Step 2: Generate incomplete/missing models
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 2: Generating models")
    print("=" * 70)
    result = execute_models(args.output, models_to_generate)

    if result != 0:
        print("\nERROR: Initial generation failed")
        return result

    # ========================================
    # Step 3: Verify all models and retry failures
    # ========================================
    if not args.skip_verify:
        print("\n" + "=" * 70)
        print("STEP 3: Verifying all models and retrying failures")
        print("=" * 70)

        retry_count = 0
        max_retries = args.max_retries

        while retry_count < max_retries:
            # Verify all models
            failed = verify_all_scenarios(args.output, all_models)

            if not failed:
                print("\n" + "=" * 70)
                print("SUCCESS: All models verified!")
                print("=" * 70)
                break

            retry_count += 1
            print("\n" + "=" * 70)
            print("RETRY %d/%d: Found %d failed model(s)" % (retry_count, max_retries, len(failed)))
            print("=" * 70)

            for name, msg in failed:
                print("  - %s: %s" % (name, msg))

            if retry_count >= max_retries:
                print("\nERROR: Maximum retries (%d) reached" % max_retries)
                print("Failed models:")
                for name, msg in failed:
                    print("  - %s: %s" % (name, msg))
                return 1

            # Retry failed models
            print("\nRegenerating failed models...")
            retry_models = [name for name, _ in failed]
            result = execute_models(args.output, retry_models)

            if result != 0:
                print("\nWARNING: Retry generation encountered errors")

    # ========================================
    # Step 4: Final status report
    # ========================================
    print("\n" + "=" * 70)
    print("FINAL STATUS REPORT")
    print("=" * 70)
    final_status = scan_output_directory(args.output, all_models)
    print_status_summary(final_status)

    # Check if all complete
    all_complete = all(s['complete'] for s in final_status.values())
    if all_complete:
        print("\n" + "=" * 70)
        print("ALL MODELS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("WARNING: Some models failed to complete")
        print("Please check the output above for details")
        print("=" * 70)
        return 1


# ========================================
# External Config Interface (for AAA_oneclick_run.py)
# ========================================

def run_from_config(config: dict) -> int:
    """
    从外部配置运行本脚本

    解析TL_settings.jsonc中的:
    - region_by_elem_ID: 局部单元区域定义
    - FEM_models: 模型配置（含local_change）

    Args:
        config: 从TL_settings.jsonc加载的配置字典

    Returns:
        0 表示成功，非0 表示失败
    """
    global PREDEFINED_REGIONS, MODELS

    print("\n" + "=" * 70)
    print("[AA] 从外部配置运行模型创建")
    print("=" * 70)

    # 1. 解析区域定义
    region_by_elem_ID = config.get('region_by_elem_ID', {})
    if not region_by_elem_ID:
        print("[错误] 未找到 region_by_elem_ID 配置")
        return 1

    PREDEFINED_REGIONS.clear()
    PREDEFINED_REGIONS.update(region_by_elem_ID)
    print(f"[配置] 加载区域定义: {list(PREDEFINED_REGIONS.keys())}")

    # 2. 解析FEM模型配置
    fem_models = config.get('FEM_models', {})
    if not fem_models:
        print("[错误] 未找到 FEM_models 配置")
        return 1

    # 将TL_settings格式转换为原脚本格式
    converted_models = {}
    for model_name, model_config in fem_models.items():
        if model_name == 'health':
            # 健康模型跳过，由execute_models处理
            continue

        local_change = model_config.get('local_change')
        if local_change is None:
            continue

        # 转换local_change格式
        # TL_settings格式: {"local_region": "region_1", "thickness": 15.0}
        # 或: [{"local_region": "region_1", "thickness": 15.0}, ...]
        # 原脚本格式: {'regions': [{'name': 'region_1', 'thickness': 15.0}]}

        if isinstance(local_change, dict):
            regions_list = [local_change]
        elif isinstance(local_change, list):
            regions_list = local_change
        else:
            print(f"[警告] 模型 {model_name} 的 local_change 格式无效")
            continue

        converted_regions = []
        for change in regions_list:
            converted_regions.append({
                'name': change.get('local_region'),
                'thickness': change.get('thickness')
            })

        converted_models[model_name] = {
            'description': f'模型: {model_name}',
            'regions': converted_regions
        }

    # 覆盖全局MODELS配置
    MODELS.clear()
    MODELS.update(converted_models)
    print(f"[配置] 加载模型定义: {list(MODELS.keys())}")

    # 3. 显示配置摘要
    print("\n[配置摘要]")
    print(f"  区域数: {len(PREDEFINED_REGIONS)}")
    for region_name, elem_ids in PREDEFINED_REGIONS.items():
        print(f"    - {region_name}: {len(elem_ids)} 个单元")

    print(f"  模型数: {len(MODELS) + 1} (含健康模型)")
    print(f"    - health: 健康模型")
    for model_name, model_config in MODELS.items():
        print(f"    - {model_name}: {len(model_config.get('regions', []))} 个区域修改")

    # 4. 调用原有的main函数逻辑
    print("\n[执行] 开始创建模型...")
    return main()


def load_jsonc(filepath):
    """加载JSONC文件（支持//注释和尾随逗号）"""
    import re
    import json
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
    try:
        # 独立运行时，从 TL_settings.jsonc 加载配置
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "TL_settings.jsonc")

        if os.path.exists(config_path):
            print(f"[配置] 从 {config_path} 加载配置")
            config = load_jsonc(config_path)
            start_time = datetime.now()
            exit_code = run_from_config(config)
            end_time = datetime.now()
        else:
            print(f"[错误] 配置文件不存在: {config_path}")
            print("[提示] 请确保 TL_settings.jsonc 存在，或通过 AAA_oneclick_run.py 运行")
            sys.exit(1)

        print("\nEnd time: %s" % end_time.strftime('%Y-%m-%d %H:%M:%S'))
        print("Total execution time: %s" % (end_time - start_time))

        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR: Script execution failed")
        print("=" * 70)
        print("Error message: %s" % str(e))

        import traceback
        print("\nDetailed error:")
        print(traceback.format_exc())
        print("=" * 70)

        sys.exit(1)
