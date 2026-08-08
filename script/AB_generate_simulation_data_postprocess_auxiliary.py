"""
AB_generate_simulation_data_postprocess_auxiliary.py
=====================================================
Abaqus ODB post-processing auxiliary. Extracts SNEG stress at
integration points and saves as .npy. Called after each FEA job.
"""

from __future__ import annotations

import math
import os
import sys
import time

try:
    from odbAccess import openOdb
except Exception:
    try:
        from OdbAccess import openOdb
    except Exception:
        raise

# 配置参数(从环境变量读取)
import os as _os
POSTPROCESS_DELETE_ODB = _os.environ.get('SHM_DELETE_ODB', '1') == '1'

# 解析参数
if len(sys.argv) < 4:
    print("用法: script -- <odb_path> <output_folder> <iteration_num>")
    sys.exit(2)

args = [a for a in sys.argv[1:] if a != "--"]
odb_path = os.path.abspath(args[-3])
output_folder = os.path.abspath(args[-2])
iteration_num = int(args[-1])

# 辅助函数
def _accept_section(v):
    """只保留 SNEG 截面过滤 (其他可选: "ALL", "SPOS", "SMID")"""
    try:
        sp = getattr(v, "sectionPoint", None)
        if sp is None:
            return True
        desc = str(getattr(sp, "description", "") or str(sp)).upper()
        return ("SNEG" in desc) or ("-1.0" in desc)
    except Exception:
        return True


# 主处理流程
T = {}
_t0_total = time.time()

# 安全检查
if not os.path.exists(odb_path):
    print(f"[post] ODB文件不存在: {odb_path}")
    sys.exit(1)
if not os.path.isdir(output_folder):
    try:
        os.makedirs(output_folder)
    except Exception:
        pass

# 打开 ODB
print(f"[post] 迭代 {iteration_num} 开始后处理 (时间: {time.strftime('%H:%M:%S')})")
_t = time.time()
odb = openOdb(odb_path, readOnly=True)
T["open_odb"] = time.time() - _t
print(f"[post][timing] 打开ODB: {T['open_odb']:.3f}秒")

# 选择最后一帧
step_names = list(odb.steps.keys())
step = odb.steps[step_names[0]]
frame = step.frames[-1]

# 目标元素集 MIDDLEWHOLE
elset = None
try:
    elset = odb.rootAssembly.elementSets['MIDDLEWHOLE']
except Exception:
    try:
        for inst in odb.rootAssembly.instances.values():
            if 'MIDDLEWHOLE' in inst.elementSets:
                elset = inst.elementSets['MIDDLEWHOLE']
                break
    except Exception:
        elset = None

# 应力场
stress = frame.fieldOutputs["S"]

# 提取位置：积分点 (其他可选: CENTROID, ELEMENT_NODAL)
_t = time.time()
try:
    from abaqusConstants import INTEGRATION_POINT
    _pos = INTEGRATION_POINT
except Exception:
    _pos = None

# 提取应力子集
try:
    sub = (
        stress.getSubset(region=elset, position=_pos)
        if (_pos is not None and elset is not None)
        else (
            stress.getSubset(position=_pos)
            if _pos is not None
            else (stress.getSubset(region=elset) if elset is not None else stress)
        )
    )
except Exception:
    sub = stress

try:
    from abaqusConstants import MISES
    mises = sub.getScalarField(invariant=MISES)
except Exception:
    try:
        mises = sub.getScalarField(invariant="Mises")
    except Exception:
        mises = None
T["extract_field"] = time.time() - _t
print(f"[post][timing] 提取应力场: {T['extract_field']:.3f}秒")

# 收集 values
_t = time.time()
try:
    values_iter = list(mises.values) if mises is not None else []
except Exception:
    values_iter = []
if not values_iter:
    try:
        values_iter = list(sub.values)
    except Exception:
        values_iter = []
T["collect_values"] = time.time() - _t
print(f"[post][timing] 收集数值: {T['collect_values']:.3f}秒")

# 直接输出积分点数据（不聚合）
_t = time.time()
rows = []
colnames = ["Element Label", "S-Mises"]

for v in values_iter:
    try:
        if not _accept_section(v):
            continue
        lab = int(v.elementLabel)
        sval = None
        try:
            sval = float(v.data)
        except Exception:
            try:
                sval = float(v.mises)
            except Exception:
                sval = float(v.data.mises)
        if math.isnan(sval) or math.isinf(sval):
            continue
        rows.append([lab, sval])
    except Exception:
        continue

rows_out = rows
T["process_data"] = time.time() - _t
print(f"[post][timing] 数据处理: {T['process_data']:.3f}秒")

# NPY 导出
_t = time.time()
arr = None
try:
    import numpy as _np
    arr = _np.array(rows_out, dtype=float)
    npy_path = os.path.join(output_folder, "iteration.npy")
    _np.save(npy_path, arr)
    T["npy_export"] = time.time() - _t
    print(f"[post][timing] NPY导出: {T['npy_export']:.3f}秒")
except Exception as e:
    print(f"[post][ERROR] NPY导出失败: {e}")
    T["npy_export"] = time.time() - _t

# 导出NPY描述信息
if arr is not None:
    _t = time.time()
    try:
        info_path = os.path.join(output_folder, "iteration_info.txt")
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(f"NPY数据描述信息\n")
            f.write("=" * 60 + "\n")
            f.write(f"迭代编号: {iteration_num}\n")
            f.write(f"配置: SNEG截面, 积分点, 直接输出\n")
            f.write(f"数据形状: {arr.shape}\n")
            f.write(f"总行数: {arr.shape[0]}\n")
            f.write(f"列数: {arr.shape[1]}\n")
            f.write(f"列名: {colnames}\n")
            f.write("\n")

            f.write("前5行数据:\n")
            f.write("-" * 60 + "\n")
            for i in range(min(5, arr.shape[0])):
                f.write(f"行{i+1}: {arr[i].tolist()}\n")
            f.write("\n")

            if arr.shape[0] > 5:
                f.write("后5行数据:\n")
                f.write("-" * 60 + "\n")
                for i in range(max(0, arr.shape[0] - 5), arr.shape[0]):
                    f.write(f"行{i+1}: {arr[i].tolist()}\n")
        T["write_info"] = time.time() - _t
        print(f"[post][timing] Info文件: {T['write_info']:.3f}秒")
    except Exception as e:
        print(f"[post][ERROR] Info文件失败: {e}")
        T["write_info"] = time.time() - _t

# 关闭 ODB
_t = time.time()
try:
    odb.close()
    T["close_odb"] = time.time() - _t
    print(f"[post][timing] 关闭ODB: {T['close_odb']:.3f}秒")
except Exception as e:
    print(f"[post][ERROR] 关闭ODB失败: {e}")
    T["close_odb"] = time.time() - _t

# 删除 ODB（可选）
if POSTPROCESS_DELETE_ODB:
    _t = time.time()
    try:
        os.remove(odb_path)
        T["delete_odb"] = time.time() - _t
        print(f"[post][timing] 删除ODB: {T['delete_odb']:.3f}秒")
    except Exception:
        try:
            time.sleep(0.5)
            os.remove(odb_path)
            T["delete_odb"] = time.time() - _t
            print(f"[post][timing] 删除ODB(retry): {T['delete_odb']:.3f}秒")
        except Exception as e:
            T["delete_odb"] = time.time() - _t
            print(f"[post][ERROR] 删除ODB失败: {e}")
else:
    T["delete_odb"] = 0.0
    print(f"[post] 跳过删除ODB（DELETE_ODB_AFTER_POST=False）")

# 输出总时间
total_time = time.time() - _t0_total
print(f"[post][timing] 后处理总耗时: {total_time:.3f}秒")
print(f"[post] 迭代 {iteration_num} 完成")
