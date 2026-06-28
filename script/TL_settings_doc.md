# TL_settings.jsonc 配置说明

## 配置结构

| 区块 | 用途 |
|------|------|
| `region_by_elem_ID` | 定义局部单元区域，供FEM模型引用 |
| `FEM_models` | 定义各状态CAE模型的结构特性 |
| `simulation_counts` | 指定各模型生成的仿真数据组数 |
| `cases` | 执行场景配置（含消融实验和性能对比参数） |

---

## region_by_elem_ID

定义局部单元区域，格式：`{ "区域名": [单元ID列表] }`

- `region_1`: 条形区域，含测点120344（测点ID 209）
- `region_2`: 圆形区域，含测点72048（测点ID 124）
- `region_3`: 不规则区域

---

## FEM_models

定义各状态模型的结构修改，`local_change` 格式：
- 单区域：`{"local_region": "区域名", "thickness": 值}`
- 多区域：`[{...}, {...}]`
- 无修改：省略该键

| 模型名 | 说明 |
|--------|------|
| `health` | 健康基准状态 |
| `damage_repaired` | region_2厚度改为15.0（修补加厚） |
| `second_damage` | region_2保持修补 + region_1厚度3.0（新损伤） |
| `first_damage` | region_1和region_2厚度均为3.0（初次损伤） |

---

## simulation_counts

指定各模型生成的仿真数据组数。AB脚本支持断点续传和缺失补充。

---

## cases

每个case定义一个完整的迁移学习场景。

### 基本字段

| 字段 | 说明 |
|------|------|
| `name` | 场景名称，用作输出目录名前缀 |
| `description` | 场景描述 |
| `old_baseline` | 预训练数据（旧基线，始终使用原始测点） |
| `new_baseline` | 迁移学习/从头训练数据（新基线） |
| `damage_test` | 损伤检测测试数据 |

### offset（case级别偏移配置）

在case内部配置`offset`字段，实现测点偏移迁移学习场景。

**支持两种配置格式：**

1. **单个偏移方案**：`[2, 6, 9]` - 生成1个场景
2. **批量偏移方案**：`[[0], [1], [2], [3], [1,2], [3,4]]` - 自动展开为6个独立场景

**单个偏移方案配置：**
```jsonc
{
  "name": "Sensor_Offset",
  "offset": [2, 6, 9],  // 单个偏移方案
  "old_baseline": "health",
  "new_baseline": "health",
  "damage_test": "first_damage"
}
```

**批量偏移方案配置：**
```jsonc
{
  "name": "Sensor_Offset",
  "offset": [[0], [1], [2], [3], [1, 2], [3, 4]],  // 批量偏移方案
  "old_baseline": "health",
  "new_baseline": "health",
  "damage_test": "first_damage"
}
```
上述配置会自动展开为6个独立的case：
- `Sensor_Offset_0`
- `Sensor_Offset_1`
- `Sensor_Offset_2`
- `Sensor_Offset_3`
- `Sensor_Offset_1to2`
- `Sensor_Offset_3to4`

**配置了offset后的自动行为：**

| 数据集角色 | 行为 |
|-----------|------|
| `old_baseline` | 始终使用原始非偏移测点数据 |
| `new_baseline` | 自动使用偏移测点数据 |
| `damage_test` | 自动使用偏移测点数据 |

**自动命名规则：**

offset数组会自动转换为命名后缀：
- `[2]` → `offset_2`
- `[2, 3, 4]` → `offset_2to4`（连续）
- `[1, 3, 5]` → `offset_1_3_5`（非连续）

**单个偏移示例：**

配置：
```jsonc
{
  "name": "Sensor_Offset",
  "offset": [2, 6, 9],
  "old_baseline": "health",
  "new_baseline": "health",
  "damage_test": "first_damage"
}
```

自动解析结果：
| 项目 | 值 |
|------|-----|
| 输出目录名 | `Sensor_Offset_2to9` |
| old_baseline数据集 | `health_original_2000` |
| new_baseline数据集 | `health_offset_2to9_2000` |
| damage_test数据集 | `first_damage_offset_2to9_100` |
| 偏移测点文件 | `measures_ID_offset_2to9.csv` |

**批量偏移示例：**

配置：
```jsonc
{
  "name": "Sensor_Offset",
  "offset": [[0], [1], [1, 2]],
  "old_baseline": "health",
  "new_baseline": "health",
  "damage_test": "first_damage"
}
```

自动展开为3个场景：

| 展开场景 | 输出目录名 | new_baseline | damage_test |
|---------|-----------|--------------|-------------|
| 1 | `Sensor_Offset_0` | `health_offset_0_2000` | `first_damage_offset_0_100` |
| 2 | `Sensor_Offset_1` | `health_offset_1_2000` | `first_damage_offset_1_100` |
| 3 | `Sensor_Offset_1to2` | `health_offset_1to2_2000` | `first_damage_offset_1to2_100` |

**与旧配置的兼容性：**

旧的全局`offset`配置仍然兼容，但推荐使用case内部配置：
```jsonc
// 旧方式（兼容但不推荐）
{
  "offset": [2],
  "cases": [
    {
      "new_baseline": "health_offset_2",
      "damage_test": "first_damage_offset_2"
    }
  ]
}

// 新方式（推荐）
{
  "cases": [
    {
      "offset": [2],
      "new_baseline": "health",
      "damage_test": "first_damage"
    }
  ]
}
```

### drift（case级别温漂配置）

在case内部配置`drift`字段，实现测点温漂迁移学习场景。温漂模拟传感器读数漂移，通过数学方法对指定测点的数据施加比例或偏移变换。

**支持两种配置格式：**

1. **单个漂移方案**：`[0]` - 第0个测点漂移，生成1个场景
2. **批量漂移方案**：`[[0], [1], [2], [3]]` - 自动展开为4个独立场景

**温漂参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `drift` | 要漂移的测点行索引列表 | 无（必需） |
| `drift_type` | 漂移类型：`"scale"`（比例）或 `"offset"`（偏移） | `"scale"` |
| `drift_ratio` | 漂移幅度（比例漂移时0.1表示10%） | `0.1` |

**温漂数学模型：**

- **scale（比例漂移）**：`drifted_value = original_value * (1 + drift_ratio)`
  - 模拟传感器增益变化导致的读数偏移
  - `drift_ratio=0.1` 表示所有读数放大10%

- **offset（绝对偏移）**：`drifted_value = original_value + mean(original) * drift_ratio`
  - 模拟传感器零点漂移
  - 偏移量基于该测点原始数据的均值

**配置示例：**

```jsonc
{
  "name": "Sensor_Drift",
  "description": "测点温漂迁移学习场景",
  "drift": [[0], [1], [2], [3]],  // 批量漂移方案
  "drift_type": "scale",          // 比例漂移
  "drift_ratio": 0.1,             // 10%漂移幅度
  "old_baseline": "health",
  "new_baseline": "health",
  "damage_test": "first_damage"
}
```

**配置了drift后的自动行为：**

| 数据集角色 | 行为 |
|-----------|------|
| `old_baseline` | 始终使用原始非漂移数据 |
| `new_baseline` | 自动对指定测点应用漂移 |
| `damage_test` | 自动对指定测点应用漂移 |

**自动命名规则：**

drift数组会自动转换为命名后缀：
- `[0]` → `drift_0`
- `[0, 1, 2]` → `drift_0to2`（连续）
- `[0, 2, 4]` → `drift_0_2_4`（非连续）

**批量漂移示例：**

配置：
```jsonc
{
  "name": "Sensor_Drift",
  "drift": [[0], [1], [2], [3]],
  "drift_type": "scale",
  "drift_ratio": 0.1,
  "old_baseline": "health",
  "new_baseline": "health",
  "damage_test": "first_damage"
}
```

自动展开为4个场景：

| 展开场景 | 输出目录名 | new_baseline | damage_test |
|---------|-----------|--------------|-------------|
| 1 | `Sensor_Drift_0` | `health_drift_0_2000` | `first_damage_drift_0_100` |
| 2 | `Sensor_Drift_1` | `health_drift_1_2000` | `first_damage_drift_1_100` |
| 3 | `Sensor_Drift_2` | `health_drift_2_2000` | `first_damage_drift_2_100` |
| 4 | `Sensor_Drift_3` | `health_drift_3_2000` | `first_damage_drift_3_100` |

**温漂与位置偏移的区别：**

| 特性 | offset（位置偏移） | drift（温漂） |
|------|-------------------|---------------|
| 物理含义 | 测点物理位置改变 | 测点读数系统性偏移 |
| 实现位置 | AC流程（修改测点ID） | AD流程（数学变换） |
| 测点文件 | 生成新的偏移测点CSV | 使用原始测点CSV |
| 数据变化 | 读取不同位置的应力值 | 对同一位置的值施加数学变换 |

### TL_ablation（必需，case级别）

消融实验配置，每个case独立配置。格式：
- 参数值为数组 → 消融变量，遍历实验
- 参数值为标量 → 固定参数

可用参数：
- `freeze_strategy`: `"none"` | `"bottom"` | `"encoder"`
- `trainset_size`: 训练样本数

> 注：所有训练保留数据集后100个样本作为测试集

### TL_performance（必需，case级别）

性能对比配置，使用消融实验确定的最优参数。对比三个模型：
1. **TL模型**：预训练 + 迁移学习微调
2. **OldModel**：仅预训练，不微调
3. **FromScratch**：在新基线数据上从头训练

---

## 脚本对应关系

| 脚本 | 读取配置 |
|------|----------|
| AA脚本 | `region_by_elem_ID`, `FEM_models` |
| AB脚本 | `simulation_counts` |
| AC脚本 | 收集所有case的`offset`配置，生成对应的偏移测点文件 |
| AD脚本 | 根据case的`offset`和`drift`配置解析数据集目录，对drift数据集应用数学漂移变换，命名格式：`{model}_{measures_suffix}_{count}` |
| AE脚本 | `cases`（含`offset`, `drift`, `TL_ablation`, `TL_performance`） |
