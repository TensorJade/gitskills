# Predictive Maintenance LSTM Demo

一个面向生产制造场景的预测性维护演示项目。系统基于多传感器时序数据，使用 LSTM 持续预测设备剩余可用寿命（Remaining Useful Life, RUL），并实时输出故障风险、健康指数与维护建议。

## 项目功能

- 多设备温度、振动、压力、电流、转速时序监测
- 基于滑动时间窗口的 LSTM 剩余寿命预测
- 故障风险概率与设备健康指数
- 自动播放模拟实时数据流
- 分级维护建议：正常、关注、高风险、严重
- Streamlit 可视化仪表盘
- 默认合成数据开箱即用，支持 NASA C-MAPSS FD001 数据

## 快速运行

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

首次启动时会自动生成制造设备退化数据并训练轻量级 LSTM 模型。

## 使用 NASA C-MAPSS 数据

从 NASA Prognostics Data Repository 下载 `Turbofan Engine Degradation Simulation Data Set`，将 `train_FD001.txt` 放入：

```text
data/raw/train_FD001.txt
```

重新启动应用后，系统会自动切换到 C-MAPSS FD001 数据。

## 单独训练与测试

```bash
python train.py --epochs 20 --sequence-length 24
python -m pytest -q
```

## 项目结构

```text
app.py                 # Streamlit 实时监测 Demo
train.py               # 独立训练入口
src/data.py            # 合成数据、C-MAPSS 读取、滑动窗口
src/model.py           # PyTorch LSTM 模型
src/pipeline.py        # 训练、推理、风险分级
tests/                 # 单元测试
data/raw/              # 可选真实数据
artifacts/             # 模型与预处理器输出
```

## 工作流程

```text
MAT / MES / SCADA 设备数据
        ↓
数据清洗与标准化
        ↓
滑动时间窗口构造
        ↓
LSTM 时序特征学习
        ↓
RUL 持续预测
        ↓
风险分级与维护决策
```

## 可扩展方向

- 接入真实 MAT/MES/SCADA 接口或消息队列
- 将 RUL 回归升级为未来故障概率分类
- 增加异常检测、根因分析和传感器贡献解释
- 接入维护工单系统，自动生成检修任务
- 使用 Docker、FastAPI 与时序数据库部署

## 声明

默认数据由项目代码合成，仅用于功能演示。NASA C-MAPSS 原始数据不在本仓库重新分发。本项目不应直接用于真实设备安全决策。
