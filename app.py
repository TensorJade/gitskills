from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data import SENSOR_COLUMNS, load_demo_data
from src.pipeline import maintenance_advice, predict_machine_cycle, risk_probability, train_demo_bundle

st.set_page_config(page_title="预测性维护 LSTM Demo", page_icon="🛠️", layout="wide")


@st.cache_resource(show_spinner="正在训练轻量级 LSTM 模型……")
def load_bundle():
    frame, source_name = load_demo_data()
    return train_demo_bundle(frame, source_name=source_name, sequence_length=24, epochs=12)


bundle = load_bundle()
frame = bundle.frame

st.title("制造设备预测性维护 Demo")
st.caption("通过多传感器时序数据持续预测剩余可用寿命（RUL），提前识别潜在故障并生成维护建议。")

with st.sidebar:
    st.header("监测控制")
    machine_ids = sorted(frame["machine_id"].unique().tolist())
    selected_machine = st.selectbox("设备编号", machine_ids, index=0)
    machine_all = frame[frame["machine_id"] == selected_machine].sort_values("cycle")
    min_cycle = bundle.sequence_length
    max_cycle = int(machine_all["cycle"].max())

    if "current_cycle" not in st.session_state:
        st.session_state.current_cycle = min_cycle
    if "auto_play" not in st.session_state:
        st.session_state.auto_play = False
    if st.session_state.current_cycle > max_cycle:
        st.session_state.current_cycle = min_cycle

    current_cycle = st.slider(
        "当前运行周期",
        min_value=min_cycle,
        max_value=max_cycle,
        value=int(st.session_state.current_cycle),
    )
    st.session_state.current_cycle = current_cycle

    col_start, col_step = st.columns(2)
    with col_start:
        if st.button("开始/暂停", use_container_width=True):
            st.session_state.auto_play = not st.session_state.auto_play
    with col_step:
        if st.button("下一周期", use_container_width=True):
            st.session_state.current_cycle = min(max_cycle, st.session_state.current_cycle + 1)
            st.rerun()

    playback_speed = st.select_slider("模拟刷新间隔", options=[0.3, 0.5, 0.8, 1.0], value=0.5)
    st.divider()
    st.write(f"数据源：**{bundle.source_name}**")
    st.write(f"验证 RMSE：**{bundle.training.validation_rmse:.2f} 周期**")
    st.write(f"验证 MAE：**{bundle.training.validation_mae:.2f} 周期**")

machine = machine_all[machine_all["cycle"] <= st.session_state.current_cycle]
latest = machine.iloc[-1]
predicted_rul = predict_machine_cycle(bundle, selected_machine, st.session_state.current_cycle)
risk = risk_probability(predicted_rul)
level, advice = maintenance_advice(predicted_rul)
health_score = float(np.clip(100.0 * predicted_rul / max(max_cycle, 1), 0.0, 100.0))

metric_cols = st.columns(5)
metric_cols[0].metric("设备状态", level)
metric_cols[1].metric("预测剩余寿命", f"{predicted_rul:.1f} 周期")
metric_cols[2].metric("故障风险", f"{risk * 100:.1f}%")
metric_cols[3].metric("健康指数", f"{health_score:.1f}/100")
metric_cols[4].metric("当前周期", f"{int(latest['cycle'])}")

if level == "严重":
    st.error(advice)
elif level == "高风险":
    st.warning(advice)
elif level == "关注":
    st.info(advice)
else:
    st.success(advice)

left, right = st.columns([1.7, 1.0])
with left:
    st.subheader("实时传感器趋势")
    recent = machine.tail(60).melt(
        id_vars=["cycle"], value_vars=SENSOR_COLUMNS, var_name="sensor", value_name="value"
    )
    sensor_cn = {
        "temperature": "温度",
        "vibration": "振动",
        "pressure": "压力",
        "current": "电流",
        "rpm": "转速",
    }
    recent["sensor"] = recent["sensor"].map(sensor_cn)
    fig = px.line(recent, x="cycle", y="value", facet_row="sensor", color="sensor")
    fig.update_layout(height=620, showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
    fig.update_yaxes(matches=None)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("风险判断")
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk * 100,
            number={"suffix": "%"},
            title={"text": "未来故障风险"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 35]},
                    {"range": [35, 70]},
                    {"range": [70, 100]},
                ],
                "threshold": {"line": {"width": 4}, "value": risk * 100},
            },
        )
    )
    gauge.update_layout(height=300, margin=dict(l=30, r=30, t=50, b=20))
    st.plotly_chart(gauge, use_container_width=True)

    st.subheader("当前传感器快照")
    snapshot = pd.DataFrame(
        {
            "指标": ["温度", "振动", "压力", "电流", "转速"],
            "数值": [
                f"{latest['temperature']:.2f}",
                f"{latest['vibration']:.3f}",
                f"{latest['pressure']:.3f}",
                f"{latest['current']:.2f}",
                f"{latest['rpm']:.1f}",
            ],
        }
    )
    st.dataframe(snapshot, hide_index=True, use_container_width=True)

st.subheader("RUL 持续预测轨迹")
prediction_cycles = list(range(bundle.sequence_length, st.session_state.current_cycle + 1))[-80:]
predicted_history = [predict_machine_cycle(bundle, selected_machine, c) for c in prediction_cycles]
actual_history = [float(machine_all.loc[machine_all["cycle"] == c, "rul"].iloc[0]) for c in prediction_cycles]
rul_frame = pd.DataFrame(
    {"cycle": prediction_cycles, "LSTM预测RUL": predicted_history, "实际RUL": actual_history}
).melt(id_vars="cycle", var_name="类型", value_name="RUL")
rul_fig = px.line(rul_frame, x="cycle", y="RUL", color="类型")
rul_fig.update_layout(height=360, margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(rul_fig, use_container_width=True)

with st.expander("系统工作流程"):
    st.markdown(
        """
1. MAT/MES/SCADA 等系统持续采集温度、振动、压力、电流和转速等时序数据。  
2. 滑动时间窗口将最近若干周期的数据送入 LSTM，捕捉长期退化趋势。  
3. 模型持续输出剩余可用寿命（RUL），并转换为故障风险和健康指数。  
4. 当风险超过阈值时，系统生成维护建议，供维修人员提前安排停机窗口和备件。
        """
    )

if st.session_state.auto_play and st.session_state.current_cycle < max_cycle:
    time.sleep(float(playback_speed))
    st.session_state.current_cycle += 1
    st.rerun()
elif st.session_state.current_cycle >= max_cycle:
    st.session_state.auto_play = False
