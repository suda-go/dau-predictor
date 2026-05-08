import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from calculator import (
    fit_retention, get_retention_curve, calc_lt_fitting, calc_lt_sum,
    predict_dau, generate_new_users_array, extend_retention_curve
)

st.set_page_config(page_title="DAU 预测工具", page_icon="📈", layout="wide")
st.title("DAU 预测工具")
st.markdown("基于留存数据和新增用户预测产品未来 DAU")

EXAMPLE_RETENTION = [0.45, 0.35, 0.28, 0.23, 0.20, 0.18, 0.16,
                     0.14, 0.13, 0.12, 0.11, 0.105, 0.10, 0.095,
                     0.09, 0.087, 0.084, 0.081, 0.078, 0.076,
                     0.074, 0.072, 0.070, 0.068, 0.066, 0.064,
                     0.062, 0.060, 0.058, 0.056]

# --- 侧边栏：参数设置 ---
with st.sidebar:
    st.header("参数设置")

    st.subheader("1. 留存数据")
    use_example = st.checkbox("使用示例数据", value=True)

    if use_example:
        retention_input = "\n".join([str(r) for r in EXAMPLE_RETENTION])
    else:
        retention_input = st.text_area(
            "输入每日留存率（每行一个，从次留开始）",
            placeholder="0.45\n0.35\n0.28\n0.23\n...",
            height=200
        )

    st.subheader("2. LT 计算方式")
    lt_method = st.radio("选择方法", ["幂函数拟合", "分段求和"], index=0)
    lt_max_day = st.number_input("LT 计算截止天数", min_value=7, max_value=365, value=60)

    st.subheader("3. 新增用户")
    new_user_mode = st.radio("新增模式", ["固定日新增", "增长率递增", "自定义输入"])
    daily_new = st.number_input("日新增用户", min_value=1, value=1000, step=100)

    growth_rate = 0.0
    custom_new_users = None
    if new_user_mode == "增长率递增":
        growth_rate = st.number_input("日增长率 (%)", min_value=-10.0, max_value=50.0, value=1.0, step=0.1) / 100.0
    elif new_user_mode == "自定义输入":
        custom_input = st.text_area("输入每日新增（每行一个）", placeholder="1000\n1100\n1200\n...")
        if custom_input.strip():
            custom_new_users = [float(x.strip()) for x in custom_input.strip().split("\n") if x.strip()]

    st.subheader("4. 预测天数")
    predict_days = st.number_input("预测天数", min_value=7, max_value=365, value=90)

# --- 解析留存数据 ---
try:
    retention_rates = [float(x.strip()) for x in retention_input.strip().split("\n") if x.strip()]
    if any(r > 1.0 for r in retention_rates):
        retention_rates = [r / 100.0 for r in retention_rates]
except (ValueError, AttributeError):
    st.error("留存数据格式错误，请检查输入")
    st.stop()

if len(retention_rates) < 2:
    st.error("至少需要2天的留存数据")
    st.stop()

days = np.arange(1, len(retention_rates) + 1)
retention_arr = np.array(retention_rates)

# --- LT 计算 ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("留存曲线")

    if lt_method == "幂函数拟合":
        try:
            a, b = fit_retention(days, retention_arr)
            fitted_curve = get_retention_curve(a, b, int(lt_max_day))
            lt_value = calc_lt_fitting(a, b, lt_max_day)

            fig_ret = go.Figure()
            fig_ret.add_trace(go.Scatter(
                x=days, y=retention_arr,
                mode='markers+lines', name='实际留存',
                marker=dict(size=6)
            ))
            fitted_days = np.arange(1, lt_max_day + 1)
            fig_ret.add_trace(go.Scatter(
                x=fitted_days, y=fitted_curve,
                mode='lines', name=f'拟合: {a:.4f} × n^({b:.4f})',
                line=dict(dash='dash')
            ))
            fig_ret.update_layout(
                xaxis_title="天数", yaxis_title="留存率",
                yaxis_tickformat=".0%", height=400
            )
            st.plotly_chart(fig_ret, use_container_width=True)

            st.info(f"拟合参数: a = {a:.4f}, b = {b:.4f}")
            retention_for_predict = fitted_curve
        except Exception as e:
            st.error(f"拟合失败: {e}")
            st.stop()
    else:
        lt_value = calc_lt_sum(retention_arr)
        retention_for_predict = extend_retention_curve(retention_arr, int(lt_max_day))

        fig_ret = go.Figure()
        fig_ret.add_trace(go.Scatter(
            x=days, y=retention_arr,
            mode='markers+lines', name='实际留存',
            marker=dict(size=6)
        ))
        if len(retention_for_predict) > len(retention_arr):
            ext_days = np.arange(len(retention_arr) + 1, len(retention_for_predict) + 1)
            fig_ret.add_trace(go.Scatter(
                x=ext_days, y=retention_for_predict[len(retention_arr):],
                mode='lines', name='外推',
                line=dict(dash='dot')
            ))
        fig_ret.update_layout(
            xaxis_title="天数", yaxis_title="留存率",
            yaxis_tickformat=".0%", height=400
        )
        st.plotly_chart(fig_ret, use_container_width=True)

with col2:
    st.subheader("关键指标")
    st.metric("LT（用户生命周期）", f"{lt_value:.2f} 天")
    st.metric("次日留存", f"{retention_rates[0]*100:.1f}%")
    if len(retention_rates) >= 7:
        st.metric("7日留存", f"{retention_rates[6]*100:.1f}%")
    if len(retention_rates) >= 30:
        st.metric("30日留存", f"{retention_rates[29]*100:.1f}%")

    if new_user_mode == "固定日新增":
        steady_dau = daily_new * lt_value
        st.metric("稳态 DAU（理论值）", f"{steady_dau:,.0f}")

# --- DAU 预测 ---
st.subheader("DAU 预测")

mode_map = {"固定日新增": "fixed", "增长率递增": "growth", "自定义输入": "custom"}
new_users_array = generate_new_users_array(
    mode=mode_map[new_user_mode],
    predict_days=int(predict_days),
    daily_new=daily_new,
    growth_rate=growth_rate,
    custom_array=custom_new_users
)

dau_curve = predict_dau(new_users_array, retention_for_predict, int(predict_days))

fig_dau = go.Figure()
pred_days = np.arange(1, predict_days + 1)
fig_dau.add_trace(go.Scatter(
    x=pred_days, y=dau_curve,
    mode='lines', name='DAU',
    fill='tozeroy', fillcolor='rgba(66,133,244,0.1)',
    line=dict(color='#4285F4', width=2)
))
fig_dau.add_trace(go.Scatter(
    x=pred_days, y=new_users_array,
    mode='lines', name='日新增',
    line=dict(color='#34A853', width=1, dash='dot')
))
fig_dau.update_layout(
    xaxis_title="天数", yaxis_title="用户数",
    height=400, yaxis_tickformat=","
)
st.plotly_chart(fig_dau, use_container_width=True)

# --- 数据表格 ---
with st.expander("查看详细数据"):
    df = pd.DataFrame({
        "天数": pred_days,
        "DAU": dau_curve.astype(int),
        "日新增": new_users_array.astype(int),
    })
    st.dataframe(df, use_container_width=True, hide_index=True)
