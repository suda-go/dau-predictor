import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from calculator import (
    fit_retention, get_retention_curve, calc_lt_fitting, calc_lt_sum,
    calc_lt_multi, parse_imported_retention, calc_required_retention,
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
    data_source = st.radio("数据来源", ["示例数据", "手动输入", "导入文件"])

    retention_input = ""
    if data_source == "示例数据":
        retention_input = "\n".join([str(r) for r in EXAMPLE_RETENTION])
    elif data_source == "手动输入":
        retention_input = st.text_area(
            "输入每日留存率（每行一个，从次留开始）",
            placeholder="0.45\n0.35\n0.28\n0.23\n...",
            height=200
        )
    else:
        uploaded_file = st.file_uploader(
            "上传留存数据文件",
            type=["csv", "xlsx", "xls"],
            help="支持 CSV/Excel，列名可为: day/天数, retention/留存率；或单列留存率数据"
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_import = pd.read_csv(uploaded_file)
                else:
                    df_import = pd.read_excel(uploaded_file)
                imported_retention = parse_imported_retention(df_import)
                retention_input = "\n".join([str(r) for r in imported_retention])
                st.success(f"导入成功，共 {len(imported_retention)} 天数据")
                with st.expander("预览导入数据"):
                    st.dataframe(df_import.head(10))
            except Exception as e:
                st.error(f"文件解析失败: {e}")

    st.subheader("2. LT 计算方式")
    lt_method = st.radio("选择方法", ["幂函数拟合", "分段求和"], index=0)

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
if not retention_input or not retention_input.strip():
    st.info("请在左侧输入或导入留存数据")
    st.stop()

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

# --- LT 多维度计算 ---
method_key = "fitting" if lt_method == "幂函数拟合" else "sum"
lt_multi = calc_lt_multi(retention_rates, method=method_key)

# --- 留存曲线 & 指标 ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("留存曲线")

    if lt_method == "幂函数拟合":
        try:
            a, b = fit_retention(days, retention_arr)
            fitted_curve = get_retention_curve(a, b, 365)
            retention_for_predict = get_retention_curve(a, b, int(predict_days))

            fig_ret = go.Figure()
            fig_ret.add_trace(go.Scatter(
                x=days, y=retention_arr,
                mode='markers+lines', name='实际留存',
                marker=dict(size=6)
            ))
            fitted_days = np.arange(1, 366)
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
        except Exception as e:
            st.error(f"拟合失败: {e}")
            st.stop()
    else:
        retention_for_predict = extend_retention_curve(retention_arr, int(predict_days))
        extended_full = extend_retention_curve(retention_arr, 365)

        fig_ret = go.Figure()
        fig_ret.add_trace(go.Scatter(
            x=days, y=retention_arr,
            mode='markers+lines', name='实际留存',
            marker=dict(size=6)
        ))
        if len(extended_full) > len(retention_arr):
            ext_days = np.arange(len(retention_arr) + 1, len(extended_full) + 1)
            fig_ret.add_trace(go.Scatter(
                x=ext_days, y=extended_full[len(retention_arr):],
                mode='lines', name='外推',
                line=dict(dash='dot')
            ))
        fig_ret.update_layout(
            xaxis_title="天数", yaxis_title="留存率",
            yaxis_tickformat=".0%", height=400
        )
        st.plotly_chart(fig_ret, use_container_width=True)

with col2:
    st.subheader("LT 多维度指标")

    lt_cols = st.columns(3)
    for i, (key, val) in enumerate(lt_multi.items()):
        with lt_cols[i % 3]:
            st.metric(key, f"{val:.2f} 天")

    st.divider()
    st.subheader("留存关键节点")
    ret_cols = st.columns(3)
    ret_points = [("次留", 0), ("3留", 2), ("7留", 6), ("14留", 13), ("30留", 29), ("60留", 59)]
    shown = 0
    for label, idx in ret_points:
        if idx < len(retention_rates):
            with ret_cols[shown % 3]:
                st.metric(label, f"{retention_rates[idx]*100:.1f}%")
            shown += 1

    st.divider()
    if new_user_mode == "固定日新增":
        st.metric("稳态 DAU（LT365）", f"{daily_new * lt_multi['LT365']:,.0f}")

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

# --- 反推建议留存 ---
st.divider()
st.subheader("目标 DAU 反推建议留存")
st.markdown("输入目标 DAU 和日新增，基于当前留存曲线形状，反推各关键节点需要达到的留存率")

rcol1, rcol2 = st.columns(2)
with rcol1:
    target_dau = st.number_input("目标 DAU", min_value=100, value=50000, step=1000)
    target_daily_new = st.number_input("计划日新增", min_value=1, value=int(daily_new), step=100, key="target_new")

with rcol2:
    result = calc_required_retention(target_dau, target_daily_new, retention_rates, method=method_key)

    st.metric("需要的 LT", f"{result['required_lt']:.2f} 天")
    st.metric("当前 LT（365天）", f"{result['current_lt']:.2f} 天")
    improvement = (result['scale_factor'] - 1) * 100
    if improvement > 0:
        st.metric("留存需提升", f"{improvement:.1f}%", delta=f"↑ 整体需提升 {improvement:.1f}%")
    else:
        st.metric("留存余量", f"{-improvement:.1f}%", delta=f"当前留存已满足目标")

st.markdown("**建议各关键节点留存率：**")
sug = result['suggested_retention']
sug_cols = st.columns(len(sug))
for i, (day_label, rate) in enumerate(sug.items()):
    with sug_cols[i]:
        current_idx = int(day_label.replace("第", "").replace("天", "")) - 1
        current_val = retention_rates[current_idx] if current_idx < len(retention_rates) else None
        delta_str = None
        if current_val is not None:
            diff = (rate - current_val) * 100
            delta_str = f"{diff:+.1f}pp" if diff != 0 else "已达标"
        st.metric(day_label, f"{rate*100:.1f}%", delta=delta_str)
