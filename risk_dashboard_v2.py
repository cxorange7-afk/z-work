# risk_dashboard_v2.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="债券违约风险智能机器人", layout="wide")
st.title("📊 债券违约风险智能机器人")
st.markdown("输入债券代码，查看该债券的详细风险画像")

# ==================== 加载数据 ====================
@st.cache_data
def load_data():
    pred_dir = "./output_expanding"
    target_file = None
    for f in os.listdir(pred_dir):
        if f.startswith("predictions_20250630") and f.endswith(".csv"):
            target_file = f
            break
    if not target_file:
        files = [f for f in os.listdir(pred_dir) if f.startswith("predictions_") and f.endswith(".csv")]
        if not files:
            st.error("未找到预测结果文件")
            st.stop()
        target_file = sorted(files)[-1]
    df = pd.read_csv(os.path.join(pred_dir, target_file))
    return df

df = load_data()

# 后台模式开关
show_detail = st.sidebar.checkbox("🔍 显示详细概率（后台模式）", value=False)

# ==================== 1. 债券查询区 ====================
st.header("🔎 单支债券风险查询")

# 债券代码输入
bond_code = st.text_input("请输入债券代码（Liscd）", placeholder="例如：112526")

if bond_code:
    mask = df['Liscd'].astype(str).str.strip() == bond_code.strip()
    if mask.any():
        bond = df[mask].iloc[0]
        
        # ---- 风险等级卡片 ----
        risk = bond.get('risk_level', '未知')
        risk_color = {"高": "🔴", "中": "🟡", "低": "🟢"}.get(risk, "⚪")
        col1, col2, col3 = st.columns(3)
        col1.metric("风险等级", f"{risk_color} {risk}")
        col2.metric("最可能违约时段", bond.get('most_likely_period', '未知'))
        top1 = "是" if bond.get('is_top1_percent', 0) == 1 else "否"
        col3.metric("是否 Top 1% 高风险", top1)
        
        # ---- 该债券的动态可视化 ----
        st.subheader(f"📈 债券 {bond_code} 的风险画像")
        
        # 计算区间概率（用累计概率差分）
        y6 = bond['y6m_cum_prob']
        y12 = bond['y12m_cum_prob']
        y18 = bond['y18m_cum_prob']
        y24 = bond['y24m_cum_prob']
        intervals = {
            '0-6月': y6,
            '6-12月': max(0, y12 - y6),
            '12-18月': max(0, y18 - y12),
            '18-24月': max(0, y24 - y18)
        }
        df_interval = pd.DataFrame(list(intervals.items()), columns=['时段', '违约概率'])
        
        # 柱状图：分段违约概率
        fig_bar = px.bar(df_interval, x='时段', y='违约概率', 
                         title=f"债券 {bond_code} 各时段违约概率（分段）",
                         labels={'违约概率': '概率', '时段': '时段'},
                         color='违约概率', color_continuous_scale='Reds')
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # 累计风险曲线
        periods = ['6个月', '12个月', '18个月', '24个月']
        cum_probs = [y6, y12, y18, y24]
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=periods, y=cum_probs, mode='lines+markers',
                                      name='累计违约概率', line=dict(color='red', width=2)))
        fig_line.update_layout(title=f"债券 {bond_code} 的累计违约风险曲线",
                               xaxis_title="预测窗口", yaxis_title="累计违约概率",
                               yaxis=dict(tickformat=".0%"))
        st.plotly_chart(fig_line, use_container_width=True)
        
        # 后台模式：显示具体数值
        if show_detail:
            with st.expander("📋 后台详细数据"):
                st.write(f"**6个月累计概率**: {y6:.4f} ({y6:.2%})")
                st.write(f"**12个月累计概率**: {y12:.4f} ({y12:.2%})")
                st.write(f"**18个月累计概率**: {y18:.4f} ({y18:.2%})")
                st.write(f"**24个月累计概率**: {y24:.4f} ({y24:.2%})")
                if 'max_period_prob' in bond:
                    st.write(f"**最可能时段概率**: {bond['max_period_prob']:.4f} ({bond['max_period_prob']:.2%})")
    else:
        st.error("未找到该债券代码，请检查输入。")
else:
    st.info("👆 在上方输入债券代码，即可查看该债券的风险画像。")

# ==================== 2. 全市场总览（可选参考）====================
st.header("📊 全市场风险总览")
st.caption("以下为所有债券的统计信息，供参考。查询具体债券时请使用上方查询区。")

col1, col2 = st.columns(2)
with col1:
    if 'risk_level' in df.columns:
        risk_counts = df['risk_level'].value_counts().reset_index()
        risk_counts.columns = ['风险等级', '债券数量']
        fig_pie = px.pie(risk_counts, names='风险等级', values='债券数量', 
                         title="全市场风险等级分布", hole=0.3,
                         color='风险等级', color_discrete_map={'高':'red','中':'orange','低':'green'})
        st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    if 'most_likely_period' in df.columns:
        period_counts = df['most_likely_period'].value_counts().reset_index()
        period_counts.columns = ['违约时段', '债券数量']
        fig_bar = px.bar(period_counts, x='违约时段', y='债券数量', title="全市场最可能违约时段分布")
        st.plotly_chart(fig_bar, use_container_width=True)

# 可选：高风险榜单（折叠）
with st.expander("🏆 高风险债券榜单（前20）"):
    if 'risk_level' in df.columns:
        risk_order = {"高": 0, "中": 1, "低": 2}
        df_sorted = df[df['risk_level'].notna()].copy()
        df_sorted['risk_order'] = df_sorted['risk_level'].map(risk_order)
        df_sorted = df_sorted.sort_values('risk_order').head(20)
        st.dataframe(df_sorted[['Liscd', 'risk_level', 'most_likely_period', 'is_top1_percent']],
                     use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("风险等级定义：高（>5%）、中（1%~5%）、低（<1%）。\n模型基于展开窗口第4轮预测结果（数据截止2025-12-31）")