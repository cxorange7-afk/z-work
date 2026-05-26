# risk_dashboard_v2.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="债券违约风险智能机器人", layout="wide")

# 自定义CSS美化
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==================== 数据加载函数 ====================
@st.cache_data
def load_predictions():
    """加载第四 fold 预测结果（2025-12-31 截面）"""
    pred_dir = "./output_expanding"
    target_file = None
    for f in os.listdir(pred_dir):
        if f.startswith("predictions_20250630") and f.endswith(".csv"):
            target_file = f
            break
    if not target_file:
        files = [f for f in os.listdir(pred_dir) if f.startswith("predictions_") and f.endswith(".csv")]
        target_file = sorted(files)[-1] if files else None
    if target_file is None:
        st.error("未找到预测结果文件，请先运行 expanding_window_final.py 生成 predictions_20250630.csv")
        st.stop()
    df = pd.read_csv(os.path.join(pred_dir, target_file))
    return df

@st.cache_data
def load_builtin_metrics():
    """根据用户提供的 fold.docx 构建历史指标 DataFrame"""
    data = []
    # Fold1 测试集指标
    fold1 = [
        ('Fold1', '6m', 0.9895, 0.6918, 0.7912, 0.3750),
        ('Fold1', '12m', 0.9647, 0.7174, 0.9121, 0.3374),
        ('Fold1', '18m', 0.9517, 0.6842, 0.9121, 0.2804),
        ('Fold1', '24m', 0.9423, 0.6868, 0.8837, 0.2331),
    ]
    # Fold2 测试集指标（docx中只有6m和12m，18m/24m暂用None，但图表中会跳过）
    fold2 = [
        ('Fold2', '6m', 0.9709, 0.5838, 0.7937, 0.3802),
        ('Fold2', '12m', 0.9774, 0.6059, 0.8103, 0.3219),
        # 可选：如果后续有18m/24m可补充，没有则留空
    ]
    # Fold3 测试集指标
    fold3 = [
        ('Fold3', '6m', 0.9829, 0.6901, 0.7397, 0.3195),
        ('Fold3', '12m', 0.9848, 0.7247, 0.7808, 0.3024),
        ('Fold3', '18m', 0.9859, 0.7561, 0.8356, 0.3073),
        ('Fold3', '24m', 0.9866, 0.7388, 0.7808, 0.2774),
    ]
    for row in fold1+fold2+fold3:
        data.append(row)
    df = pd.DataFrame(data, columns=['fold', 'horizon', 'auc', 'prauc', 'top1_precision', 'top1_recall'])
    return df

# ==================== 初始化数据 ====================
df_pred = load_predictions()
df_metrics = load_builtin_metrics()

# ==================== 侧边栏导航 ====================
st.sidebar.title("📊 债券风控机器人")
page = st.sidebar.radio(
    "导航菜单",
    ["🏠 首页总览", "📈 模型历史表现", "🔍 债券查询"]
)

# ==================== 页面1：首页总览 ====================
if page == "🏠 首页总览":
    st.title("📊 债券违约风险总览")
    st.markdown("**数据截止日期：2025年12月31日**")

    # 1. 关键统计卡片
    total_bonds = len(df_pred)
    if 'risk_level' in df_pred.columns:
        high_count = (df_pred['risk_level'] == '高').sum()
        high_pct = high_count / total_bonds
        medium_count = (df_pred['risk_level'] == '中').sum()
        medium_pct = medium_count / total_bonds
        low_count = (df_pred['risk_level'] == '低').sum()
        low_pct = low_count / total_bonds
    else:
        # 如果没有风险等级列，基于24m概率自定义
        prob_24 = df_pred['y24m_cum_prob']
        high_count = (prob_24 > 0.05).sum()
        medium_count = ((prob_24 > 0.01) & (prob_24 <= 0.05)).sum()
        low_count = (prob_24 <= 0.01).sum()
        high_pct = high_count / total_bonds
        medium_pct = medium_count / total_bonds
        low_pct = low_count / total_bonds

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📋 截面债券总数", f"{total_bonds:,}")
    with col2:
        st.metric("🔴 高风险债券", f"{high_count} ({high_pct:.1%})")
    with col3:
        st.metric("🟡 中风险债券", f"{medium_count} ({medium_pct:.1%})")
    with col4:
        st.metric("🟢 低风险债券", f"{low_count} ({low_pct:.1%})")

    st.markdown("---")

    # 2. 风险等级分布饼图
    risk_counts = pd.DataFrame({
        '风险等级': ['高', '中', '低'],
        '债券数量': [high_count, medium_count, low_count]
    })
    fig_pie = px.pie(risk_counts, names='风险等级', values='债券数量',
                     color='风险等级',
                     color_discrete_map={'高':'#e74c3c', '中':'#f1c40f', '低':'#2ecc71'},
                     hole=0.3,
                     title="全市场债券风险等级占比")
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

    # 3. 最可能违约时段分布
    if 'most_likely_period' in df_pred.columns:
        st.subheader("最可能违约时段分布")
        period_counts = df_pred['most_likely_period'].value_counts().reset_index()
        period_counts.columns = ['违约时段', '债券数量']
        fig_bar = px.bar(period_counts, x='违约时段', y='债券数量',
                         title="各时段作为最可能违约时段的债券数量",
                         color='债券数量', color_continuous_scale='Blues')
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("预测文件中未包含 most_likely_period 字段，跳过时段分布图")

    # 4. 自然语言摘要
    st.markdown("### 📌 执行摘要")
    with st.expander("点击查看详细解读", expanded=True):
        st.markdown(f"""
        - **截至2025年12月31日，共观测到 {total_bonds} 只债券**。
        - 其中 **高风险债券占比 {high_pct:.1%}**，中风险占比 {medium_pct:.1%}，低风险占比 {low_pct:.1%}。
        - 模型建议重点关注高风险债券名单，并结合财务与市场信息进行动态跟踪。
        """)

# ==================== 页面2：模型历史表现 ====================
elif page == "📈 模型历史表现":
    st.title("📈 模型历史表现评估")
    st.markdown("基于三个展开窗口（Fold1~Fold3）测试集的评估指标，验证模型稳定性。")

    # 展示指标表格（按 fold + horizon 整理）
    st.subheader("各 Fold 测试集指标汇总")
    # 将宽表转为展示表
    display_df = df_metrics.copy()
    display_df = display_df[['fold', 'horizon', 'auc', 'prauc', 'top1_precision', 'top1_recall']]
    display_df.columns = ['Fold', '期限', 'ROC-AUC', 'PR-AUC', 'Top1%精准率', 'Top1%召回率']
    st.dataframe(display_df.style.format({
        'ROC-AUC': '{:.4f}',
        'PR-AUC': '{:.4f}',
        'Top1%精准率': '{:.4f}',
        'Top1%召回率': '{:.4f}'
    }), use_container_width=True)

    # 绘制 ROC-AUC 折线图
    fig_auc = px.line(df_metrics, x='horizon', y='auc', color='fold',
                      markers=True, title="不同 Fold 在各预测期限下的 ROC-AUC")
    st.plotly_chart(fig_auc, use_container_width=True)

    # 绘制 PR-AUC 柱状图分组
    fig_prauc = px.bar(df_metrics, x='horizon', y='prauc', color='fold',
                       barmode='group', title="不同 Fold 的 PR-AUC 对比")
    st.plotly_chart(fig_prauc, use_container_width=True)

    # Top1% 精准率和召回率的组合图
    fig_top1 = go.Figure()
    for fold in df_metrics['fold'].unique():
        sub = df_metrics[df_metrics['fold'] == fold]
        fig_top1.add_trace(go.Scatter(x=sub['horizon'], y=sub['top1_precision'],
                                      mode='lines+markers', name=f'{fold} - Top1%精准率',
                                      line=dict(dash='solid')))
        fig_top1.add_trace(go.Scatter(x=sub['horizon'], y=sub['top1_recall'],
                                      mode='lines+markers', name=f'{fold} - Top1%召回率',
                                      line=dict(dash='dot')))
    fig_top1.update_layout(title="Top 1% 精准率与召回率对比",
                           xaxis_title="预测期限", yaxis_title="比例")
    st.plotly_chart(fig_top1, use_container_width=True)

    st.markdown("""
    **指标解释**：
    - **ROC-AUC**：区分违约与非违约的能力，越接近1越好。
    - **PR-AUC**：对不平衡数据更敏感，反映模型对违约样本的捕捉能力。
    - **Top 1% 精准率**：模型评分最高的1%债券中实际违约的比例。
    - **Top 1% 召回率**：模型捕捉到的违约债券占所有违约债券的比例。
    """)

# ==================== 页面3：债券查询 ====================
elif page == "🔍 债券查询":
    st.title("🔍 单只债券风险查询")
    st.markdown("输入债券代码，查看该债券的详细违约风险评估。")

    # 债券代码选择框
    bond_list = df_pred['Liscd'].astype(str).tolist()
    bond_code = st.selectbox("选择或输入债券代码", bond_list)

    if bond_code:
        bond = df_pred[df_pred['Liscd'].astype(str) == bond_code].iloc[0]

        # 风险等级卡片
        if 'risk_level' in df_pred.columns:
            risk = bond['risk_level']
        else:
            # 如果没有 risk_level 列，基于24m概率计算
            prob_24 = bond['y24m_cum_prob']
            risk = '高' if prob_24 > 0.05 else ('中' if prob_24 > 0.01 else '低')
        risk_color = {"高": "🔴", "中": "🟡", "低": "🟢"}.get(risk, "⚪")
        col1, col2, col3 = st.columns(3)
        col1.metric("风险等级", f"{risk_color} {risk}")
        col2.metric("最可能违约时段", bond.get('most_likely_period', '未知'))
        top1 = "是（前1%高风险）" if bond.get('is_top1_percent', 0) == 1 else "否"
        col3.metric("是否为 Top 1% 高风险", top1)

        # 分段概率柱状图
        st.subheader("各时段违约概率（分段）")
        y6 = bond.get('y6m_cum_prob', 0)
        y12 = bond.get('y12m_cum_prob', 0)
        y18 = bond.get('y18m_cum_prob', 0)
        y24 = bond.get('y24m_cum_prob', 0)
        intervals = {
            '0-6月': y6,
            '6-12月': max(0, y12 - y6),
            '12-18月': max(0, y18 - y12),
            '18-24月': max(0, y24 - y18)
        }
        df_interval = pd.DataFrame(list(intervals.items()), columns=['时段', '违约概率'])
        fig_bar = px.bar(df_interval, x='时段', y='违约概率',
                         title=f"债券 {bond_code} 各时段违约概率",
                         labels={'违约概率': '概率'}, color='违约概率', color_continuous_scale='Reds')
        st.plotly_chart(fig_bar, use_container_width=True)

        # 累计风险曲线
        st.subheader("累计违约风险曲线")
        periods = ['6个月', '12个月', '18个月', '24个月']
        cum_probs = [y6, y12, y18, y24]
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=periods, y=cum_probs, mode='lines+markers',
                                      name='累计违约概率', line=dict(color='red', width=2)))
        fig_line.update_layout(xaxis_title="预测窗口", yaxis_title="累计违约概率",
                               yaxis=dict(tickformat=".0%"))
        st.plotly_chart(fig_line, use_container_width=True)

        # 后台详细数据（折叠）
        with st.expander("📋 后台详细概率数值（供参考）"):
            st.write(f"6个月累计概率：{y6:.4f} ({y6:.2%})")
            st.write(f"12个月累计概率：{y12:.4f} ({y12:.2%})")
            st.write(f"18个月累计概率：{y18:.4f} ({y18:.2%})")
            st.write(f"24个月累计概率：{y24:.4f} ({y24:.2%})")
            if 'max_period_prob' in bond:
                st.write(f"最可能时段发生概率：{bond['max_period_prob']:.4f} ({bond['max_period_prob']:.2%})")

# ==================== 页脚 ====================
st.sidebar.markdown("---")
st.sidebar.caption("模型基于分段条件概率（hazard）\n数据截止2025-12-31")
