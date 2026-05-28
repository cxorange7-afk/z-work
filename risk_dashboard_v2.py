# -*- coding: utf-8 -*-
"""
债券违约预测可视化前端（Streamlit · v4 Final）
====================================
本文件可直接作为 app.py 部署，用于展示债券违约预测结果、历史测试表现、
高风险债券排行、单券风险查询和数据运行状态。

推荐目录：
    app.py
    requirements.txt
    output_expanding/
        predictions_20251231.csv
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# 0. 页面基础配置
# =============================================================================

APP_TITLE = "债券违约预测"
APP_SUBTITLE = "BOND DEFAULT · PREDICTION"
MODEL_NAME = "Expanding Window XGBoost Hazard"
DEFAULT_OUTPUT_DIR = "output_expanding"

HORIZON_LABELS = {
    "y6m": "未来6个月",
    "y12m": "未来12个月",
    "y18m": "未来18个月",
    "y24m": "未来24个月",
}
HORIZON_ORDER = ["y6m", "y12m", "y18m", "y24m"]
HORIZON_SHORT = {"y6m": "未来6个月", "y12m": "未来12个月", "y18m": "未来18个月", "y24m": "未来24个月"}

RISK_BUCKET_ORDER = ["Top 1%", "1–5%", "5–10%", "10–20%", "Other"]

# st.set_page_config 必须是第一个 Streamlit 命令；要放在 @st.cache_data 等装饰器之前。
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)
BUCKET_BADGE_CLASS = {
    "Top 1%": "bucket-top1",
    "1–5%": "bucket-top5",
    "5–10%": "bucket-top10",
    "10–20%": "bucket-top20",
    "Other": "bucket-other",
}

# =============================================================================
# 1. 前三个 Fold 测试集指标
# =============================================================================

FOLD_TEST_METRICS: List[Dict[str, object]] = [
    # Fold1 test
    {"fold": "Fold1", "test_year": "Test set 1", "horizon": "6m", "n_samples": 9118, "n_positive": 192, "base_rate": 0.0211, "roc_auc": 0.9895, "pr_auc": 0.6918, "brier": 0.0425, "log_loss": 0.2231, "top1_precision": 0.7912, "top1_recall": 0.3750, "top5_precision": 0.3978, "top5_recall": 0.9427, "top10_precision": 0.2053, "top10_recall": 0.9740, "note": ""},
    {"fold": "Fold1", "test_year": "Test set 1", "horizon": "12m", "n_samples": 9118, "n_positive": 246, "base_rate": 0.0270, "roc_auc": 0.9647, "pr_auc": 0.7174, "brier": 0.0625, "log_loss": 0.2822, "top1_precision": 0.9121, "top1_recall": 0.3374, "top5_precision": 0.4681, "top5_recall": 0.8659, "top10_precision": 0.2492, "top10_recall": 0.9228, "note": ""},
    {"fold": "Fold1", "test_year": "Test set 1", "horizon": "18m", "n_samples": 9118, "n_positive": 296, "base_rate": 0.0325, "roc_auc": 0.9517, "pr_auc": 0.6842, "brier": 0.0670, "log_loss": 0.2939, "top1_precision": 0.9121, "top1_recall": 0.2804, "top5_precision": 0.5033, "top5_recall": 0.7736, "top10_precision": 0.2733, "top10_recall": 0.8412, "note": ""},
    {"fold": "Fold1", "test_year": "Test set 1", "horizon": "24m", "n_samples": 4312, "n_positive": 163, "base_rate": 0.0378, "roc_auc": 0.9423, "pr_auc": 0.6868, "brier": 0.0930, "log_loss": 0.3601, "top1_precision": 0.8837, "top1_recall": 0.2331, "top5_precision": 0.5674, "top5_recall": 0.7485, "top10_precision": 0.3202, "top10_recall": 0.8466, "note": ""},
    # Fold2 test
    {"fold": "Fold2", "test_year": "Test set 2", "horizon": "6m", "n_samples": 12649, "n_positive": 263, "base_rate": 0.0208, "roc_auc": 0.9709, "pr_auc": 0.5838, "brier": 0.0133, "log_loss": 0.0743, "top1_precision": 0.7937, "top1_recall": 0.3802, "top5_precision": 0.2864, "top5_recall": 0.6882, "top10_precision": 0.1946, "top10_recall": 0.9354, "note": ""},
    {"fold": "Fold2", "test_year": "Test set 2", "horizon": "12m", "n_samples": 5827, "n_positive": 146, "base_rate": 0.0251, "roc_auc": 0.9774, "pr_auc": 0.6059, "brier": 0.0159, "log_loss": 0.0868, "top1_precision": 0.8103, "top1_recall": 0.3219, "top5_precision": 0.3505, "top5_recall": 0.6986, "top10_precision": 0.2388, "top10_recall": 0.9521, "note": "该测试窗口仅展示已获得的期限指标。"},
    # Fold3 test
    {"fold": "Fold3", "test_year": "Test set 3", "horizon": "6m", "n_samples": 14693, "n_positive": 338, "base_rate": 0.0230, "roc_auc": 0.9829, "pr_auc": 0.6901, "brier": 0.0854, "log_loss": 0.3428, "top1_precision": 0.7397, "top1_recall": 0.3195, "top5_precision": 0.4237, "top5_recall": 0.9201, "top10_precision": 0.2233, "top10_recall": 0.9704, "note": "测试窗口统计口径"},
    {"fold": "Fold3", "test_year": "Test set 3", "horizon": "12m", "n_samples": 14693, "n_positive": 377, "base_rate": 0.0257, "roc_auc": 0.9848, "pr_auc": 0.7247, "brier": 0.0853, "log_loss": 0.3426, "top1_precision": 0.7808, "top1_recall": 0.3024, "top5_precision": 0.4755, "top5_recall": 0.9257, "top10_precision": 0.2498, "top10_recall": 0.9735, "note": "测试窗口统计口径"},
    {"fold": "Fold3", "test_year": "Test set 3", "horizon": "18m", "n_samples": 14693, "n_positive": 397, "base_rate": 0.0270, "roc_auc": 0.9859, "pr_auc": 0.7561, "brier": 0.0851, "log_loss": 0.3421, "top1_precision": 0.8356, "top1_recall": 0.3073, "top5_precision": 0.5000, "top5_recall": 0.9244, "top10_precision": 0.2634, "top10_recall": 0.9748, "note": "测试窗口统计口径"},
    {"fold": "Fold3", "test_year": "Test set 3", "horizon": "24m", "n_samples": 14693, "n_positive": 411, "base_rate": 0.0280, "roc_auc": 0.9866, "pr_auc": 0.7388, "brier": 0.0868, "log_loss": 0.3460, "top1_precision": 0.7808, "top1_recall": 0.2774, "top5_precision": 0.5204, "top5_recall": 0.9294, "top10_precision": 0.2737, "top10_recall": 0.9781, "note": "测试窗口统计口径"},
]

# =============================================================================
# 2. CSS：暗色仪表盘风格 + 不使用按钮式导航
# =============================================================================

def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #080d18;
            --panel: #0f1727;
            --panel2: #121d31;
            --border: rgba(148, 163, 184, 0.16);
            --text: #e5edf7;
            --muted: #92a3b8;
            --accent: #23d3c3;
            --accent2: #7c3aed;
            --warning: #f59e0b;
            --danger: #ef4444;
        }
        .stApp { background: radial-gradient(circle at 12% 8%, #12213b 0, #080d18 28%, #060914 100%); color: var(--text); }
        header[data-testid="stHeader"] { display: none !important; }
        div[data-testid="stToolbar"] { visibility: hidden !important; height: 0 !important; position: fixed !important; }
        #MainMenu { visibility: hidden !important; }
        footer { visibility: hidden !important; }
        div[data-testid="stDecoration"] { display: none !important; }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #08111f 0%, #0d1525 100%);
            border-right: 1px solid var(--border);
        }
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span { color: var(--text) !important; }
        .block-container { padding-top: .35rem !important; max-width: 1440px; }
        h1, h2, h3 { letter-spacing: -0.02em; color: var(--text); }
        .hero {
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 22px 24px;
            background: linear-gradient(135deg, rgba(35, 211, 195, .10), rgba(124, 58, 237, .09) 52%, rgba(15, 23, 42, .85));
            box-shadow: 0 18px 55px rgba(0,0,0,.28);
            margin-bottom: 18px;
        }
        .eyebrow { color: var(--accent); font-size: 0.78rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
        .hero-title { color: var(--text); font-size: 2.05rem; font-weight: 800; line-height: 1.12; margin-top: 6px; }
        .hero-sub { color: var(--muted); font-size: .96rem; margin-top: 8px; max-width: 900px; }
        .card {
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 18px 18px;
            background: rgba(15, 23, 42, .82);
            box-shadow: 0 12px 32px rgba(0,0,0,.22);
        }
        .metric-card {
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 14px 16px;
            background: rgba(16, 26, 46, .86);
            min-height: 104px;
        }
        .metric-label { color: var(--muted); font-size: .78rem; margin-bottom: 8px; }
        .metric-value { color: var(--text); font-size: 1.75rem; font-weight: 800; font-variant-numeric: tabular-nums; }
        .metric-help { color: var(--muted); font-size: .76rem; margin-top: 6px; }
        .notice {
            border: 1px solid rgba(35, 211, 195, .22);
            background: rgba(35, 211, 195, .08);
            color: #c7fff8;
            padding: 11px 14px;
            border-radius: 14px;
            font-size: .86rem;
            margin: 10px 0 16px 0;
        }
        .small-muted { color: var(--muted); font-size: .82rem; }
        .badge {
            display: inline-block; padding: 3px 9px; border-radius: 999px;
            font-size: .76rem; font-weight: 700; margin: 2px 4px 2px 0;
            border: 1px solid rgba(255,255,255,.10);
        }
        .badge-green { background: rgba(35, 211, 195, .14); color: #55f0de; }
        .badge-purple { background: rgba(124, 58, 237, .16); color: #c4b5fd; }
        .bucket-top1 { background: rgba(239, 68, 68, .20); color: #fecaca; }
        .bucket-top5 { background: rgba(249, 115, 22, .20); color: #fed7aa; }
        .bucket-top10 { background: rgba(245, 158, 11, .18); color: #fde68a; }
        .bucket-top20 { background: rgba(34, 197, 94, .15); color: #bbf7d0; }
        .bucket-other { background: rgba(148, 163, 184, .12); color: #cbd5e1; }
        div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }
        div[data-testid="stTabs"] button { color: #cbd5e1; }
        div[data-testid="stTabs"] button[aria-selected="true"] { color: #55f0de; border-bottom-color: #23d3c3; }
        .stDownloadButton button, .stButton button {
            border-radius: 12px !important; border: 1px solid rgba(35, 211, 195, .35) !important;
            background: rgba(35, 211, 195, .09) !important; color: #dffdfa !important;
        }
        .cover-wrap {
            min-height: calc(100vh - 28px);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 24px 0 42px 0;
        }
        .cover-card {
            width: min(1120px, 96vw);
            border: 1px solid rgba(35, 211, 195, .22);
            border-radius: 30px;
            padding: 46px 48px;
            background:
                radial-gradient(circle at 82% 18%, rgba(124,58,237,.28), transparent 28%),
                radial-gradient(circle at 16% 20%, rgba(35,211,195,.20), transparent 24%),
                linear-gradient(135deg, rgba(9,16,31,.95), rgba(13,21,37,.88));
            box-shadow: 0 28px 90px rgba(0,0,0,.42);
        }
        .cover-title { font-size: clamp(2.2rem, 5vw, 4.1rem); line-height: 1.05; font-weight: 900; color: #f8fbff; margin: 12px 0; letter-spacing: -0.06em; }
        .cover-sub { color: #aab9cd; font-size: 1.02rem; max-width: 760px; line-height: 1.75; }
        .cover-grid { display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-top: 30px; }
        .cover-feature { border:1px solid rgba(148,163,184,.16); border-radius:18px; padding:18px; background:rgba(15,23,42,.66); }
        .cover-feature .icon { font-size:1.7rem; margin-bottom:8px; }
        .cover-feature b { color:#e5edf7; }
        .cover-feature p { color:#93a4ba; font-size:.82rem; margin:6px 0 0 0; line-height:1.55; }
        .risk-card {
            border: 1px solid rgba(148,163,184,.16);
            border-radius: 18px;
            padding: 18px 18px;
            background: rgba(15,23,42,.82);
            min-height: 128px;
        }
        .risk-card-label { color: var(--muted); font-size: .78rem; margin-bottom: 8px; }
        .risk-card-value { color: var(--text); font-size: 1.75rem; font-weight: 850; }
        .risk-card-help { color: var(--muted); font-size: .76rem; margin-top: 8px; }

        .feature-grid { display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 14px 0 20px 0; }
        .feature-card {
            border:1px solid rgba(148,163,184,.15); border-radius:18px; padding:16px 16px;
            background: linear-gradient(180deg, rgba(19,31,52,.86), rgba(12,19,34,.80));
            min-height: 132px; box-shadow: 0 10px 28px rgba(0,0,0,.18);
        }
        .feature-icon { width:38px; height:38px; display:flex; align-items:center; justify-content:center; border-radius:13px;
            background: rgba(35,211,195,.10); border:1px solid rgba(35,211,195,.22); font-size:1.18rem; margin-bottom:10px; }
        .feature-title { color:#f1f5f9; font-weight:800; font-size:.98rem; margin-bottom:5px; }
        .feature-body { color:#9fb0c6; font-size:.82rem; line-height:1.62; }
        .panel-title { display:flex; align-items:center; gap:8px; color:#eef6ff; font-size:1.05rem; font-weight:820; margin: 4px 0 12px 0; }
        .pill-row { display:flex; flex-wrap:wrap; gap:8px; margin: 10px 0 2px 0; }
        .soft-pill { border:1px solid rgba(148,163,184,.16); border-radius:999px; padding:6px 10px; color:#cbd5e1; background:rgba(15,23,42,.55); font-size:.78rem; }
        .summary-box {
            border: 1px solid rgba(35,211,195,.20); border-radius:18px; padding:16px 18px;
            background: linear-gradient(135deg, rgba(35,211,195,.08), rgba(124,58,237,.07));
            color:#dbeafe; line-height:1.75; margin: 14px 0 18px 0;
        }
        .summary-box b { color:#ffffff; }
        .glossary-grid { display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 12px; }
        .glossary-card { border:1px solid rgba(148,163,184,.14); border-radius:16px; padding:14px; background:rgba(15,23,42,.70); }
        .glossary-card b { color:#55f0de; }
        .glossary-card p { color:#9fb0c6; font-size:.82rem; line-height:1.62; margin:6px 0 0 0; }
        .rank-band { height:8px; border-radius:999px; overflow:hidden; display:flex; margin-top:10px; border:1px solid rgba(255,255,255,.08); }
        .rank-band span:nth-child(1){background:#ef4444;width:1%}.rank-band span:nth-child(2){background:#f97316;width:4%}.rank-band span:nth-child(3){background:#f59e0b;width:5%}.rank-band span:nth-child(4){background:#22c55e;width:10%}.rank-band span:nth-child(5){background:#64748b;width:80%}
        @media (max-width: 900px) { .cover-grid, .feature-grid, .glossary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } .cover-card { padding: 32px 24px; } }
        </style>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# 3. 工具函数
# =============================================================================

def pct(x: Optional[float], digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x) * 100:.{digits}f}%"


def fmt_num(x: Optional[float], digits: int = 4) -> str:
    if x is None or pd.isna(x):
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    return f"{float(x):.{digits}f}"


def risk_bucket_from_rank_pct(rank_pct: float) -> str:
    if rank_pct <= 0.01:
        return "Top 1%"
    if rank_pct <= 0.05:
        return "1–5%"
    if rank_pct <= 0.10:
        return "5–10%"
    if rank_pct <= 0.20:
        return "10–20%"
    return "Other"


def display_bucket(bucket: str) -> str:
    klass = BUCKET_BADGE_CLASS.get(bucket, "bucket-other")
    return f'<span class="badge {klass}">{bucket}</span>'


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_box(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="card" style="margin-top:10px; padding:14px 16px; background:rgba(20,30,48,.72);">
            <div style="font-size:.82rem; color:#55f0de; font-weight:700; margin-bottom:6px;">📌 {title}</div>
            <div style="font-size:.88rem; color:#d8e3f0; line-height:1.72;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_card(icon: str, title: str, body: str) -> str:
    return (
        f'<div class="feature-card">'
        f'<div class="feature-icon">{icon}</div>'
        f'<div class="feature-title">{title}</div>'
        f'<div class="feature-body">{body}</div>'
        f'</div>'
    )


def render_feature_grid(cards: List[Tuple[str, str, str]]) -> None:
    html = "<div class='feature-grid'>" + "".join(feature_card(*card) for card in cards) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def probability_sentence(row: pd.Series) -> str:
    y6 = float(row.get("y6m_prob_display", 0.0))
    y12 = float(row.get("y12m_prob_display", 0.0))
    y24 = float(row.get("y24m_prob_display", 0.0))
    near_ratio = y6 / max(y24, 1e-9)
    if y24 < 0.03:
        return "该债券整体违约概率较低，当前更适合作为常规监测对象。"
    if near_ratio >= 0.55:
        return "该债券风险更多集中在短期窗口，建议优先关注近期偿债压力、价格异动和负面舆情。"
    if y24 - y12 >= 0.08:
        return "该债券风险主要在中长期累积，短期压力相对有限，但需要持续跟踪基本面变化。"
    return "该债券风险随期限逐步上升，整体风险释放较为连续，适合纳入滚动观察名单。"


def rank_position_sentence(rank_pct: float, bucket: str) -> str:
    if bucket == "Top 1%":
        return "处于全样本最靠前的 1% 风险区间，是当前预警名单中的核心关注对象。"
    if bucket == "1–5%":
        return "处于全样本前 1%–5% 风险区间，建议作为重点跟踪债券。"
    if bucket == "5–10%":
        return "处于全样本前 5%–10% 风险区间，风险相对靠前，需要定期复核。"
    if bucket == "10–20%":
        return "处于全样本前 10%–20% 风险区间，属于次重点观察范围。"
    return "未进入前 20% 高风险区间，当前相对风险排序不靠前。"


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">{APP_SUBTITLE}</div>
          <div class="hero-title">{title}</div>
          <div class="hero-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbeafe"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=24, r=18, t=46, b=28),
    )
    fig.update_xaxes(gridcolor="rgba(148, 163, 184, .12)", zerolinecolor="rgba(148, 163, 184, .12)")
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, .12)", zerolinecolor="rgba(148, 163, 184, .12)")
    return fig


def safe_read_csv(path: str | Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def latest_file(patterns: Iterable[str]) -> Optional[str]:
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    if not files:
        return None
    files = sorted(files, key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

# =============================================================================
# 4. 数据读取与标准化
# =============================================================================

@st.cache_data(show_spinner=False)
def default_fold_metrics() -> pd.DataFrame:
    df = pd.DataFrame(FOLD_TEST_METRICS)
    df["horizon_order"] = df["horizon"].map({"6m": 1, "12m": 2, "18m": 3, "24m": 4})
    return df.sort_values(["fold", "horizon_order"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def make_demo_predictions(n: int = 360, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    codes = [f"BOND{100000 + i}" for i in range(n)]
    issuers = [f"发行人{rng.integers(1, 80):03d}" for _ in range(n)]

    # 稀有高风险 + 大量低风险，更像违约预测分布
    base = np.clip(rng.beta(0.65, 18, n) + rng.choice([0, 0.05, 0.12], n, p=[0.88, 0.09, 0.03]), 0, 0.75)
    y6 = np.clip(base * rng.uniform(0.25, 0.55, n), 0, 1)
    y12 = np.clip(np.maximum(y6, base * rng.uniform(0.45, 0.78, n)), 0, 1)
    y18 = np.clip(np.maximum(y12, base * rng.uniform(0.70, 0.95, n)), 0, 1)
    y24 = np.clip(np.maximum(y18, base), 0, 1)

    df = pd.DataFrame(
        {
            "Liscd": codes,
            "BondCode": codes,
            "BondName": [f"示例债券{i:03d}" for i in range(n)],
            "Issuer": issuers,
            "PeriodEnd": "2025-12-31",
            "sem_str": "2025H2",
            "y6m_cum_prob": y6,
            "y12m_cum_prob": y12,
            "y18m_cum_prob": y18,
            "y24m_cum_prob": y24,
        }
    )
    return normalize_predictions(df, source="demo")


def read_prediction_file(uploaded_file=None, explicit_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """读取预测结果。优先级：上传文件 > 显式路径 > output_expanding/latest csv > parquet > demo。"""
    status: Dict[str, object] = {
        "mode": "demo",
        "source_path": None,
        "message": "未找到预测结果文件，已使用演示数据。",
    }

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        status.update({"mode": "uploaded", "source_path": getattr(uploaded_file, "name", "uploaded.csv"), "message": "已读取侧边栏上传的预测 CSV。"})
        return normalize_predictions(df, source="uploaded"), status

    if explicit_path and Path(explicit_path).exists():
        path = str(explicit_path)
        if path.lower().endswith(".csv"):
            df = safe_read_csv(path)
        elif path.lower().endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            raise ValueError("仅支持 csv 或 parquet 文件。")
        status.update({"mode": "file", "source_path": path, "message": f"已读取 {path}"})
        return normalize_predictions(df, source=path), status

    csv_path = latest_file(
        [
            f"{DEFAULT_OUTPUT_DIR}/predictions_*.csv",
            "predictions_*.csv",
            "**/predictions_*.csv",
        ]
    )
    if csv_path:
        df = safe_read_csv(csv_path)
        status.update({"mode": "file", "source_path": csv_path, "message": f"已读取 {csv_path}"})
        return normalize_predictions(df, source=csv_path), status

    parquet_path = latest_file(
        [
            "**/inference_predictions_calibrated.parquet",
            "**/production_predictions_long.parquet",
        ]
    )
    if parquet_path:
        try:
            df = pd.read_parquet(parquet_path)
            status.update({"mode": "file", "source_path": parquet_path, "message": f"已读取 {parquet_path}"})
            return normalize_predictions(df, source=parquet_path), status
        except Exception as exc:  # pragma: no cover - 仅部署环境中使用
            status["message"] = f"发现 parquet 但读取失败：{exc}，已使用演示数据。"

    return make_demo_predictions(), status


def normalize_predictions(df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """把建模脚本输出、生产 parquet 或手工 CSV 统一成前端需要的列。"""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    # ID / 名称字段兼容
    if "Liscd" not in out.columns:
        for alt in ["BondCode", "bond_code", "code", "债券代码"]:
            if alt in out.columns:
                out["Liscd"] = out[alt]
                break
    if "Liscd" not in out.columns:
        out["Liscd"] = [f"ROW{i:06d}" for i in range(len(out))]

    if "BondCode" not in out.columns:
        out["BondCode"] = out["Liscd"].astype(str)
    if "BondName" not in out.columns:
        out["BondName"] = out.get("债券名称", out["BondCode"].astype(str))
    if "Issuer" not in out.columns:
        out["Issuer"] = out.get("发行人", "未知发行人")
    if "PeriodEnd" not in out.columns:
        out["PeriodEnd"] = out.get("inference_date", out.get("sem_str", "未知"))

    # 概率字段兼容：优先使用生产版 raw/cal，否则用 expanding_window 的 *_cum_prob。
    col_candidates = {
        "y6m": ["y6m_prob_cal", "y6m_prob", "y6m_prob_raw", "y6m_cum_prob", "未来6月", "未来6个月"],
        "y12m": ["y12m_prob_cal", "y12m_prob", "y12m_prob_raw", "y12m_cum_prob", "未来12月", "未来1年"],
        "y18m": ["y18m_prob_cal", "y18m_prob", "y18m_prob_raw", "y18m_cum_prob", "未来18月"],
        "y24m": ["y24m_prob_cal", "y24m_prob", "y24m_prob_raw", "y24m_cum_prob", "未来24月", "未来2年"],
    }
    raw_candidates = {
        "y6m": ["y6m_prob_raw", "y6m_prob", "y6m_cum_prob"],
        "y12m": ["y12m_prob_raw", "y12m_prob", "y12m_cum_prob"],
        "y18m": ["y18m_prob_raw", "y18m_prob", "y18m_cum_prob"],
        "y24m": ["y24m_prob_raw", "y24m_prob", "y24m_cum_prob"],
    }

    for h in HORIZON_ORDER:
        cal_col = next((c for c in col_candidates[h] if c in out.columns), None)
        raw_col = next((c for c in raw_candidates[h] if c in out.columns), cal_col)
        if cal_col is None:
            out[f"{h}_prob_display"] = np.nan
        else:
            out[f"{h}_prob_display"] = pd.to_numeric(out[cal_col], errors="coerce")
        if raw_col is None:
            out[f"{h}_prob_rank"] = out[f"{h}_prob_display"]
        else:
            out[f"{h}_prob_rank"] = pd.to_numeric(out[raw_col], errors="coerce")

    # 兜底：如果概率全空，填 0，避免页面崩溃。
    for h in HORIZON_ORDER:
        out[f"{h}_prob_display"] = out[f"{h}_prob_display"].fillna(0.0).clip(0, 1)
        out[f"{h}_prob_rank"] = out[f"{h}_prob_rank"].fillna(out[f"{h}_prob_display"]).clip(0, 1)

    # 排名和风险桶：默认按 y24m raw/rank 口径。
    n = max(len(out), 1)
    out["report_rank"] = out["y24m_prob_rank"].rank(method="first", ascending=False).astype(int)
    out["report_rank_pct"] = out["report_rank"] / n
    if "risk_bucket_by_y24m" not in out.columns:
        out["risk_bucket_by_y24m"] = out["report_rank_pct"].apply(risk_bucket_from_rank_pct)
    else:
        out["risk_bucket_by_y24m"] = out["risk_bucket_by_y24m"].fillna(out["report_rank_pct"].apply(risk_bucket_from_rank_pct))

    # 最可能违约时段。若建模脚本已经保存 most_likely_period，则直接使用；否则用各期差分粗略推断。
    if "most_likely_period" not in out.columns:
        y6 = out["y6m_prob_display"].to_numpy()
        y12 = out["y12m_prob_display"].to_numpy()
        y18 = out["y18m_prob_display"].to_numpy()
        y24 = out["y24m_prob_display"].to_numpy()
        intervals = np.vstack(
            [
                y6,
                np.clip(y12 - y6, 0, 1),
                np.clip(y18 - y12, 0, 1),
                np.clip(y24 - y18, 0, 1),
            ]
        ).T
        labels = np.array(["0–6月", "6–12月", "12–18月", "18–24月"])
        out["most_likely_period"] = labels[np.argmax(intervals, axis=1)]
        out["max_period_prob"] = np.max(intervals, axis=1)
    else:
        out["max_period_prob"] = pd.to_numeric(out.get("max_period_prob", np.nan), errors="coerce").fillna(0.0)

    out["source"] = source
    return out.sort_values("report_rank").reset_index(drop=True)

# =============================================================================
# 5. 页面组件
# =============================================================================

def sidebar_controls() -> Tuple[str, Optional[object], Optional[str]]:
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding: 16px 4px 10px 4px;">
              <div class="eyebrow">{APP_SUBTITLE}</div>
              <div style="font-weight: 850; font-size: 1.15rem; color: #e5edf7; margin-top: 6px;">债券违约预测</div>
              <div class="small-muted" style="margin-top: 6px;">Expanding Window · XGBoost Hazard</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "导航",
            ["首页总览", "验证结果", "高风险预测", "单券查询", "模型说明", "数据状态"],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")
        uploaded_file = st.file_uploader("上传预测 CSV（可选）", type=["csv"])
        explicit_path = st.text_input("预测文件路径（可选）", value="", placeholder="output_expanding/predictions_20251231.csv")
        st.markdown(
            "<div class='small-muted'>没有上传或本地文件时，页面会使用演示数据；部署到 Streamlit Cloud 时，把 output_expanding 一起上传即可。</div>",
            unsafe_allow_html=True,
        )
        return page, uploaded_file, explicit_path.strip() or None


def render_home(pred_df: pd.DataFrame, metrics_df: pd.DataFrame, status: Dict[str, object]) -> None:
    section_header(
        "首页总览",
        "汇总当前预测结果、风险分层结构和模型历史测试表现，帮助快速把握债券池整体风险。",
    )
    st.markdown(
        f"""
        <div class="notice">
        当前数据源：{status.get('message', '')} &nbsp; <span class="badge badge-green">{MODEL_NAME}</span>
        <span class="badge badge-purple">按未来24个月风险排序</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_feature_grid([
        ("📊", "整体风险看板", "汇总预测债券数量、高风险比例和最高风险水平，适合快速浏览债券池状态。"),
        ("✅", "历史测试表现", "展示前三个 Fold 测试窗口表现，观察模型区分能力和稳定性。"),
        ("⚠️", "高风险预警", "按未来24个月违约概率排序，定位 Top 1%、1–5% 等重点债券。"),
        ("🔎", "单券查询", "支持按债券代码、名称或发行人查询，输出单券风险摘要和期限结构。"),
    ])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("预测债券数", f"{len(pred_df):,}", "当前预测主表行数")
    with c2:
        top_count = int((pred_df["risk_bucket_by_y24m"] == "Top 1%").sum())
        metric_card("Top 1% 数量", f"{top_count:,}", "按未来24个月风险排名")
    with c3:
        metric_card("最高未来24个月风险", pct(pred_df["y24m_prob_display"].max()), "当前样本最高预测值")
    with c4:
        metric_card("Fold 测试记录", f"{len(metrics_df):,}", "历史测试窗口指标行数")

    left, right = st.columns([1.05, 1])
    with left:
        bucket_counts = (
            pred_df["risk_bucket_by_y24m"]
            .value_counts()
            .reindex(RISK_BUCKET_ORDER)
            .fillna(0)
            .reset_index()
        )
        bucket_counts.columns = ["风险分层", "债券数"]
        fig = px.bar(bucket_counts, x="风险分层", y="债券数", title="风险分层分布（按未来24个月风险排序）")
        fig = plot_layout(fig, height=360)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        insight_box("图表解读", "该图展示不同风险层级的债券数量。Top 1% 和 1–5% 是最需要优先关注的债券池；10–20% 可作为次级观察名单，Other 表示当前排序相对靠后。")

    with right:
        summary = metrics_df.groupby("horizon", as_index=False).agg(
            pr_auc_mean=("pr_auc", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            top5_recall_mean=("top5_recall", "mean"),
        )
        summary["horizon_order"] = summary["horizon"].map({"6m": 1, "12m": 2, "18m": 3, "24m": 4})
        summary["horizon_cn"] = summary["horizon"].map({"6m": "未来6个月", "12m": "未来12个月", "18m": "未来18个月", "24m": "未来24个月"})
        summary = summary.sort_values("horizon_order")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=summary["horizon_cn"], y=summary["pr_auc_mean"], mode="lines+markers", name="平均 PR-AUC"))
        fig.add_trace(go.Scatter(x=summary["horizon_cn"], y=summary["roc_auc_mean"], mode="lines+markers", name="平均 ROC-AUC"))
        fig.add_trace(go.Scatter(x=summary["horizon_cn"], y=summary["top5_recall_mean"], mode="lines+markers", name="平均 Top5%召回率"))
        fig.update_layout(title="前三个 Fold 历史测试表现摘要")
        fig = plot_layout(fig, height=360)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        insight_box("图表解读", "ROC-AUC 反映整体区分能力；PR-AUC 更适合违约样本稀缺的场景；Top5%召回率表示模型把真实高风险样本筛入前 5% 名单的能力。")

    st.markdown(
        f"""
        <div class="summary-box">
          <b>当前债券池解读：</b>系统默认使用未来24个月风险排序进行分层，当前 Top 1% 债券共 <b>{top_count:,}</b> 只，
          最高未来24个月预测概率为 <b>{pct(pred_df['y24m_prob_display'].max())}</b>。建议先查看“高风险预测”页面形成预警名单，再进入“单券查询”查看具体债券的风险释放窗口。
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("高风险债券 TOP 10")
    show_cols = ["report_rank", "BondCode", "BondName", "Issuer", "risk_bucket_by_y24m", "y6m_prob_display", "y12m_prob_display", "y18m_prob_display", "y24m_prob_display", "most_likely_period"]
    view = pred_df[show_cols].head(10).copy()
    view = format_prediction_table(view)
    st.dataframe(view, use_container_width=True, hide_index=True)


def render_validation(metrics_df: pd.DataFrame) -> None:
    section_header(
        "验证结果：前三个 Fold 测试集表现",
        "展示模型在历史滚动测试窗口中的区分能力、Top-k 捕获能力、误差水平和跨期稳定性。",
    )
    st.markdown(
        "<div class='notice'>测试集表现用于观察模型在不同时间窗口中的稳定性；图表中的指标包括 ROC-AUC、PR-AUC、Brier、Log Loss 与 Top-k Precision / Recall。</div>",
        unsafe_allow_html=True,
    )

    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("平均 PR-AUC", fmt_num(metrics_df["pr_auc"].mean()), "越高越好，越能识别少数违约样本")
        with c2:
            metric_card("平均 ROC-AUC", fmt_num(metrics_df["roc_auc"].mean()), "越高越好，表示区分能力更强")
        with c3:
            metric_card("平均 Top5%召回率", pct(metrics_df["top5_recall"].mean()), "越高说明高风险债券捕捉越充分")
        with c4:
            metric_card("平均 Brier", fmt_num(metrics_df["brier"].mean()), "越低越好，表示概率误差更小")

    horizon_map = {"未来6个月": "6m", "未来12个月": "12m", "未来18个月": "18m", "未来24个月": "24m"}
    selected_horizon_cn = st.multiselect(
        "选择预测期限",
        options=list(horizon_map.keys()),
        default=list(horizon_map.keys()),
    )
    selected_horizons = [horizon_map[x] for x in selected_horizon_cn]
    selected_folds = st.multiselect(
        "选择 Fold",
        options=["Fold1", "Fold2", "Fold3"],
        default=["Fold1", "Fold2", "Fold3"],
    )
    df = metrics_df[metrics_df["horizon"].isin(selected_horizons) & metrics_df["fold"].isin(selected_folds)].copy()
    df["期限中文"] = df["horizon"].map({"6m": "未来6个月", "12m": "未来12个月", "18m": "未来18个月", "24m": "未来24个月"})

    tab1, tab2, tab3, tab4 = st.tabs(["区分能力", "Top-k 捕获", "稳定性", "明细表"])
    with tab1:
        long = df.melt(id_vars=["fold", "期限中文"], value_vars=["roc_auc", "pr_auc"], var_name="指标", value_name="数值")
        long["指标"] = long["指标"].replace({"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"})
        fig = px.bar(long, x="期限中文", y="数值", color="fold", barmode="group", facet_col="指标", title="各 Fold × 预测期限的区分能力")
        fig.update_yaxes(range=[0, 1])
        fig = plot_layout(fig, height=420)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        insight_box("图表解读", "ROC-AUC 用于观察模型对违约与非违约债券的整体区分能力；PR-AUC 更适合违约预测这类样本不平衡场景。若两项指标在不同 Fold 中波动不大，说明模型具有较好的时间稳定性。")

    with tab2:
        metric_name_map = {
            "top1_precision": "Top1% 精确率",
            "top1_recall": "Top1% 召回率",
            "top5_precision": "Top5% 精确率",
            "top5_recall": "Top5% 召回率",
            "top10_precision": "Top10% 精确率",
            "top10_recall": "Top10% 召回率",
        }
        top_metric = st.selectbox("Top-k 指标", list(metric_name_map.keys()), format_func=lambda x: metric_name_map[x])
        fig = px.bar(df, x="期限中文", y=top_metric, color="fold", barmode="group", title=f"{metric_name_map[top_metric]}：不同 Fold 对比")
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        fig = plot_layout(fig, height=400)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(df, x="期限中文", y="top5_precision", color="fold", markers=True, title="Top5% 精确率")
            fig.update_yaxes(range=[0, 1], tickformat=".0%")
            fig = plot_layout(fig, height=330)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with c2:
            fig = px.line(df, x="期限中文", y="top5_recall", color="fold", markers=True, title="Top5% 召回率")
            fig.update_yaxes(range=[0, 1], tickformat=".0%")
            fig = plot_layout(fig, height=330)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        insight_box("图表解读", "Top-k 指标反映模型把高风险债券排在前列的能力。召回率高，说明真实违约债券中有更大比例被提前筛入重点名单；精确率高，则说明前列名单中“真正高风险”的占比更高。")

    with tab3:
        stability = df.groupby("fold", as_index=False).agg(
            pr_auc=("pr_auc", "mean"),
            roc_auc=("roc_auc", "mean"),
            brier=("brier", "mean"),
            top5_recall=("top5_recall", "mean"),
        )
        fig = go.Figure()
        for col, name in [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("top5_recall", "Top5% 召回率")]:
            fig.add_trace(go.Scatter(x=stability["fold"], y=stability[col], mode="lines+markers", name=name))
        fig.update_layout(title="跨 Fold 平均表现稳定性")
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        fig = plot_layout(fig, height=380)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        insight_box("图表解读", "如果不同 Fold 之间曲线较平稳，说明模型在不同时间窗口上的表现较一致，具有更好的泛化能力。若出现明显下降，则提示模型可能对某些年份或市场环境更敏感。")

        st.dataframe(format_metric_table(stability), use_container_width=True, hide_index=True)

    with tab4:
        st.dataframe(format_metric_table(df), use_container_width=True, hide_index=True)
        csv = df.drop(columns=["horizon_order"], errors="ignore").to_csv(index=False, encoding="utf-8-sig")
        st.download_button("下载 Fold 测试集指标 CSV", data=csv, file_name="fold_test_metrics.csv", mime="text/csv")


def render_predictions(pred_df: pd.DataFrame) -> None:
    section_header(
        "高风险预测",
        "展示当前预测日期下的高风险债券列表。默认按未来24个月预测概率排序，并用排名百分位划分风险桶。",
    )
    st.markdown(
        "<div class='notice'>风险分层使用排名百分位：Top 1% / 1–5% / 5–10% / 10–20% / Other。用户可优先关注排序靠前且长期违约概率较高的债券。</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='rank-band'><span></span><span></span><span></span><span></span><span></span></div>", unsafe_allow_html=True)
    st.markdown("<div class='pill-row'><span class='soft-pill'>红色：Top 1%</span><span class='soft-pill'>橙色：1–5%</span><span class='soft-pill'>黄色：5–10%</span><span class='soft-pill'>绿色：10–20%</span><span class='soft-pill'>灰色：Other</span></div>", unsafe_allow_html=True)

    top_cols = st.columns([1, 1, 2])
    with top_cols[0]:
        horizon = st.selectbox("排序/展示期限", HORIZON_ORDER, index=3, format_func=lambda h: HORIZON_LABELS[h])
    with top_cols[1]:
        buckets = st.multiselect("风险分层", RISK_BUCKET_ORDER, default=RISK_BUCKET_ORDER[:4])
    with top_cols[2]:
        query = st.text_input("搜索债券代码 / 名称 / 发行人", value="")

    df = pred_df.copy()
    df["rank_for_view"] = df[f"{horizon}_prob_rank"].rank(method="first", ascending=False).astype(int)
    df = df[df["risk_bucket_by_y24m"].isin(buckets)] if buckets else df.iloc[0:0]
    if query:
        q = query.lower().strip()
        mask = (
            df["BondCode"].astype(str).str.lower().str.contains(q, na=False)
            | df["BondName"].astype(str).str.lower().str.contains(q, na=False)
            | df["Issuer"].astype(str).str.lower().str.contains(q, na=False)
            | df["Liscd"].astype(str).str.lower().str.contains(q, na=False)
        )
        df = df[mask]

    df = df.sort_values(f"{horizon}_prob_rank", ascending=False)
    if df.empty:
        st.warning("当前筛选条件下没有匹配债券。")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("筛选后债券数", f"{len(df):,}", "当前条件下的债券数量")
    with c2:
        metric_card("筛选均值", pct(df[f"{horizon}_prob_display"].mean()), HORIZON_LABELS[horizon])
    with c3:
        metric_card("筛选最大值", pct(df[f"{horizon}_prob_display"].max()), "当前筛选集最高风险")
    with c4:
        metric_card("Top 1% 占比", pct((df["risk_bucket_by_y24m"] == "Top 1%").mean()), "筛选集中最高风险桶比例")

    c1, c2 = st.columns([1.1, 1])
    with c1:
        top_n = df.head(20).copy()
        fig = px.bar(
            top_n.sort_values(f"{horizon}_prob_display"),
            x=f"{horizon}_prob_display",
            y="BondCode",
            orientation="h",
            color="risk_bucket_by_y24m",
            category_orders={"risk_bucket_by_y24m": RISK_BUCKET_ORDER},
            title=f"TOP 20：{HORIZON_LABELS[horizon]}预测概率",
        )
        fig.update_xaxes(tickformat=".1%")
        fig = plot_layout(fig, height=530)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        insight_box("图表解读", "左图按照所选期限对高风险债券进行排序。柱子越长，表示该债券在对应预测期限内的违约风险越高；同时处于 Top 1% 或 1–5% 的债券应优先进入预警名单。")
    with c2:
        means = pd.DataFrame({
            "horizon": [HORIZON_SHORT[h] for h in HORIZON_ORDER],
            "avg_prob": [df[f"{h}_prob_display"].mean() if len(df) else 0 for h in HORIZON_ORDER],
            "max_prob": [df[f"{h}_prob_display"].max() if len(df) else 0 for h in HORIZON_ORDER],
        })
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=means["horizon"], y=means["avg_prob"], mode="lines+markers", name="筛选均值"))
        fig.add_trace(go.Scatter(x=means["horizon"], y=means["max_prob"], mode="lines+markers", name="筛选最大值"))
        fig.update_yaxes(tickformat=".1%")
        fig.update_layout(title="筛选债券的期限结构概览")
        fig = plot_layout(fig, height=530)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        insight_box("图表解读", "右图展示当前筛选债券在不同预测期限上的平均风险与最大风险。平均值反映筛选组合整体风险，最大值用于识别极端高风险样本。")

    st.markdown(
        f"""
        <div class="summary-box">
          <b>筛选结果摘要：</b>当前条件下共筛出 <b>{len(df):,}</b> 只债券，{HORIZON_LABELS[horizon]}平均预测概率为 <b>{pct(df[f'{horizon}_prob_display'].mean())}</b>，
          最高值为 <b>{pct(df[f'{horizon}_prob_display'].max())}</b>。建议优先查看排名靠前、风险分层为 Top 1% 或 1–5% 的债券。
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("预测明细")
    show_cols = ["report_rank", "BondCode", "BondName", "Issuer", "risk_bucket_by_y24m", "y6m_prob_display", "y12m_prob_display", "y18m_prob_display", "y24m_prob_display", "report_rank_pct", "most_likely_period"]
    view = format_prediction_table(df[show_cols].head(500).copy())
    st.dataframe(view, use_container_width=True, hide_index=True, height=430)

    csv = df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button("下载当前筛选结果 CSV", data=csv, file_name="bond_default_predictions_filtered.csv", mime="text/csv")


def render_bond_query(pred_df: pd.DataFrame) -> None:
    section_header(
        "单券查询",
        "输入债券代码、债券名称或发行人关键词，查看单支债券的风险等级、风险排名、违约时间结构和模型解释摘要。",
    )

    search_cols = st.columns([2.2, 1, 1])
    with search_cols[0]:
        default_code = str(pred_df.iloc[0]["BondCode"]) if len(pred_df) else ""
        query = st.text_input("债券代码 / 名称 / 发行人", value=default_code, placeholder="例如：111023、某某债、发行人名称")
    with search_cols[1]:
        rank_horizon = st.selectbox("重点观察期限", HORIZON_ORDER, index=3, format_func=lambda h: HORIZON_LABELS[h])
    with search_cols[2]:
        max_matches = st.number_input("最多显示匹配数", min_value=5, max_value=100, value=20, step=5)

    if not query:
        st.info("请输入关键词进行查询。")
        return

    q = query.lower().strip()
    matches = pred_df[
        pred_df["BondCode"].astype(str).str.lower().str.contains(q, na=False)
        | pred_df["BondName"].astype(str).str.lower().str.contains(q, na=False)
        | pred_df["Issuer"].astype(str).str.lower().str.contains(q, na=False)
        | pred_df["Liscd"].astype(str).str.lower().str.contains(q, na=False)
    ].copy()

    if matches.empty:
        st.warning("没有找到匹配债券。可以尝试输入更短的债券代码、债券简称或发行人关键词。")
        return

    matches = matches.sort_values("report_rank").head(int(max_matches))
    if len(matches) > 1:
        label_options = [
            f"{r.BondCode}｜{r.BondName}｜{r.Issuer}｜Rank #{int(r.report_rank)}"
            for r in matches.itertuples()
        ]
        selected_label = st.selectbox("找到多条结果，请选择一只债券", label_options)
        selected_code = selected_label.split("｜", 1)[0]
        row = matches[matches["BondCode"].astype(str) == selected_code].iloc[0]
    else:
        row = matches.iloc[0]

    y6 = float(row.get("y6m_prob_display", 0.0))
    y12 = float(row.get("y12m_prob_display", 0.0))
    y18 = float(row.get("y18m_prob_display", 0.0))
    y24 = float(row.get("y24m_prob_display", 0.0))
    rank_pct = float(row.get("report_rank_pct", 1.0))
    bucket = str(row.get("risk_bucket_by_y24m", "Other"))
    if bucket in ["Top 1%", "1–5%"] or y24 >= 0.30:
        risk_level, risk_tone = "高风险", "#fecaca"
    elif bucket in ["5–10%", "10–20%"] or y24 >= 0.12:
        risk_level, risk_tone = "中风险", "#fde68a"
    else:
        risk_level, risk_tone = "低风险", "#bbf7d0"
    is_top1 = bucket == "Top 1%" or rank_pct <= 0.01

    st.markdown(
        f"""
        <div class="card">
          <div class="eyebrow">Bond Detail</div>
          <h3 style="margin: 6px 0 8px 0;">{row.get('BondName', row.get('BondCode', ''))}</h3>
          <div class="small-muted">代码：{row.get('BondCode', row.get('Liscd', ''))}　|　发行人：{row.get('Issuer', '未知')}　|　预测日期：{row.get('PeriodEnd', '未知')}</div>
          <div style="margin-top: 12px;">{display_bucket(bucket)} <span class="badge badge-purple">Rank #{int(row.get('report_rank', 0))}</span> <span class="badge badge-green">排名百分位 {pct(rank_pct)}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="risk-card"><div class="risk-card-label">综合风险等级</div><div class="risk-card-value" style="color:{risk_tone};">{risk_level}</div><div class="risk-card-help">未来24个月累计违约概率：{pct(y24)}</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="risk-card"><div class="risk-card-label">最可能违约时段</div><div class="risk-card-value">{row.get('most_likely_period', '—')}</div><div class="risk-card-help">区间概率：{pct(row.get('max_period_prob', np.nan))}</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="risk-card"><div class="risk-card-label">Top 1% 高风险</div><div class="risk-card-value" style="color:{'#fecaca' if is_top1 else '#bbf7d0'};">{'是' if is_top1 else '否'}</div><div class="risk-card-help">按未来24个月风险排名判断</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="risk-card"><div class="risk-card-label">重点期限风险</div><div class="risk-card-value">{pct(row[f'{rank_horizon}_prob_display'])}</div><div class="risk-card-help">{HORIZON_LABELS[rank_horizon]}</div></div>""", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="summary-box">
          <b>单券风险结论：</b>{probability_sentence(row)} {rank_position_sentence(rank_pct, bucket)}
          该债券未来6个月、12个月、18个月、24个月累计违约概率分别为 <b>{pct(y6)}</b>、<b>{pct(y12)}</b>、<b>{pct(y18)}</b>、<b>{pct(y24)}</b>。
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.1, 1])
    with left:
        term_df = pd.DataFrame({
            "期限": [HORIZON_LABELS[h] for h in HORIZON_ORDER],
            "累计违约概率": [row[f"{h}_prob_display"] for h in HORIZON_ORDER],
        })
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=term_df["期限"], y=term_df["累计违约概率"], mode="lines+markers", name="累计违约概率", line=dict(width=3)))
        fig.add_trace(go.Bar(x=term_df["期限"], y=term_df["累计违约概率"], name="概率柱", opacity=0.35))
        fig.update_yaxes(tickformat=".1%")
        fig.update_layout(title="累计违约概率期限结构")
        fig = plot_layout(fig, height=430)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        insight_box("图表解读", "该图展示单只债券在不同预测期限上的累计违约概率。曲线越陡，说明随着时间拉长，风险累积越明显；如果未来6个月概率已经较高，则更需要关注近期风险。")

    with right:
        intervals = pd.DataFrame({
            "时间段": ["0–6个月", "6–12个月", "12–18个月", "18–24个月"],
            "区间概率": [y6, max(y12 - y6, 0), max(y18 - y12, 0), max(y24 - y18, 0)],
        })
        fig = px.bar(intervals, x="时间段", y="区间概率", title="区间违约风险拆分")
        fig.update_yaxes(tickformat=".1%")
        fig = plot_layout(fig, height=430)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        insight_box("图表解读", "该图把累计违约概率拆分为各时间区间的新增风险贡献，有助于判断风险更可能集中在近期还是中长期。")

    detail = pd.DataFrame({
        "项目": ["债券代码", "债券名称", "发行人", "预测日期", "风险分层", "排名", "排名百分位", "未来6个月违约概率", "未来12个月违约概率", "未来18个月违约概率", "未来24个月违约概率"],
        "内容": [
            row.get("BondCode", "—"), row.get("BondName", "—"), row.get("Issuer", "—"), row.get("PeriodEnd", "—"), bucket,
            int(row.get("report_rank", 0)), pct(rank_pct), pct(y6), pct(y12), pct(y18), pct(y24),
        ],
    })
    st.dataframe(detail, use_container_width=True, hide_index=True)


def render_model_note(metrics_df: pd.DataFrame) -> None:
    section_header(
        "模型说明",
        "说明系统的预测口径、核心页面功能和风险解释方式，便于用户理解各项输出。",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("模型框架", "XGBoost", "离散 Hazard 违约预测")
    with c2:
        metric_card("预测期限", "4 个", "未来6个月 / 12个月 / 18个月 / 24个月")
    with c3:
        metric_card("验证方式", "Rolling Fold", "按时间窗口检验稳定性")

    st.markdown(
        """
        <div class="card">
        <h3>系统功能</h3>
        <p><b>首页总览</b>：查看债券池规模、风险分层分布、期限结构均值与历史测试表现摘要。</p>
        <p><b>验证结果</b>：比较模型在多个历史测试窗口中的 ROC-AUC、PR-AUC、Brier、Log Loss 和 Top-k 捕获效果。</p>
        <p><b>高风险预测</b>：按照风险排名筛选 Top 1%、1–5%、5–10%、10–20% 等重点债券，并支持导出明细。</p>
        <p><b>单券查询</b>：输入债券代码、债券名称或发行人关键词，查看单只债券的违约概率期限结构、风险分层、排名位置和风险摘要。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 建模口径")
    st.markdown(
        """
        - 模型采用 expanding window 思路，训练窗口随时间扩张，保持时间顺序，降低未来信息泄漏风险。
        - 输出结果包括未来 6 个月、12 个月、18 个月、24 个月的累计违约概率，页面已统一使用中文标签展示。
        - 风险分层以未来24个月风险排名百分位为核心口径：Top 1%、1–5%、5–10%、10–20%、Other。
        - 概率数值用于描述模型判断强弱，风险排名用于识别相对更需要关注的债券。
        """
    )
    st.markdown(
        """
        <div class="glossary-grid">
          <div class="glossary-card"><b>ROC-AUC</b><p>衡量模型区分违约与非违约债券的整体能力，越高越好。</p></div>
          <div class="glossary-card"><b>PR-AUC</b><p>更适合违约样本较少的场景，反映模型识别少数高风险样本的能力。</p></div>
          <div class="glossary-card"><b>Top-k 召回率</b><p>表示真实高风险债券中有多少被筛入前 k% 预警名单。</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 历史测试表现摘要")
    st.dataframe(format_metric_table(metrics_df), use_container_width=True, hide_index=True)

def render_data_status(pred_df: pd.DataFrame, status: Dict[str, object]) -> None:
    section_header(
        "数据与运行状态",
        "查看当前页面读取的预测文件、预测行数、字段要求和数据预览。",
    )

    expected = [
        f"{DEFAULT_OUTPUT_DIR}/predictions_*.csv",
        "fold_test_metrics.csv（可选，用于替换内置测试集指标）",
    ]
    rows = []
    for item in expected:
        if "*.csv" in item:
            matched = glob.glob(item)
            rows.append({"项目": item, "状态": "存在" if matched else "缺失", "说明": "; ".join(matched[:3]) if matched else "未发现"})
        else:
            rows.append({"项目": item, "状态": "内置", "说明": "当前页面已内置测试集指标，也可以后续替换为外部 CSV。"})

    st.markdown(
        f"""
        <div class="notice">
        当前读取模式：<b>{status.get('mode')}</b>；来源：<b>{status.get('source_path') or 'demo data'}</b>；当前预测行数：<b>{len(pred_df):,}</b>。
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### 预测文件需要包含的最低字段")
    st.code(
        """Liscd, PeriodEnd, sem_str, y6m_cum_prob, y12m_cum_prob, y18m_cum_prob, y24m_cum_prob
# 可选字段：BondCode, BondName, Issuer, most_likely_period, max_period_prob""",
        language="text",
    )

    preview = format_prediction_table(pred_df.head(20).copy())
    st.markdown("### 当前数据预览")
    st.dataframe(preview, use_container_width=True, hide_index=True)


# =============================================================================
# 6. 表格格式化
# =============================================================================

def format_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {
        "fold": "Fold",
        "test_year": "测试集",
        "horizon": "期限",
        "n_samples": "样本数",
        "n_positive": "正样本",
        "base_rate": "基准率",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "brier": "Brier",
        "log_loss": "LogLoss",
        "top1_precision": "Top1% P",
        "top1_recall": "Top1% R",
        "top5_precision": "Top5% P",
        "top5_recall": "Top5% R",
        "top10_precision": "Top10% P",
        "top10_recall": "Top10% R",
        "note": "备注",
    }
    out = out.drop(columns=["horizon_order"], errors="ignore")
    out = out.rename(columns=rename)
    if "期限" in out.columns:
        out["期限"] = out["期限"].replace({"6m": "未来6个月", "12m": "未来12个月", "18m": "未来18个月", "24m": "未来24个月"})
    percent_cols = ["基准率", "Top1% P", "Top1% R", "Top5% P", "Top5% R", "Top10% P", "Top10% R"]
    for c in percent_cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: pct(x, 2))
    for c in ["ROC-AUC", "PR-AUC", "Brier", "LogLoss"]:
        if c in out.columns:
            out[c] = out[c].map(lambda x: fmt_num(x, 4))
    return out


def format_prediction_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {
        "report_rank": "排名",
        "rank_for_view": "展示排名",
        "BondCode": "债券代码",
        "BondName": "债券名称",
        "Issuer": "发行人",
        "risk_bucket_by_y24m": "风险分层",
        "y6m_prob_display": "未来6个月违约概率",
        "y12m_prob_display": "未来12个月违约概率",
        "y18m_prob_display": "未来18个月违约概率",
        "y24m_prob_display": "未来24个月违约概率",
        "report_rank_pct": "排名百分位",
        "most_likely_period": "最可能时段",
    }
    out = out.rename(columns=rename)
    for c in ["未来6个月违约概率", "未来12个月违约概率", "未来18个月违约概率", "未来24个月违约概率", "排名百分位"]:
        if c in out.columns:
            out[c] = out[c].map(lambda x: pct(x, 2))
    return out


def render_cover_page() -> None:
    st.markdown(
        """
        <div class="cover-wrap">
          <div class="cover-card">
            <div class="eyebrow">BOND DEFAULT · PREDICTION</div>
            <div class="cover-title">债券违约预测<br/>可视化系统</div>
            <div class="cover-sub">
              本系统面向债券风险识别与预警场景，基于机器学习模型输出不同期限的累计违约概率，
              并通过风险分层、历史测试表现、高风险排行和单券查询，帮助用户快速定位需要重点关注的债券。
            </div>
            <div class="cover-grid">
              <div class="cover-feature"><div class="icon">📊</div><b>首页总览</b><p>查看债券池整体风险、分层结构和核心指标。</p></div>
              <div class="cover-feature"><div class="icon">✅</div><b>验证结果</b><p>展示多个历史测试窗口下的模型表现。</p></div>
              <div class="cover-feature"><div class="icon">⚠️</div><b>高风险预测</b><p>按风险排名筛选 Top 债券并支持导出。</p></div>
              <div class="cover-feature"><div class="icon">🔎</div><b>单券查询</b><p>输入代码、名称或发行人，查看单券风险详情。</p></div>
            </div>
            <div class="pill-row" style="margin-top:26px;">
              <span class="soft-pill">Expanding Window</span><span class="soft-pill">XGBoost Hazard</span><span class="soft-pill">未来6/12/18/24个月</span><span class="soft-pill">Top-k 风险捕获</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1.6, 1, 1.6])
    with col2:
        if st.button("Continue", use_container_width=True):
            st.session_state["entered_app"] = True
            st.rerun()

# =============================================================================
# 7. 主入口
# =============================================================================

def main() -> None:
    inject_css()

    if "entered_app" not in st.session_state:
        st.session_state["entered_app"] = False

    if not st.session_state["entered_app"]:
        render_cover_page()
        return

    page, uploaded_file, explicit_path = sidebar_controls()
    pred_df, status = read_prediction_file(uploaded_file=uploaded_file, explicit_path=explicit_path)
    metrics_df = default_fold_metrics()

    with st.sidebar:
        st.markdown("---")
        if st.button("返回封面", use_container_width=True):
            st.session_state["entered_app"] = False
            st.rerun()

    if page == "首页总览":
        render_home(pred_df, metrics_df, status)
    elif page == "验证结果":
        render_validation(metrics_df)
    elif page == "高风险预测":
        render_predictions(pred_df)
    elif page == "单券查询":
        render_bond_query(pred_df)
    elif page == "模型说明":
        render_model_note(metrics_df)
    elif page == "数据状态":
        render_data_status(pred_df, status)


if __name__ == "__main__":
    main()
