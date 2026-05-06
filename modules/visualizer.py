"""
Visualization Module
Generates base64-encoded charts for embedding in HTML reports.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# Style config
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor': '#1a1d2e',
    'axes.edgecolor': '#2d3154',
    'axes.labelcolor': '#c8d0e7',
    'xtick.color': '#8892b0',
    'ytick.color': '#8892b0',
    'text.color': '#ccd6f6',
    'grid.color': '#1e2340',
    'grid.alpha': 0.5,
    'font.family': 'DejaVu Sans',
    'axes.titlecolor': '#e6f1ff',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})
ACCENT = '#64ffda'
ACCENT2 = '#7b68ee'
WARN = '#f7c59f'
DANGER = '#ff6b6b'

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    plt.close('all')  # belt-and-braces: prevent fig cache growth across requests
    return img_b64

def plot_missing_values(missing_data):
    per_col = missing_data.get('per_column', {})
    if not per_col:
        return None
    cols = list(per_col.keys())[:20]
    pcts = [per_col[c]['percentage'] for c in cols]
    fig, ax = plt.subplots(figsize=(10, max(4, len(cols) * 0.4)))
    colors = [DANGER if p > 30 else WARN if p > 10 else ACCENT for p in pcts]
    bars = ax.barh(cols, pcts, color=colors, height=0.6)
    ax.set_xlabel('Missing Percentage (%)')
    ax.set_title('Missing Values by Column')
    ax.axvline(x=30, color=DANGER, linestyle='--', alpha=0.6, linewidth=1.2, label='High (30%)')
    ax.axvline(x=10, color=WARN, linestyle='--', alpha=0.6, linewidth=1.2, label='Medium (10%)')
    ax.legend(loc='lower right', facecolor='#1a1d2e', edgecolor='#2d3154')
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{pct:.1f}%', va='center', fontsize=9, color='#ccd6f6')
    plt.tight_layout()
    return fig_to_b64(fig)

def plot_class_distribution(imbalance_data):
    dist = imbalance_data.get('class_distribution', {})
    if not dist:
        return None
    labels = [str(k) for k in dist.keys()]
    values = list(dist.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = [ACCENT, ACCENT2, WARN, DANGER, '#9b59b6', '#2ecc71']
    colors = palette[:len(labels)]
    bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor='#2d3154', linewidth=0.8)
    ax.set_title(f'Class Distribution — Target: {imbalance_data.get("target_column", "")}')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f'{bar.get_height():,}', ha='center', fontsize=9, color='#ccd6f6')
    plt.tight_layout()
    return fig_to_b64(fig)

def plot_correlation_heatmap(corr_data):
    matrix = corr_data.get('correlation_matrix', {})
    if not matrix:
        return None
    cols = list(matrix.keys())
    df_corr = pd.DataFrame(matrix, index=cols, columns=cols)
    fig, ax = plt.subplots(figsize=(max(8, len(cols)), max(6, len(cols) * 0.7)))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(df_corr, ax=ax, cmap=cmap, center=0, vmin=-1, vmax=1,
                annot=len(cols) <= 12, fmt='.2f', linewidths=0.5,
                linecolor='#0f1117', annot_kws={'size': 8, 'color': '#ccd6f6'},
                cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    return fig_to_b64(fig)

def plot_outlier_summary(outlier_data):
    per_col = outlier_data.get('per_column', {})
    if not per_col:
        return None
    cols = list(per_col.keys())[:15]
    pcts = [per_col[c]['percentage'] for c in cols]
    fig, ax = plt.subplots(figsize=(10, max(4, len(cols) * 0.45)))
    colors = [DANGER if p > 10 else WARN if p > 3 else ACCENT for p in pcts]
    ax.barh(cols, pcts, color=colors, height=0.6, edgecolor='#2d3154', linewidth=0.5)
    ax.set_xlabel('Outlier Percentage (%)')
    ax.set_title('Outlier Distribution by Feature')
    plt.tight_layout()
    return fig_to_b64(fig)

def plot_skewness(skew_data):
    skewed = skew_data.get('skewed_columns', {})
    if not skewed:
        return None
    cols = list(skewed.keys())[:15]
    values = [skewed[c]['skewness'] for c in cols]
    fig, ax = plt.subplots(figsize=(10, max(4, len(cols) * 0.45)))
    colors = [DANGER if abs(v) > 2 else WARN if abs(v) > 1.5 else ACCENT for v in values]
    ax.barh(cols, values, color=colors, height=0.6, edgecolor='#2d3154', linewidth=0.5)
    ax.axvline(x=0, color='#8892b0', linewidth=1)
    ax.axvline(x=1, color=WARN, linestyle='--', alpha=0.6, linewidth=1)
    ax.axvline(x=-1, color=WARN, linestyle='--', alpha=0.6, linewidth=1)
    ax.set_xlabel('Skewness')
    ax.set_title('Feature Skewness')
    plt.tight_layout()
    return fig_to_b64(fig)

def plot_health_radar(health_score_data, severity_breakdown):
    categories = list(severity_breakdown.keys())
    values = [3 - v for v in severity_breakdown.values()]  # Invert: 3=good
    N = len(categories)
    if N < 3:
        return None
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_facecolor('#1a1d2e')
    ax.plot(angles, values_plot, color=ACCENT, linewidth=2)
    ax.fill(angles, values_plot, color=ACCENT, alpha=0.25)
    ax.set_xticks(angles[:-1])
    clean_labels = [c.replace('_', '\n').title() for c in categories]
    ax.set_xticklabels(clean_labels, size=9, color='#ccd6f6')
    ax.set_ylim(0, 3)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Low', 'Med', 'High'], color='#8892b0', size=8)
    ax.set_title(f'Dataset Health Overview\nScore: {health_score_data["score"]}/100 (Grade {health_score_data["grade"]})',
                 pad=20, color='#e6f1ff', size=13)
    ax.grid(color='#2d3154', linewidth=0.8)
    ax.spines['polar'].set_color('#2d3154')
    plt.tight_layout()
    return fig_to_b64(fig)

def plot_model_comparison(comparison_data):
    improvement = comparison_data.get('improvement', {})
    if not improvement:
        return None
    metrics = list(improvement.keys())
    before_vals = [improvement[m]['before'] for m in metrics]
    after_vals = [improvement[m]['after'] for m in metrics]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, before_vals, width, label='Before Repair', color=DANGER, alpha=0.85, edgecolor='#2d3154')
    bars2 = ax.bar(x + width / 2, after_vals, width, label='After Repair', color=ACCENT, alpha=0.85, edgecolor='#2d3154')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance: Before vs After Dataset Repair')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend(facecolor='#1a1d2e', edgecolor='#2d3154')
    ax.set_ylim(0, 1.1)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', fontsize=8, color='#ccd6f6')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', fontsize=8, color='#ccd6f6')
    plt.tight_layout()
    return fig_to_b64(fig)

def generate_all_charts(diagnostics_results):
    charts = {}
    try:
        mv_chart = plot_missing_values(diagnostics_results.get('missing_values', {}))
        if mv_chart:
            charts['missing_values'] = mv_chart
    except Exception:
        pass
    try:
        ci_chart = plot_class_distribution(diagnostics_results.get('class_imbalance', {}))
        if ci_chart:
            charts['class_imbalance'] = ci_chart
    except Exception:
        pass
    try:
        corr_chart = plot_correlation_heatmap(diagnostics_results.get('feature_correlation', {}))
        if corr_chart:
            charts['correlation'] = corr_chart
    except Exception:
        pass
    try:
        out_chart = plot_outlier_summary(diagnostics_results.get('outliers', {}))
        if out_chart:
            charts['outliers'] = out_chart
    except Exception:
        pass
    try:
        skew_chart = plot_skewness(diagnostics_results.get('distribution_skew', {}))
        if skew_chart:
            charts['skewness'] = skew_chart
    except Exception:
        pass
    try:
        health = diagnostics_results.get('health_score', {})
        severity = health.get('severity_breakdown', {})
        if severity:
            radar = plot_health_radar(health, severity)
            if radar:
                charts['health_radar'] = radar
    except Exception:
        pass
    return charts
