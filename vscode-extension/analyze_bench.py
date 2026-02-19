#!/usr/bin/env python3
"""
DEBATE EXT — Benchmark Analysis & Paper-Ready Figures

Reads the JSON report produced by benchmark.ts and generates:
  1. Statistical tests (Wilcoxon signed-rank between config pairs)
  2. Bar charts (quality by config, by category)
  3. Heatmap (config × category quality)
  4. Radar chart (5 criteria per config)
  5. Judge agreement analysis (inter-judge correlation + divergence audit)
  6. Paper-ready summary tables (LaTeX + Markdown)

Usage:
  python3 analyze_bench.py                           # auto-find latest report
  python3 analyze_bench.py path/to/bench_*.json      # specific report
"""

import json
import sys
import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ─── Load report ──────────────────────────────────────────────────────

def find_latest_report():
    pattern = os.path.join(os.path.dirname(__file__), 'benchmark-results', 'bench_*.json')
    files = sorted(glob.glob(pattern))
    if not files:
        print("ERROR: No benchmark report found in benchmark-results/")
        sys.exit(1)
    return files[-1]

def load_report(path):
    with open(path) as f:
        return json.load(f)

# ─── Extract data ─────────────────────────────────────────────────────

CONFIGS = ['claude-solo', 'openai-solo', 'claude-lead', 'openai-lead', 'claude-orch', 'openai-orch']
CONFIG_SHORT = {
    'claude-solo': 'Cl.Solo', 'openai-solo': 'OA.Solo',
    'claude-lead': 'Cl.Lead', 'openai-lead': 'OA.Lead',
    'claude-orch': 'Cl.Orch', 'openai-orch': 'OA.Orch',
}
CRITERIA = ['correctness', 'completeness', 'edgeCases', 'codeQuality', 'readability']

def extract_quality_df(report):
    """Build a DataFrame: rows=cases, columns=config quality scores."""
    rows = []
    for case in report['cases']:
        q = case.get('quality')
        if not q:
            continue
        row = {'case': case['caseName'], 'category': case['category'], 'complexity': case['complexity']}
        for cfg in CONFIGS:
            if cfg in q and q[cfg]:
                row[cfg] = q[cfg]['total']
                for crit in CRITERIA:
                    row[f'{cfg}_{crit}'] = q[cfg].get(crit, None)
            else:
                row[cfg] = None
        rows.append(row)
    return pd.DataFrame(rows)

def extract_judge_details(report):
    """Build a DataFrame with per-judge scores for agreement analysis."""
    rows = []
    for case in report['cases']:
        details = case.get('judgeDetails')
        if not details:
            continue
        for jd in details:
            judge = jd['judge']
            scores = jd['scores']
            for cfg in CONFIGS:
                if cfg in scores and scores[cfg]:
                    rows.append({
                        'case': case['caseName'],
                        'category': case['category'],
                        'judge': judge,
                        'config': cfg,
                        'total': scores[cfg]['total'],
                        **{c: scores[cfg].get(c, None) for c in CRITERIA},
                    })
    return pd.DataFrame(rows)

def extract_performance_df(report):
    """Build a DataFrame with duration per config per case."""
    rows = []
    for case in report['cases']:
        row = {'case': case['caseName'], 'category': case['category'], 'complexity': case['complexity']}
        for cr in case['configs']:
            cfg = cr['config']
            row[f'{cfg}_ms'] = cr['duration'] if not cr.get('error') else None
            row[f'{cfg}_chars'] = cr['outputChars'] if not cr.get('error') else None
        rows.append(row)
    return pd.DataFrame(rows)

# ─── Statistical tests ────────────────────────────────────────────────

def pairwise_wilcoxon(df):
    """Wilcoxon signed-rank test between all config pairs."""
    results = []
    active = [c for c in CONFIGS if c in df.columns and df[c].notna().sum() > 5]
    for i, c1 in enumerate(active):
        for c2 in active[i+1:]:
            mask = df[c1].notna() & df[c2].notna()
            if mask.sum() < 5:
                continue
            a, b = df.loc[mask, c1].values, df.loc[mask, c2].values
            try:
                stat, p = stats.wilcoxon(a, b)
            except ValueError:
                continue
            results.append({
                'config_A': CONFIG_SHORT[c1],
                'config_B': CONFIG_SHORT[c2],
                'mean_A': round(np.mean(a), 2),
                'mean_B': round(np.mean(b), 2),
                'diff': round(np.mean(a) - np.mean(b), 2),
                'W_stat': round(stat, 1),
                'p_value': round(p, 4),
                'significant': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns',
                'n': int(mask.sum()),
            })
    return pd.DataFrame(results)

# ─── Judge agreement analysis ─────────────────────────────────────────

def judge_agreement(jdf):
    """Analyze inter-judge agreement and flag large divergences."""
    if jdf.empty:
        return None, None

    # Correlation matrix between judges
    judges = jdf['judge'].unique()
    corr_rows = []
    for j1 in judges:
        for j2 in judges:
            if j1 >= j2:
                continue
            merged = jdf[jdf['judge'] == j1][['case', 'config', 'total']].merge(
                jdf[jdf['judge'] == j2][['case', 'config', 'total']],
                on=['case', 'config'], suffixes=(f'_{j1}', f'_{j2}')
            )
            if len(merged) < 3:
                continue
            r, p = stats.pearsonr(merged[f'total_{j1}'], merged[f'total_{j2}'])
            corr_rows.append({
                'judge_A': j1, 'judge_B': j2,
                'pearson_r': round(r, 3), 'p_value': round(p, 4),
                'n': len(merged),
                'mean_abs_diff': round((merged[f'total_{j1}'] - merged[f'total_{j2}']).abs().mean(), 2),
            })
    corr_df = pd.DataFrame(corr_rows)

    # Flag cases where judges diverge by > 3 points
    divergences = []
    for (case, cfg), grp in jdf.groupby(['case', 'config']):
        if len(grp) < 2:
            continue
        scores = grp['total'].values
        max_diff = float(np.max(scores) - np.min(scores))
        if max_diff >= 3.0:
            judge_scores = {r['judge']: r['total'] for _, r in grp.iterrows()}
            divergences.append({
                'case': case, 'config': cfg,
                'max_diff': round(max_diff, 1),
                **judge_scores,
            })
    div_df = pd.DataFrame(divergences).sort_values('max_diff', ascending=False) if divergences else pd.DataFrame()

    return corr_df, div_df

# ─── Charts ───────────────────────────────────────────────────────────

def chart_quality_by_config(df, out_dir):
    """Bar chart: average quality per config."""
    active = [c for c in CONFIGS if c in df.columns and df[c].notna().any()]
    means = [df[c].mean() for c in active]
    stds = [df[c].std() for c in active]
    labels = [CONFIG_SHORT[c] for c in active]

    colors = ['#4A90D9', '#E8913A', '#5BB55B', '#D94A4A', '#9B59B6', '#F1C40F']
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, means, yerr=stds, capsize=4, color=colors[:len(active)], edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Quality Score (avg of 3 judges)', fontsize=12)
    ax.set_title('Average Quality by Configuration', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 10)
    ax.axhline(y=np.mean(means), color='gray', linestyle='--', alpha=0.5, label=f'Overall avg: {np.mean(means):.1f}')
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{m:.1f}', ha='center', fontsize=11, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'quality_by_config.png'), dpi=150)
    plt.close()

def chart_quality_heatmap(df, out_dir):
    """Heatmap: config × category."""
    active = [c for c in CONFIGS if c in df.columns and df[c].notna().any()]
    categories = sorted(df['category'].unique())
    matrix = []
    for cat in categories:
        row = []
        for cfg in active:
            vals = df.loc[df['category'] == cat, cfg].dropna()
            row.append(vals.mean() if len(vals) > 0 else np.nan)
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(12, 7))
    data = np.array(matrix)
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=1, vmax=10)
    ax.set_xticks(range(len(active)))
    ax.set_xticklabels([CONFIG_SHORT[c] for c in active], fontsize=11)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=11)
    for i in range(len(categories)):
        for j in range(len(active)):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f'{data[i,j]:.1f}', ha='center', va='center', fontsize=10, fontweight='bold')
    plt.colorbar(im, label='Quality Score')
    ax.set_title('Quality by Category × Configuration', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'quality_heatmap.png'), dpi=150)
    plt.close()

def chart_radar(df, out_dir):
    """Radar chart: 5 criteria per config."""
    active = [c for c in CONFIGS if all(f'{c}_{cr}' in df.columns for cr in CRITERIA) and df[f'{c}_correctness'].notna().any()]
    if not active:
        return

    angles = np.linspace(0, 2 * np.pi, len(CRITERIA), endpoint=False).tolist()
    angles += angles[:1]
    labels = ['Correct.', 'Complete.', 'Edge Cases', 'Code Qual.', 'Readability']

    colors = ['#4A90D9', '#E8913A', '#5BB55B', '#D94A4A', '#9B59B6', '#F1C40F']
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    for idx, cfg in enumerate(active):
        values = [df[f'{cfg}_{cr}'].mean() for cr in CRITERIA]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=CONFIG_SHORT[cfg], color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 10)
    ax.set_title('Quality Criteria by Configuration', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'quality_radar.png'), dpi=150)
    plt.close()

def chart_performance(perf_df, out_dir):
    """Bar chart: average response time per config."""
    active = [c for c in CONFIGS if f'{c}_ms' in perf_df.columns and perf_df[f'{c}_ms'].notna().any()]
    means = [perf_df[f'{c}_ms'].mean() / 1000 for c in active]
    labels = [CONFIG_SHORT[c] for c in active]

    colors = ['#4A90D9', '#E8913A', '#5BB55B', '#D94A4A', '#9B59B6', '#F1C40F']
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, means, color=colors[:len(active)], edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Avg Response Time (seconds)', fontsize=12)
    ax.set_title('Average Latency by Configuration', fontsize=14, fontweight='bold')
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{m:.1f}s', ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'performance_latency.png'), dpi=150)
    plt.close()

def chart_judge_agreement(jdf, out_dir):
    """Scatter plots of judge pairs."""
    if jdf.empty:
        return
    judges = sorted(jdf['judge'].unique())
    if len(judges) < 2:
        return

    pairs = [(judges[i], judges[j]) for i in range(len(judges)) for j in range(i+1, len(judges))]
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5))
    if len(pairs) == 1:
        axes = [axes]

    for ax, (j1, j2) in zip(axes, pairs):
        merged = jdf[jdf['judge'] == j1][['case', 'config', 'total']].merge(
            jdf[jdf['judge'] == j2][['case', 'config', 'total']],
            on=['case', 'config'], suffixes=(f'_{j1}', f'_{j2}')
        )
        if merged.empty:
            continue
        ax.scatter(merged[f'total_{j1}'], merged[f'total_{j2}'], alpha=0.4, s=20)
        ax.plot([1, 10], [1, 10], 'k--', alpha=0.3)
        r, _ = stats.pearsonr(merged[f'total_{j1}'], merged[f'total_{j2}'])
        ax.set_xlabel(f'{j1} score', fontsize=11)
        ax.set_ylabel(f'{j2} score', fontsize=11)
        ax.set_title(f'{j1} vs {j2} (r={r:.2f})', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 10.5)
        ax.set_ylim(0, 10.5)

    plt.suptitle('Inter-Judge Agreement', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'judge_agreement.png'), dpi=150)
    plt.close()

# ─── Tables (Markdown + LaTeX) ────────────────────────────────────────

def table_summary_md(df, wilcox_df, out_dir):
    """Generate paper-ready Markdown tables."""
    active = [c for c in CONFIGS if c in df.columns and df[c].notna().any()]
    lines = ['# DEBATE EXT — Benchmark Results\n']

    # Overall summary
    lines.append('## Overall Quality Scores (avg of 3 judges)\n')
    lines.append('| Config | Mean | Std | Median | N |')
    lines.append('|--------|------|-----|--------|---|')
    for cfg in active:
        vals = df[cfg].dropna()
        lines.append(f'| {CONFIG_SHORT[cfg]} | {vals.mean():.2f} | {vals.std():.2f} | {vals.median():.1f} | {len(vals)} |')

    # By category
    lines.append('\n## Quality by Category\n')
    header = '| Category | N |' + '|'.join(f' {CONFIG_SHORT[c]} ' for c in active) + '| Best |'
    lines.append(header)
    lines.append('|' + '---|' * (len(active) + 3))
    for cat in sorted(df['category'].unique()):
        sub = df[df['category'] == cat]
        row = f'| {cat} | {len(sub)} |'
        best_score, best_cfg = 0, '-'
        for cfg in active:
            vals = sub[cfg].dropna()
            m = vals.mean() if len(vals) > 0 else 0
            row += f' {m:.1f} |'
            if m > best_score:
                best_score, best_cfg = m, CONFIG_SHORT[cfg]
        row += f' {best_cfg} |'
        lines.append(row)

    # Wilcoxon tests
    if not wilcox_df.empty:
        lines.append('\n## Statistical Tests (Wilcoxon signed-rank)\n')
        lines.append('| Config A | Config B | Mean A | Mean B | Diff | p-value | Sig. | N |')
        lines.append('|----------|----------|--------|--------|------|---------|------|---|')
        for _, r in wilcox_df.iterrows():
            lines.append(f"| {r['config_A']} | {r['config_B']} | {r['mean_A']} | {r['mean_B']} | {r['diff']:+.2f} | {r['p_value']:.4f} | {r['significant']} | {r['n']} |")

    with open(os.path.join(out_dir, 'results_summary.md'), 'w') as f:
        f.write('\n'.join(lines))

def table_judges_md(corr_df, div_df, out_dir):
    """Generate judge agreement tables."""
    lines = ['# Judge Agreement Analysis\n']

    if corr_df is not None and not corr_df.empty:
        lines.append('## Inter-Judge Correlation\n')
        lines.append('| Judge A | Judge B | Pearson r | p-value | Mean Abs Diff | N |')
        lines.append('|---------|---------|-----------|---------|---------------|---|')
        for _, r in corr_df.iterrows():
            lines.append(f"| {r['judge_A']} | {r['judge_B']} | {r['pearson_r']:.3f} | {r['p_value']:.4f} | {r['mean_abs_diff']:.2f} | {r['n']} |")

    if div_df is not None and not div_df.empty:
        lines.append(f'\n## Divergences > 3 points ({len(div_df)} cases)\n')
        judge_cols = [c for c in div_df.columns if c not in ('case', 'config', 'max_diff')]
        header = '| Case | Config | Max Diff |' + '|'.join(f' {j} ' for j in judge_cols) + '|'
        lines.append(header)
        lines.append('|' + '---|' * (len(judge_cols) + 3))
        for _, r in div_df.head(30).iterrows():
            row = f"| {r['case']} | {r['config']} | {r['max_diff']} |"
            for j in judge_cols:
                row += f" {r.get(j, '-')} |"
            lines.append(row)
    else:
        lines.append('\n## No major divergences (all judges within 3 points)\n')

    with open(os.path.join(out_dir, 'judge_agreement.md'), 'w') as f:
        f.write('\n'.join(lines))

# ─── Main ─────────────────────────────────────────────────────────────

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else find_latest_report()
    print(f"Loading report: {path}")
    report = load_report(path)

    meta = report['meta']
    print(f"  Date: {meta['date']}")
    print(f"  Cases: {meta['selectedCases']} / Configs: {', '.join(meta['selectedConfigs'])}")
    print(f"  Claude: {meta['claudeVersion']} / OpenAI: {meta['openaiModel']}")

    # Output directory
    out_dir = os.path.join(os.path.dirname(path), 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    # Extract data
    df = extract_quality_df(report)
    jdf = extract_judge_details(report)
    perf_df = extract_performance_df(report)

    if df.empty:
        print("ERROR: No quality data found in report.")
        sys.exit(1)

    print(f"  Quality data: {len(df)} cases with scores")
    print(f"  Judge details: {len(jdf)} individual judge scores")

    # Statistical tests
    print("\n--- Statistical Tests ---")
    wilcox_df = pairwise_wilcoxon(df)
    if not wilcox_df.empty:
        sig = wilcox_df[wilcox_df['significant'] != 'ns']
        print(f"  {len(wilcox_df)} pairwise comparisons, {len(sig)} significant")
        for _, r in sig.iterrows():
            print(f"    {r['config_A']} vs {r['config_B']}: {r['diff']:+.2f} (p={r['p_value']:.4f} {r['significant']})")
    else:
        print("  Not enough data for statistical tests (need >5 paired cases)")

    # Judge agreement
    print("\n--- Judge Agreement ---")
    corr_df, div_df = judge_agreement(jdf)
    if corr_df is not None and not corr_df.empty:
        for _, r in corr_df.iterrows():
            print(f"  {r['judge_A']} vs {r['judge_B']}: r={r['pearson_r']:.3f}, mean diff={r['mean_abs_diff']:.2f}")
    if div_df is not None and not div_df.empty:
        print(f"  {len(div_df)} cases with judge divergence > 3 points (audit needed)")

    # Generate charts
    print("\n--- Generating Charts ---")
    chart_quality_by_config(df, out_dir)
    print("  quality_by_config.png")
    chart_quality_heatmap(df, out_dir)
    print("  quality_heatmap.png")
    chart_radar(df, out_dir)
    print("  quality_radar.png")
    chart_performance(perf_df, out_dir)
    print("  performance_latency.png")
    chart_judge_agreement(jdf, out_dir)
    print("  judge_agreement.png")

    # Generate tables
    print("\n--- Generating Tables ---")
    table_summary_md(df, wilcox_df, out_dir)
    print("  results_summary.md")
    table_judges_md(corr_df, div_df, out_dir)
    print("  judge_agreement.md")

    print(f"\nAll outputs saved to: {out_dir}/")

if __name__ == '__main__':
    main()
