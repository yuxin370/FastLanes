import pandas as pd
import re
import os

def extract_expression(expr_str):
    """
    从 expression 字符串中提取表达式名称
    """
    match = re.search(r'\[EXP_([A-Z0-9_]+)\]', expr_str)
    if match:
        return "EXP_" + match.group(1)
    else:
        return "UNKNOWN"

def calculate_compression_ratio(row):
    """根据 data_type 和 bytes_per_value 计算压缩比"""
    orig_bytes = 2 if row['data_type'] == 'INT16' else 1
    bpv = row['bytes_per_value']
    if bpv == 0:
        return float('inf')
    return orig_bytes / bpv

def main(csv_path='fastlanes_detailed.csv'):
    if not os.path.exists(csv_path):
        print(f"错误：文件 {csv_path} 不存在！")
        return

    # 读取数据
    df = pd.read_csv(csv_path, 
                     usecols=['table_name', 'data_type', 'expression', 'bytes_per_value'],
                     skipinitialspace=True)

    # 清理 expression 字段
    df['expr_name'] = df['expression'].apply(extract_expression)

    # 计算压缩比
    df['compression_ratio'] = df.apply(calculate_compression_ratio, axis=1)

    datasets = df['table_name'].unique()
    
    print("="*80)
    print("压缩分析报告")
    print("="*80)

    all_summary = []

    for dataset in sorted(datasets):
        ddf = df[df['table_name'] == dataset].copy()
        total_cols = len(ddf)
        print(f"\n📊 数据集: {dataset.upper()} (共 {total_cols} 列)")

        # 表达式使用统计（显示占比）
        expr_counts = ddf['expr_name'].value_counts()
        print("  表达式使用频次:")
        for expr, count in expr_counts.items():
            pct = count / total_cols * 100
            print(f"    {expr}: {count} 次 ({pct:.1f}%)")

        # 按表达式分组统计压缩比
        expr_stats = ddf.groupby('expr_name')['compression_ratio'].agg(
            count='size',
            mean_cr='mean',
            min_cr='min',
            max_cr='max'
        ).round(2)

        print("  压缩比统计 (按表达式):")
        for expr, stats in expr_stats.iterrows():
            mean_cr = stats['mean_cr']
            min_cr = stats['min_cr']
            max_cr = stats['max_cr']
            mean_str = f"{mean_cr:.2f}" if mean_cr != float('inf') else "∞"
            min_str = f"{min_cr:.2f}" if min_cr != float('inf') else "∞"
            max_str = f"{max_cr:.2f}" if max_cr != float('inf') else "∞"
            print(f"    {expr}: 平均={mean_str}x, 范围=[{min_str}, {max_str}]x")

        # 整体统计
        overall_mean = ddf['compression_ratio'].replace([float('inf')], pd.NA).mean()
        overall_mean_str = f"{overall_mean:.2f}x" if pd.notna(overall_mean) else "N/A"
        print(f"  整体平均压缩比: {overall_mean_str}")

        all_summary.append({
            'dataset': dataset,
            'total_columns': total_cols,
            'overall_mean_cr': overall_mean
        })

    # ================================
    # 🔷 新增：不分数据集的全局总计
    # ================================
    total_columns_global = len(df)
    print("\n" + "="*80)
    print("🌍 全局总计（不分数据集）")
    print("="*80)
    print(f"总列数: {total_columns_global}")

    # 表达式总频次与占比
    global_expr_counts = df['expr_name'].value_counts()
    print("\n表达式使用总频次与占比:")
    for expr, count in global_expr_counts.items():
        pct = count / total_columns_global * 100
        print(f"  {expr}: {count} 次 ({pct:.2f}%)")

    # 全局压缩比统计（按表达式）
    global_expr_stats = df.groupby('expr_name')['compression_ratio'].agg(
        count='size',
        mean_cr='mean',
        min_cr='min',
        max_cr='max'
    ).round(2)

    print("\n全局压缩比统计（按表达式）:")
    for expr, stats in global_expr_stats.iterrows():
        mean_cr = stats['mean_cr']
        min_cr = stats['min_cr']
        max_cr = stats['max_cr']
        mean_str = f"{mean_cr:.2f}" if mean_cr != float('inf') else "∞"
        min_str = f"{min_cr:.2f}" if min_cr != float('inf') else "∞"
        max_str = f"{max_cr:.2f}" if max_cr != float('inf') else "∞"
        print(f"  {expr}: 平均={mean_str}x, 范围=[{min_str}, {max_str}]x")

    # 全局整体平均压缩比（排除 inf）
    global_overall_mean = df['compression_ratio'].replace([float('inf')], pd.NA).mean()
    global_mean_str = f"{global_overall_mean:.2f}x" if pd.notna(global_overall_mean) else "N/A"
    print(f"\n全局整体平均压缩比: {global_mean_str}")

    # ================================
    # 原有：各数据集汇总 + 膨胀警告
    # ================================
    print("\n" + "="*80)
    print("📈 各数据集汇总")
    print("="*80)
    for item in all_summary:
        ds = item['dataset']
        mean_cr = f"{item['overall_mean_cr']:.2f}x" if pd.notna(item['overall_mean_cr']) else "N/A"
        print(f"{ds.upper():<10} | 总列数: {item['total_columns']:<3} | 平均压缩比: {mean_cr}")

    # 检查膨胀（CR < 1）
    df_inflated = df[df['compression_ratio'] < 1.0]
    if not df_inflated.empty:
        print(f"\n⚠️  注意：发现 {len(df_inflated)} 列压缩后体积增大（CR < 1）:")
        for _, row in df_inflated.iterrows():
            print(f"    数据集: {row['table_name']}, 表达式: {row['expr_name']}, CR: {row['compression_ratio']:.2f}x")

    print("\n分析完成。")

if __name__ == "__main__":
    main('fastlanes_detailed.csv')