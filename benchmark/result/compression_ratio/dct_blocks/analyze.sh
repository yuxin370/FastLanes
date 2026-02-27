#!/bin/bash

CSV_FILE="${1:-fastlanes_detailed.csv}"

if [ ! -f "$CSV_FILE" ]; then
    echo "错误：文件 $CSV_FILE 不存在！" >&2
    exit 1
fi

# 使用 awk 安全解析列名（去除前后空格）
awk -F',' '
NR == 1 {
    for (i = 1; i <= NF; i++) {
        gsub(/^[ \t]+|[ \t]+$/, "", $i)
        cols[$i] = 1
    }
    if (!( "table_name" in cols && "data_type" in cols && "expression" in cols && "bytes_per_value" in cols )) {
        print "错误：CSV 缺少必要列（table_name, data_type, expression, bytes_per_value）" > "/dev/stderr"
        exit 1
    }
    exit 0
}
' "$CSV_FILE" || exit 1

# 临时文件
TMP_DATA=$(mktemp)
trap 'rm -f "$TMP_DATA"' EXIT

# 提取并预处理数据：table_name, expr_name, compression_ratio
awk -F',' -v OFS=',' '
BEGIN {
    tn = -1; dt = -1; expr = -1; bpv = -1
}
NR == 1 {
    for (i = 1; i <= NF; i++) {
        gsub(/^[ \t]+|[ \t]+$/, "", $i)
        if ($i == "table_name") tn = i
        else if ($i == "data_type") dt = i
        else if ($i == "expression") expr = i
        else if ($i == "bytes_per_value") bpv = i
    }
    next
}
{
    if (tn == -1 || dt == -1 || expr == -1 || bpv == -1) exit 1
    table = $tn
    dtype = $dt
    exp_str = $expr
    bpv_val = $bpv

    # 提取 expr_name: [EXP_XXX] -> EXP_XXX
    expr_name = "UNKNOWN"
    if (match(exp_str, /\[EXP_([A-Z0-9_]+)\]/, arr)) {
        expr_name = "EXP_" arr[1]
    }

    # 原始字节数
    orig_bytes = (dtype == "INT16") ? 2 : 1

    # 转换 bpv 为数值（跳过非数字）
    if (bpv_val ~ /^[0-9]+(\.[0-9]+)?$/) {
        cr = orig_bytes / (bpv_val + 0)
    } else {
        cr = "inf"
    }

    print table, expr_name, cr
}
' "$CSV_FILE" > "$TMP_DATA"

# 辅助函数：格式化压缩比（处理 inf）
format_cr() {
    if [[ "$1" == "inf" ]]; then
        echo "∞"
    else
        awk "BEGIN { printf \"%.2f\", $1 }"
    fi
}

# 获取所有数据集
datasets=$(cut -d',' -f1 "$TMP_DATA" | sort -u)
total_global=$(wc -l < "$TMP_DATA")

echo "================================================================================"
echo "压缩分析报告"
echo "================================================================================"

# 存储全局 summary 用于最后输出
summary_lines=()

# 分析每个数据集
for ds in $datasets; do
    awk -F',' -v ds="$ds" '$1 == ds' "$TMP_DATA" > "${TMP_DATA}.${ds}"
    ncol=$(wc -l < "${TMP_DATA}.${ds}")

    echo
    echo "📊 数据集: ${ds^^} (共 $ncol 列)"

    # 表达式频次（带百分比）
    echo "  表达式使用频次:"
    cut -d',' -f2 "${TMP_DATA}.${ds}" | sort | uniq -c | sort -nr | while read count name; do
        pct=$(awk "BEGIN { printf \"%.1f\", ($count / $ncol) * 100 }")
        echo "    $name: $count 次 ($pct%)"
    done

    # 按表达式统计压缩比（mean/min/max）
    echo "  压缩比统计 (按表达式):"
    awk -F',' -v ds="$ds" '
    {
        e = $2; cr = $3
        if (cr == "inf") {
            cr_val = 999999
            has_inf[e] = 1
        } else {
            cr_val = cr + 0
            sum[e] += cr_val
            count[e]++
            if (e in min) {
                if (cr_val < min[e]) min[e] = cr_val
            } else {
                min[e] = cr_val
            }
            if (cr_val > max[e]) max[e] = cr_val
        }
    }
    END {
        for (e in count) {
            mean = sum[e] / count[e]
            min_s = (min[e] == 999999) ? "∞" : sprintf("%.2f", min[e])
            max_s = (max[e] == 999999) ? "∞" : sprintf("%.2f", max[e])
            mean_s = sprintf("%.2f", mean)
            print "    " e ": 平均=" mean_s "x, 范围=[" min_s ", " max_s "]x"
        }
        for (e in has_inf) {
            if (!(e in count)) {
                print "    " e ": 平均=∞x, 范围=[∞, ∞]x"
            }
        }
    }' "${TMP_DATA}.${ds}"

    # 整体平均（排除 inf）
    overall=$(awk -F',' '
    BEGIN { s=0; n=0 }
    {
        if ($3 != "inf") { s += $3 + 0; n++ }
    }
    END { if (n>0) printf "%.2f", s/n; else printf "N/A" }
    ' "${TMP_DATA}.${ds}")
    echo "  整体平均压缩比: ${overall}x"

    summary_lines+=("$ds|$ncol|$overall")
done

# ================================
# 🌍 全局总计
# ================================
echo
echo "================================================================================"
echo "🌍 全局总计（不分数据集）"
echo "================================================================================"
echo "总列数: $total_global"

# 全局表达式频次
echo
echo "表达式使用总频次与占比:"
cut -d',' -f2 "$TMP_DATA" | sort | uniq -c | sort -nr | while read count name; do
    pct=$(awk "BEGIN { printf \"%.2f\", ($count / $total_global) * 100 }")
    echo "  $name: $count 次 ($pct%)"
done

# 全局压缩比统计（按表达式）
echo
echo "全局压缩比统计（按表达式）:"
awk -F',' '
{
    e = $2; cr = $3
    if (cr == "inf") {
        inf[e]++
    } else {
        crv = cr + 0
        sum[e] += crv
        cnt[e]++
        if (e in min) {
            if (crv < min[e]) min[e] = crv
        } else {
            min[e] = crv
        }
        if (crv > max[e]) max[e] = crv
    }
}
END {
    for (e in cnt) {
        mean = sum[e] / cnt[e]
        min_s = sprintf("%.2f", min[e])
        max_s = sprintf("%.2f", max[e])
        mean_s = sprintf("%.2f", mean)
        print "  " e ": 平均=" mean_s "x, 范围=[" min_s ", " max_s "]x"
    }
}' "$TMP_DATA"

# 全局整体平均
global_mean=$(awk -F',' '
BEGIN { s=0; n=0 }
{
    if ($3 != "inf") { s += $3 + 0; n++ }
}
END { if (n>0) printf "%.2f", s/n; else printf "N/A" }
' "$TMP_DATA")
echo
echo "全局整体平均压缩比: ${global_mean}x"

# ================================
# 📈 各数据集汇总
# ================================
echo
echo "================================================================================"
echo "📈 各数据集汇总"
echo "================================================================================"
for line in "${summary_lines[@]}"; do
    IFS='|' read -r ds ncol mean <<< "$line"
    printf "%-10s | 总列数: %-3s | 平均压缩比: %sx\n" "${ds^^}" "$ncol" "$mean"
done

# ================================
# ⚠️ 膨胀警告（CR < 1）
# ================================
inflated_count=$(awk -F',' '$3 != "inf" && ($3 + 0) < 1.0 { print }' "$TMP_DATA" | wc -l)
if [ "$inflated_count" -gt 0 ]; then
    echo
    echo "⚠️  注意：发现 $inflated_count 列压缩后体积增大（CR < 1）:"
    awk -F',' '$3 != "inf" && ($3 + 0) < 1.0 {
        cr_f = sprintf("%.2f", $3 + 0)
        printf "    数据集: %s, 表达式: %s, CR: %sx\n", $1, $2, cr_f
    }' "$TMP_DATA"
fi

echo
echo "分析完成。"