import re

SRC = r'C:\Users\wkr20\WorkBuddy\2026-08-01-22-09-01\virtual-cell\build_docs.js'

# 旧编号 -> 新编号（按正文首次出现顺序）
MAP = {1:1, 10:2, 11:3, 12:4, 6:5, 2:6, 3:7, 5:8, 8:9, 13:10, 9:11, 7:12, 4:13}
# 新编号对应的旧编号（参考文献物理顺序重排）
NEW_ORDER_OLD = [1, 10, 11, 12, 6, 2, 3, 5, 8, 13, 9, 7, 4]

INTRO_EN = ("Predicting transcriptomic responses to molecular perturbations has emerged as a central challenge in "
            "AI-driven biology. UniPert-G2CP (Li et al., Cell, 2026) [1] unified genetic and chemical perturbation "
            "prediction in a single contrastive-embedding space across five cancer cell lines. Several approaches have "
            "been proposed for related tasks: scGen [2] uses variational autoencoders for single-cell perturbation "
            "response; CPA [3] uses an autoencoder framework for combinatorial perturbation prediction; ChemPert [4] "
            "predicts drug-induced transcriptomic changes via a multi-task model; and GEARS [5] introduced a "
            "graph-based framework integrating gene co-expression. To date, no independent reimplementation or "
            "large-scale extension of UniPert-G2CP has been reported.")
INTRO_CN = ("预测分子扰动后的转录组响应已成为 AI 驱动生物学中的核心挑战。UniPert-G2CP（Li 等，Cell，2026）[1] "
            "在单一对比嵌入空间中统一了五种癌细胞系上的遗传与化学扰动预测。针对相关任务已有多种方法被提出："
            "scGen [2] 使用变分自编码器预测单细胞扰动响应；CPA [3] 使用自编码器框架进行组合扰动预测；"
            "ChemPert [4] 通过多任务模型预测药物诱导的转录组变化；GEARS [5] 引入了整合基因共表达网络的图框架。"
            "迄今为止，尚未有对 UniPert-G2CP 的独立重新实现或大规模扩展的报道。")

def renumber(text):
    """一次性映射 [n] 或 [n,m] 引用（防链式替换）"""
    def repl(m):
        inner = m.group(1)
        nums = [int(x.strip()) for x in inner.split(',')]
        return '[' + ','.join(str(MAP.get(n, n)) for n in nums) + ']'
    return re.sub(r'\[(\d+(?:\s*,\s*\d+)*)\]', repl, text)

with open(SRC, encoding='utf-8') as f:
    lines = f.readlines()

# 1) 定位参考文献区（EN: h2("References") .. ];  CN: h2("参考文献") .. ];）
ref_ranges = []  # (start_idx, end_idx_exclusive)
for i, ln in enumerate(lines):
    if 'h2("References")' in ln or 'h2("参考文献")' in ln:
        j = i + 1
        while j < len(lines) and '];' not in lines[j]:
            j += 1
        ref_ranges.append((i + 1, j))  # 参考文献段在 h2 之后
        # 注意: h2 行本身不参与

print('参考文献区(行号从1起):', [(a + 1, b + 1) for a, b in ref_ranges])

# 2) 处理参考文献区：按 NEW_ORDER_OLD 重排 + 前缀编号 1-13
new_lines = []
cursor = 0
for start, end in ref_ranges:
    # 非参考文献区：正文映射
    for ln in lines[cursor:start - 1]:  # 到 h2 行前
        new_lines.append(renumber(ln))
    # h2 行本身（不映射）
    new_lines.append(lines[start - 1])
    # 参考文献区 13 段，按新顺序重排
    refs_block = lines[start:end]
    # 提取每段的旧编号
    old_nums = []
    for ln in refs_block:
        m = re.match(r'^.*\[(\d+)\]', ln)
        old_nums.append(int(m.group(1)) if m else None)
    print('  该区旧编号顺序:', old_nums)
    # 按 NEW_ORDER_OLD 生成新顺序
    idx_by_old = {old: pos for pos, old in enumerate(old_nums)}
    for new_idx, old_idx in enumerate(NEW_ORDER_OLD, start=1):
        if old_idx in idx_by_old:
            ln = refs_block[idx_by_old[old_idx]]
            ln = re.sub(r'\[(\d+)\]', f'[{new_idx}]', ln, count=1)
            new_lines.append(ln)
        else:
            print(f'  !! 旧编号 {old_idx} 未找到')
    cursor = end
# 尾部
for ln in lines[cursor:]:
    new_lines.append(renumber(ln))

# 3) 引言第一段重写（在映射后的文本中替换）
text = ''.join(new_lines)
# EN 引言：找到旧文本片段整体替换
en_old_pat = r'Predicting transcriptomic responses to molecular perturbations has emerged as a central challenge in AI-driven biology\.(.*?)To date, no independent reimplementation or large-scale extension of UniPert-G2CP has been reported\.'
en_new_full = INTRO_EN
m = re.search(en_old_pat, text, re.S)
if m:
    # 替换整句（从 'Predicting transcriptomic' 到 'has been reported.'）
    start = m.start()
    end = m.end()
    text = text[:start] + en_new_full + text[end:]
    print('EN 引言已重写')
else:
    print('!! EN 引言未找到')

cn_old_pat = r'预测分子扰动后的转录组响应已成为 AI 驱动生物学中的核心挑战。.*?尚未有对 UniPert-G2CP 的独立重新实现或大规模扩展的报道。'
m = re.search(cn_old_pat, text, re.S)
if m:
    text = text[:m.start()] + INTRO_CN + text[m.end():]
    print('CN 引言已重写')
else:
    print('!! CN 引言未找到')

with open(SRC, 'w', encoding='utf-8') as f:
    f.write(text)
print('build_docs.js 已保存')

# 4) 验证
import subprocess
r = subprocess.run([r'C:\Users\wkr20\.workbuddy\binaries\node\versions\20.18.0\node.exe', '--check', SRC],
                   capture_output=True, text=True)
print('node --check:', 'OK' if r.returncode == 0 else r.stderr[:300])
