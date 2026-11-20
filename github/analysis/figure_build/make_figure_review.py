"""Rebuild the author figure review (Chinese) for the current 7-figure set.

Figure set = the 7 main figures of the submission package (Figure 1-7).
Captions are read from author_notes/captions_from_package.json, which holds the
verbatim caption text of the current manuscript (7 entries, numbered 1-7). Keeping
a single source means this review can never drift from the manuscript again.

The figures/ directory still contains three superseded files under their old names
(Figure5_lincs_heterogeneity, Figure6_depmap_mechanism_control,
Figure7_gtex_reference_matching). They are listed on purpose and skipped, because
historical scripts still reference them. Do not delete them blindly; the current
Figure 5-7 are Figure5_depmap_mechanism_control, Figure6_gtex_reference_matching and
Figure7_lincs_gene_mean_residualized.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

R = Path(__file__).resolve().parents[1]
F, A, P = R / "figures", R / "author_notes", R / "preview"
CJK = "Microsoft YaHei"

# Current main figures, in manuscript order 1-7. Names must match submission_package/figures/.
FILES = ["Figure1_design", "Figure2_fixed_output", "Figure3_training_evaluation",
         "Figure4_ranking_consequence", "Figure5_depmap_mechanism_control",
         "Figure6_gtex_reference_matching", "Figure7_lincs_gene_mean_residualized"]

# Superseded files still sitting in figures/ under their pre-7-figure names.
RETIRED = {"Figure5_lincs_heterogeneity", "Figure6_depmap_mechanism_control",
           "Figure7_gtex_reference_matching"}


def load_captions():
    """Read the 7 manuscript captions verbatim; never hand-write them here.

    Returns (title, body, full) per figure: `body` drops the "图 N　" prefix for the
    per-figure page (the number is already a bold heading there), while `full` keeps it
    so the appendix section can be pasted straight back into the manuscript.
    """
    raw = json.loads((A / "captions_from_package.json").read_text(encoding="utf8"))
    if len(raw) != len(FILES):
        raise SystemExit(f"expected {len(FILES)} captions, found {len(raw)}")
    caps = []
    for i, cap in enumerate(raw, 1):
        head = f"图 {i}　"
        if not cap.startswith(head):
            raise SystemExit(f"caption {i} does not start with {head!r}: {cap[:20]!r}")
        caps.append((f"图 {i}", cap[len(head):], cap))
    return caps


CAPS = load_captions()

# Guard against silently drifting back to the old 8/7-plus-supplementary naming: the
# current set and the retired set must not overlap, and every current figure must exist.
assert not (RETIRED & set(FILES)), f"current/retired overlap: {RETIRED & set(FILES)}"
_missing = [n for n in FILES if not (F / (n + ".png")).is_file()]
if _missing:
    raise SystemExit(f"missing figure files: {_missing}")


CHECKLIST = [
    ("图注与主稿同源", "本审阅件的 7 条图注逐字取自 author_notes/captions_from_package.json，"
     "该文件内容等于现行主稿的图注原文，脚本不再自带任何图注文本。"),
    ("图文件名为现行 7 图体系", "Figure1–7 与投稿包 figures/ 一一对应；"
     "figures/ 下另有 3 个旧名文件（Figure5_lincs_heterogeneity、"
     "Figure6_depmap_mechanism_control、Figure7_gtex_reference_matching）已退役，"
     "仍被历史脚本引用，故保留但不使用。"),
    ("图 4 口径已统一为全量 2,037", "A/B/C 面板与 D 面板的 Global 行同用全量 2,037 药物名单"
     "（top-10 重合 3、Spearman 0.5675）；D 的疾病上下文行为 1,399 共同可评子集，"
     "N 已逐行标在 y 轴标签上。"),
    ("跨尺度数值不做直接相减", "raw 与 residualized 尺度的相关系数不可比大小；"
     "匹配结构的结论只表述为「Ridge 仍保留 vs 冻结归零」，不使用「增强 / 放大」。"),
    ("正文与图一致", "正文 7 条图注描述面板构成与统计口径，不复述面板内的具体数值，"
     "因此改图不改图注。"),
]


def wrap_cjk(text, width=50):
    out, line, n = [], "", 0
    for ch in text:
        w = 2 if ord(ch) > 0x2E80 else 1
        if ch == "\n":
            out.append(line); line, n = "", 0
            continue
        if n + w > width:
            out.append(line); line, n = "", 0
        line += ch; n += w
    if line:
        out.append(line)
    return "\n".join(out)


def disp(t):
    return t.replace("10⁻³", "1e-3").replace("⁻", "^-")


def build_docx():
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = CJK
    st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
    h = doc.add_paragraph(); r = h.add_run("图表审阅版（现行 7 张主图）")
    r.font.size = Pt(16); r.font.bold = True
    sub = doc.add_paragraph(); rs = sub.add_run(
        "图 1–7 与投稿包 figures/ 目录一一对应。7 条图注逐字取自 "
        "author_notes/captions_from_package.json（即现行主稿的图注原文），"
        "与主稿同源、不会漂移。文末附全部 7 条图注原文。")
    rs.font.size = Pt(9.5)
    for name, (title, body, _full) in zip(FILES, CAPS):
        p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(14)
        rr = p.add_run(title); rr.font.bold = True; rr.font.size = Pt(11.5)
        img = doc.add_paragraph(); img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        img.add_run().add_picture(str(F / (name + ".png")), width=Cm(16.5))
        cp = doc.add_paragraph(); cr = cp.add_run(disp(body)); cr.font.size = Pt(9.5)
    doc.add_page_break()
    p = doc.add_paragraph(); r = p.add_run("图注与口径核对清单")
    r.font.bold = True; r.font.size = Pt(13)
    for k, v in CHECKLIST:
        para = doc.add_paragraph(style="List Bullet")
        a = para.add_run("√ " + k + "："); a.font.bold = True; a.font.size = Pt(10)
        b = para.add_run(v); b.font.size = Pt(10)
    doc.add_paragraph()
    p = doc.add_paragraph(); r = p.add_run("图 1–7 图注原文（取自主稿，可直接回填）")
    r.font.bold = True; r.font.size = Pt(13)
    for title, _body, full in CAPS:
        para = doc.add_paragraph(); rr = para.add_run(disp(full)); rr.font.size = Pt(10)
        para.paragraph_format.space_after = Pt(8)
    doc.save(A / "Figure_Review_CN.docx")
    print("wrote", A / "Figure_Review_CN.docx")


def build_pdf():
    matplotlib.rcParams["font.family"] = CJK
    matplotlib.rcParams["axes.unicode_minus"] = False
    out = P / "Figure_Review_CN.pdf"
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.94, "图表审阅版（现行 7 张主图）", fontsize=19, weight="bold")
        fig.text(0.07, 0.90, "Figure 1–7 · GigaScience 投稿用", fontsize=12, color="#444444")
        body = ("图 1–7 与投稿包 figures/ 目录一一对应。\n\n"
                "· 图 1：分析框架与三套数据的研究内容。\n"
                "· 图 2：预测结果固定时两套评价矩阵产生的性能差异。\n"
                "· 图 3：评价矩阵更换改变两个训练方案的性能比较结果。\n"
                "· 图 4：评价矩阵更换、抽样波动与选择深度对药物排序与选择的影响。\n"
                "· 图 5：DepMap 相关系数计算方式与基因均值去除。\n"
                "· 图 6：GTEx 简单基线预测器的训练矩阵与评价矩阵匹配现象。\n"
                "· 图 7：LINCS 去除基因均值前后评价矩阵差异的变化。\n\n"
                "图注来源：author_notes/captions_from_package.json，\n"
                "逐字等于现行主稿的 7 条图注，脚本不自带图注文本。\n"
                "figures/ 下另有 3 个旧名文件已退役（仍被历史脚本引用，保留不用）。\n"
                "本 PDF 与 author_notes/Figure_Review_CN.docx 内容一致。")
        fig.text(0.07, 0.84, wrap_cjk(body, 46), fontsize=10.5, va="top", linespacing=1.7)
        pdf.savefig(fig); plt.close(fig)

        for name, (title, body, _full) in zip(FILES, CAPS):
            im = Image.open(F / (name + ".png"))
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.06, 0.965, disp(title), fontsize=12.5, weight="bold", va="top")
            ax = fig.add_axes([0.06, 0.26, 0.88, 0.68])
            ax.imshow(im); ax.axis("off")
            fig.text(0.06, 0.225, wrap_cjk(disp(body), 56), fontsize=9.5, va="top", linespacing=1.7)
            pdf.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.06, 0.965, "图注与口径核对清单", fontsize=15, weight="bold", va="top")
        y = 0.925
        for k, v in CHECKLIST:
            fig.text(0.06, y, "√ " + k, fontsize=10.5, weight="bold", va="top")
            y -= 0.028
            txt = wrap_cjk(v, 60)
            fig.text(0.08, y, txt, fontsize=9, va="top", linespacing=1.55, color="#333333")
            y -= 0.028 * (txt.count("\n") + 1) + 0.018
        pdf.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.06, 0.965, "图 1–7 图注原文（取自主稿）", fontsize=15, weight="bold", va="top")
        yy = 0.91
        for title, _body, full in CAPS:
            txt = wrap_cjk(disp(full), 62)
            fig.text(0.06, yy, txt, fontsize=9.5, va="top", linespacing=1.75)
            yy -= 0.042 * (txt.count("\n") + 2)
        pdf.savefig(fig); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    build_docx()
    build_pdf()
