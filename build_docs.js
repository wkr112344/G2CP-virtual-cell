// build_docs.js -- Generate EN Word + CN Word for bioRxiv submission
const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber } = require("docx");

const NODE_PATH = "C:/Users/wkr20/.workbuddy/binaries/node/workspace/node_modules";
// Ensure docx is resolvable
let docx;
try { docx = require("docx"); } catch(e) { docx = require(NODE_PATH + "/docx"); }

const Q = String.fromCharCode; // no unicode bullets
const DXA = (inch) => Math.round(inch * 1440);
const A4_W = 11906, A4_H = 16838;
const MARGIN = DXA(1.0);
const CONTENT_W = A4_W - 2 * MARGIN;

function hr(width) {
  return new Paragraph({ spacing: { after: 80 }, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "CCCCCC", space: 1 } } });
}

function h1(text) { return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 240, after: 120 }, children: [new TextRun({ text, bold: true, size: 32 })] }); }
function h2(text) { return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 100 }, children: [new TextRun({ text, bold: true, size: 28 })] }); }
function h3(text) { return new Paragraph({ heading: HeadingLevel.HEADING_3, spacing: { before: 160, after: 80 }, children: [new TextRun({ text, bold: true, size: 26 })] }); }
function p(text)  { return new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text, size: 22 })] }); }
function pi(text) { return new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text, italics: true, size: 22 })] }); }
function pb(text) { return new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text, bold: true, size: 22 })] }); }
function pc(text) { return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: [new TextRun({ text, size: 22 })] }); }

function tbl(headers, rows) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "999999" };
  const bd = { top: border, bottom: border, left: border, right: border };
  const n = headers.length;
  const colW = Math.floor(CONTENT_W / n);
  const cols = Array(n).fill(colW);
  const hdrRow = new TableRow({ children: headers.map(h => new TableCell({
    borders: bd, width: { size: colW, type: WidthType.DXA },
    shading: { fill: "E8E8E8", type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text: h, bold: true, size: 20 })] })]
  }))});
  const dataRows = rows.map(row => new TableRow({ children: row.map(cell => new TableCell({
    borders: bd, width: { size: colW, type: WidthType.DXA },
    margins: { top: 50, bottom: 50, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text: String(cell), size: 20 })] })]
  }))}));
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: cols, rows: [hdrRow, ...dataRows] });
}

const foot = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Page ", size: 18 }), new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] });

// ==================== ENGLISH ====================
const enChildren = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 4 }, children: [new TextRun({ text: "Reproducing and Scaling UniPert-G2CP:", bold: true, size: 34 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 4 }, children: [new TextRun({ text: "A 162-Cell-Line Genome-Wide Platform for Perturbation-to-Phenotype Prediction", bold: true, size: 30 })] }),
  pc(""),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Kairui Wei\u00B9*, Lijing Zhan\u00B9, Canyang Qi\u00B9", bold: true, size: 26 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "\u00B9School of Clinical Medicine, Xinjiang Medical University, Urumqi 830011, China", size: 22 })] }),
  pc(""),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: "*Correspondence: [your-email]", size: 20, color: "666666" })] }),
  hr(CONTENT_W),
  h2("Abstract"),
  pb("Background. "), p("UniPert-G2CP (Li et al., Cell, 2026) introduced a unified framework for predicting transcriptomic responses to genetic and chemical perturbations across five cancer cell lines. To date, no independent reproduction or large-scale extension has been reported."),
  pb("Methods. "), p("We built the core UniPert-G2CP architecture, which integrates ECFP4 compound fingerprints, ESM2-8M protein anchoring, contrastive learning with CPI alignment, and two-stage transfer, and scaled it to 32,039 compounds × 162 cell lines (4× and 32× the original, respectively) with genome-wide output (12,328 genes)."),
  pb("Results. "), p("Held-out gene knockout PCC = 0.442; novel drug PCC = 0.3047; directional accuracy for top-5% effect-size genes = 73.8%. Compound-Protein Interaction (CPI) enrichment factor at top 0.5% = 139 (random baseline = 1). Benchmark-consistent evaluation: mechanism-clustering SMD = 1.288 (original 1.85); self-consistency Mantel ρ = 0.852 under the original formulation; Tanimoto SMD validation = 1.613 (original 1.61). Biological validation: CD36 knockout → PPARG/CEBPA downregulation, consistent with prior experimental evidence; metformin on HEPG2 → SLC7A5/PLIN2 downregulation."),
  pb("Conclusions. "), p("The architecture scales to 32× the original size (162 cell lines, 32,039 compounds, genome-wide), but performance is sensitive to CPI training vocabulary: mechanism-clustering SMD drops approximately 30% from the original (1.85 → 1.288), indicating a domain-transfer bottleneck. The platform is made available as an interactive API upon reasonable request."),
  p(""),
  pb("Keywords: "), p("perturbation prediction; transcriptomics; contrastive learning; drug response; CRISPR screening; virtual cell"),
  hr(CONTENT_W),
  h2("1. Introduction"),
  p("Predicting transcriptomic responses to molecular perturbations has emerged as a central challenge in AI-driven biology. Li et al. published UniPert-G2CP in Cell [1], unifying genetic and chemical perturbagens in a shared embedding space spanning 4,994 genes, 7,860 compounds, and 5 cancer lines."),
  p("This work is motivated by two observations. First, several of the original evaluation settings assess self-consistency rather than generalization (e.g., sciPlex3 holds compound identity while varying dose; Mantel ρ compares embeddings against model-predicted rather than independently measured effects). Second, training code was not released, although a reproduction repository [2] and a Zenodo archive [3] are available."),
  p("This work contributes: (1) an independent full-scale reconstruction of the architecture (32,039 compounds × 162 cell lines, genome-wide output); (2) a disentangled evaluation framework that distinguishes held-out prediction, self-consistency, and benchmark-consistent metrics; (3) biological validation against published experimental evidence."),
  h2("2. Methods"),
  h3("2.1 Datasets and Preprocessing"),
  p("We utilized LINCS L1000 (level 5, GSE92742; moderated Z-scores, 305,297 samples), DepMap CRISPR screens (23Q2 release, CERES gene-effect scores), and sciPlex3 data. Compound SMILES were obtained from ChEMBL 37. CPI pairs (19,161 pairs, 2,881 compounds \u00d7 1,442 targets) were extracted from ChEMBL activity measurements (IC50, EC50, Ki, Kd; threshold <100 \u00b5M; duplicate compound-target pairs aggregated by minimum activity value). Primary training comprised 4,994 genes, 32,039 compounds, and 143 cell lines, with 19 additional GEO-derived lines added (GSE identifiers provided in Data Availability; total 162 lines, minimum 50 samples per line). Data split: perturbation-level hold-out (10% of perturbations randomly assigned to test set unseen during training; all cell lines and doses of held-out perturbations withheld); remaining 90% split 80/20 for train/validation."),
  h3("2.2 Model Architecture"),
  p("The architecture follows UniPert-G2CP: drug encoder (ECFP4, 2048-bit → 512-dim Linear); gene encoder (4,994×512 embedding, first 320 dims ESM2-8M-anchored [4]); unified space (NT-Xent with temperature τ = 0.07 + CPI alignment via cosine distance, loss weights 1.0 : 0.8); phenotype head takes the concatenated perturbation embedding (512-dim) and cell embedding (32-dim) as a 544-dim input, processed through two fully connected layers (Linear 544→1024, GELU, Linear 1024→12,328) to produce genome-wide output; two-stage transfer (DepMap → LINCS). The gene encoder dimension (4,994) defines the perturbation vocabulary (genes that can be knocked out); the phenotype head output (12,328) defines the response vocabulary (genes whose expression is predicted). The head is a generic MLP that does not require equality between these two gene sets, following standard perturbation-model architecture (GEARS, CPA)."),
  h3("2.3 Training Protocol"),
  p("Stage A (DepMap pretraining): 10 epochs, lr=3×10⁻⁴, only phenotype head trained. Stage B (LINCS finetuning): AdamW (lr=3×10⁻⁴), batch size 256, 40 epochs, gene encoder frozen throughout, drug encoder and phenotype head trained."),
  h2("3. Results"),
  h3("3.1 Held-Out Perturbation Prediction"),
  p("Metrics. PCC: Pearson Correlation Coefficient between predicted and measured expression changes, computed across all genes per held-out sample then averaged. Directional accuracy: fraction of genes for which the sign of predicted change matches the sign of measured change; top-5% defined by ranking genes by absolute measured effect size. SMD: Standardized Mean Difference, calculated as (mean intra-class cosine similarity − mean overall cosine similarity) / pooled standard deviation, grouping drugs by Mechanism of Action. Mantel self-consistency ρ: Mantel test with Spearman correlation, 9,999 permutations using cosine distance, comparing molecular embedding similarity against model-predicted effect similarity."),
  pb("Table 1. Held-out perturbation prediction performance."),
  tbl(["Metric","Value","Context"],
    [["Gene knockout PCC","0.442","GEARS-class models: 0.4\u20130.5"],
     ["Novel drug PCC","0.3047","Not reported in original"],
     ["Directional accuracy (genome-wide)","60.8%","compared to random baseline 50%"],
     ["Directional accuracy (top-5%)","73.8%","GEARS-class: ~70\u201375%"]]),
  h3("3.2 Compound-Protein Interaction Enrichment"),
  pb("Table 2. CPI enrichment factor at top percentiles (random baseline = 1)."),
  tbl(["Percentile","This study"],
    [["Top 0.5%","139"],["Top 1%","77"]]),
  h3("3.3 Benchmark-Consistent Evaluation"),
  pb("Table 3. Benchmark-consistent metrics on the original evaluation datasets. Evaluation set notes clarify differences in data scope."),
  tbl(["Metric","This study","[1]","Evaluation set note"],
    [["SMD (cosine)","1.288","[1] 1.85","ChEMBL therapeutics (both)"],["ECFP4 Tanimoto SMD (validation)","1.613","[1] 1.61","ChEMBL therapeutics (both)"],["Mantel self-consistency \u03c1","0.852","[1] 0.43","Subramanian PCL (both); different PCL sets"],["Cell lines","162","[1] 5","—"]]),
  p("Note: the original study's sciPlex3 per-cell-type PCC of 0.98 was obtained on a top-DEG subset under near-domain splits (known compounds, novel doses/cell lines). Our per-cell-line PCC (0.359, Section 3.4) is computed on the full 12,328-gene set under perturbation-level hold-out (completely unseen drugs), constituting a stricter evaluation setting. The two numbers are not directly comparable."),
  h3("3.4 Per-Cell-Line Drug Perturbation Resolution"),
  p("We evaluate single-cell-line resolution within our training domain by computing held-out drug perturbation PCC separately for each cell line (120 lines with sufficient samples). The mean per-cell-line PCC is 0.359 (median 0.362, range -0.004 to 0.615; A549 = 0.362, MCF7 = 0.360). This metric is computed on the full gene set (12,328 genes) under perturbation-level held-out splits (drugs unseen during training). The original reports PCC = 0.98 (A549/MCF7/K562) on sciPlex3, but those values are on a top-DEG subset under near-domain splits (seen compounds, novel doses); our lower per-cell-line values reflect the more stringent setting of full-gene, unseen-perturbagen evaluation on the larger and more heterogeneous LINCS screening library."),
  h3("3.5 Biological Validation"),
  p("These four representative positive cases illustrate the model's ability to recapitulate known biological mechanisms. They are not a systematic genome-wide validation, and false-positive predictions exist."),
  pb("Table 4. Representative biological validation cases."),
  tbl(["Perturbation","Predicted genes","Consistency"],
    [["CD36 KO x ASC","PPARG\u2193, CEBPA\u2193","Adipogenic [7]"],["Dexamethasone x HA1E","TSC22D3\u2191","Glucocorticoid target [8]"],["Aspirin x PBMC-NK","PRF1\u2191, GZMB\u2191, GNLY\u2191","NK cytotoxicity [9]"],["Metformin x HEPG2","SLC7A5\u2193, PLIN2\u2193","mTORC1 + lipid droplet [10]"]]),
  p("CD36 silencing has been experimentally shown to attenuate adipogenesis and reduce PPAR\u03b3 and C/EBP\u03b1 expression [7]. TSC22D3 (GILZ) is a canonical dexamethasone-induced gene mediating glucocorticoid anti-proliferative effects [8]. Aspirin has been reported to augment natural killer cell cytotoxic activity [9]. Metformin inhibits mTORC1 signaling via AMPK activation; the combination of metformin with LAT1 (SLC7A5) inhibition produces additive mTOR suppression [10]."),
  h2("4. Discussion"),
  h3("4.1 Contributions"),
  p("This study makes four contributions. First, it independently builds the UniPert-G2CP architecture at 32-fold larger scale (162 vs. 5 cell lines) with genome-wide output, demonstrating that the framework scales but with performance sensitivity to CPI vocabulary domain (SMD drops ~30% from the original, see Section 4.3). Second, we highlight two caveats in the original evaluation framework that should be interpreted with caution when assessing generalization performance: the SMD metric inherits a high baseline from the ChEMBL therapeutics dataset (original ECFP4 Tanimoto SMD = 1.61, which we replicate at 1.613), and the alignment \u03c1 metric compares embeddings against model-predicted rather than measured effects \u2014 a self-consistency test. Third, under the same self-consistency Mantel formulation (molecular embedding similarity vs. model-predicted effect similarity, PCL-based Mantel test), we obtain ρ = 0.852. The original reports 0.43 under the same formulation but on a different PCL set; the two numbers are not directly interchangeable and we do not claim superiority on this metric. Fourth, we provide qualitative biological validation examples (CD36/ASC, dexamethasone/HA1E, aspirin/PBMC-NK, metformin/HEPG2) that the original study did not report."),
  p("Held-out perturbation metrics (gene PCC = 0.442; novel drug PCC = 0.3047; directional accuracy = 73.8% on top-5% effect-size genes) are consistent with state-of-the-art benchmarks reported for GEARS [6]. CPI enrichment (EF = 139) and self-consistency \u03c1 (0.852) are reported on their respective evaluation sets and should not be directly compared across studies."),
  h3("4.2 Protein Encoder Design Rationale"),
  p("A key architectural difference from the original is our use of ESM2-8M [4], a lightweight protein language model (8M parameters, 320-dim output), for gene-side semantic anchoring. The original UniPert employs a full pipeline of ESM + MSA + protein-similarity GNN message passing. Our choice is motivated by the following considerations, balancing practical constraints with methodological advantages. (1) Parameter efficiency: ESM2-8M achieves competitive performance on protein function prediction and variant effect prediction benchmarks despite its compact size. The ESM2 family shares a unified self-supervised pretraining objective (masked language modeling over 65 million UniRef50 sequences); the 8M variant captures core evolutionary and structural signals with substantially lower computational cost than its larger counterparts (ESM2-35M/150M/650M). (2) Dimensionality as a feature: the 320-dim output of ESM2-8M serves as a compact semantic prior for our 4,994-gene embedding table, where each gene has limited training observations. A lower-dimensional prior reduces the risk of overfitting in the low-data regime per gene, a concern that would be amplified with higher-dimensional encoders. (3) Anchoring over substitution: rather than using ESM features as a drop-in replacement for the learnable gene embeddings, we employ an anchoring strategy: the first 320 dimensions of the gene embedding table are initialized from ESM2-8M vectors and lightly regularized via L2 loss during training. This preserves the full flexibility of a learnable embedding (the remaining dimensions and all parameters adapt freely), while injecting protein-level semantics as a structured prior. The phenotype head is free to discover gene\u2192phenotype mappings that complement, or even override, the ESM prior where it is beneficial. (4) Hardware compatibility: at 8M parameters, ESM2-8M runs comfortably on consumer GPUs (NVIDIA RTX 3050 Ti, 4 GB VRAM), enabling reproducible training without cloud or institutional compute resources. We acknowledge that the anchoring strategy constitutes a simplification relative to the original GNN-based protein encoder, and that incorporating MSA and protein-similarity graph components \u2014 either as additional inputs to the embedding table or as a dedicated GNN encoder \u2014 is a natural direction for future work when larger computational resources become available."),
  h3("4.3 SMD Gap Analysis"),
  p("The mechanism-clustering SMD under identical evaluation conditions is 1.288, reaching 69.6% of the original's 1.85. We attribute the gap primarily to the lexical disjointness of compound vocabularies: the original CPI training set (53,963 PubChem-based pairs) has less than 1% overlap with our LINCS BRD vocabulary (19,161 ChEMBL-based pairs, which approaches the physical ceiling of available compound-target combinations within our vocabulary). The lightweight ESM2-8M anchoring, which omits the original's MSA and protein-similarity GNN pipeline, may further contribute."),
  h3("4.4 Limitations"),
  p("(1) Single-gene direction prediction remains noisy (60.8% genome-wide); the platform is a candidate-screening and hypothesis-generation tool, not a substitute for experimental validation. (2) The protein encoder is simplified relative to the original. (3) CPI training data has reached its physical ceiling under the current compound vocabulary. (4) The SMD evaluation dataset reflects properties of well-characterized therapeutics and is not representative of screening-library compounds. (5) Single-cell resolution is not implemented. (6) Training code of the original study has not been released; our reconstruction is based on paper descriptions and the public evaluation repository, and may contain implementation differences. (7) The 32-fold increase in cell-line coverage introduces greater biological heterogeneity, which itself can lower aggregate performance metrics; genome-wide output (12,328 genes) is more challenging than the original landmark-gene prediction, potentially inflating the apparent performance gap."),
  h2("5. Conclusion"),
  p("We have independently built and scaled the UniPert-G2CP architecture to 162 cell lines, 32,039 compounds, and genome-wide output, producing a deployed virtual-cell prediction platform. The architecture scales successfully, but performance is sensitive to CPI training vocabulary, revealing a domain-transfer bottleneck rather than a simple architectural limitation. Held-out perturbation metrics, benchmark-consistent evaluation, and biological validation examples are provided as transparent reference points. The platform is freely available as a candidate-screening tool for perturbation biology research."),
  h2("Data and Code Availability"),
  p("All training data are from public repositories: LINCS L1000 (GSE92742, level 5 moderated Z-scores), DepMap CRISPR (23Q2, CERES scores), sciPlex3, and ChEMBL 37. Nineteen additional GEO-derived cell lines were incorporated from GSE61302 (ASC), GSE22886 and GSE60235 (immune subsets: BM-PlasmaCell, PBMC-Bcell, PBMC-CD4T, PBMC-CD8T, PBMC-monocyte, PBMC-NK, PBMC-neutrophil, PBMC-plasmacytoidDC), plus additional series for HUVEC, iPSC, Chondrocyte, Islet, Keratinocyte, Skeletal-Muscle, Retinal Pigment Epithelium, Iris Pigment Epithelium, Choroid Fibroblast, and smooth muscle cells (aortic, bronchial, coronary, pulmonary); complete GSE identifiers are provided in the repository README. The original study open resources are available at GitHub [2] and Zenodo [3]. Our reproduction codebase, trained model weights (g2cp_full_cpi_v7.pt), and preprocessed datasets will be released on Zenodo/Figshare (DOI to be assigned upon submission) and GitHub at https://github.com/wkr112344/G2CP-virtual-cell. API service available upon reasonable request."),
  h2("Declarations"),
  p("Competing interests: The author declares no competing interests."),
  p("Funding: This work received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors."),
  p("Ethics statement: This study used exclusively publicly available, anonymized datasets and involved no human or animal subjects; therefore, no ethical approval was required."),
  p("Author contributions: K.W. conceived the study, implemented the core architecture and training pipeline, conducted all computational experiments, and wrote the manuscript. L.Z. developed the interactive web interface and visualization components. C.Q. contributed to evaluation benchmarking and biological validation data curation. All authors reviewed and approved the manuscript."),
  h2("References"),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "[1] Li Y, et al. Unified genetic and chemical perturbation prediction across cell lines with UniPert-G2CP. ", size: 22, font: "Times New Roman" }), new TextRun({ text: "Cell", italics: true, size: 22, font: "Times New Roman" }), new TextRun({ text: ". 2026;189(13):2678-2695.e18. doi:10.1016/j.cell.2026.06.005.", size: 22, font: "Times New Roman" })] }),
  p("[2] lynn-1998. UniPert-G2CP_reproduce [Source code]. GitHub. 2026. https://github.com/lynn-1998/UniPert-G2CP_reproduce. Accessed August 7, 2026."),
  p("[3] Li Y, et al. UniPert-G2CP: evaluation data and model outputs [Data set]. Zenodo. 2026. doi:10.5281/zenodo.20355906."),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "[4] Lin Z, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. ", size: 22, font: "Times New Roman" }), new TextRun({ text: "Science", italics: true, size: 22, font: "Times New Roman" }), new TextRun({ text: ". 2023;379(6637):1123-1130. doi:10.1126/science.ade2574.", size: 22, font: "Times New Roman" })] }),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "[5] Subramanian A, et al. A next generation connectivity map: L1000 platform and the first 1,000,000 profiles. ", size: 22, font: "Times New Roman" }), new TextRun({ text: "Cell", italics: true, size: 22, font: "Times New Roman" }), new TextRun({ text: ". 2017;171(6):1437-1452.e17. doi:10.1016/j.cell.2017.10.049.", size: 22, font: "Times New Roman" })] }),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "[6] Roohani Y, Huang K, Leskovec J. Predicting transcriptional outcomes of novel and known multi-gene perturbations with GEARS. ", size: 22, font: "Times New Roman" }), new TextRun({ text: "Nat Biotechnol", italics: true, size: 22, font: "Times New Roman" }), new TextRun({ text: ". 2024;42(6):927-936. doi:10.1038/s41587-023-01905-6.", size: 22, font: "Times New Roman" })] }),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "[7] Gao H, et al. Suppression of CD36 attenuates adipogenesis with a reduction of P2X7 expression in 3T3-L1 cells. ", size: 22, font: "Times New Roman" }), new TextRun({ text: "Biochem Biophys Res Commun", italics: true, size: 22, font: "Times New Roman" }), new TextRun({ text: ". 2017;491(1):42-48. doi:10.1016/j.bbrc.2017.07.077.", size: 22, font: "Times New Roman" })] }),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "[8] Ayroldi E, et al. GILZ mediates the antiproliferative activity of glucocorticoids by negative regulation of Ras signaling. ", size: 22, font: "Times New Roman" }), new TextRun({ text: "J Clin Invest", italics: true, size: 22, font: "Times New Roman" }), new TextRun({ text: ". 2007;117(6):1605-1615. doi:10.1172/JCI30724.", size: 22, font: "Times New Roman" })] }),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "[9] Muss C, et al. Stimulation of natural killer cell activity by acetylsalicylic acid. ", size: 22, font: "Times New Roman" }), new TextRun({ text: "Dtsch Z Onkol", italics: true, size: 22, font: "Times New Roman" }), new TextRun({ text: ". 2002;34(4):140-145. doi:10.1055/s-2002-36552.", size: 22, font: "Times New Roman" })] }),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "[10] Ueno S, et al. Metformin enhances anti-tumor effect of L-type amino acid transporter 1 (LAT1) inhibitor. ", size: 22, font: "Times New Roman" }), new TextRun({ text: "J Pharmacol Sci", italics: true, size: 22, font: "Times New Roman" }), new TextRun({ text: ". 2016;131(2):110-117. doi:10.1016/j.jphs.2016.04.021.", size: 22, font: "Times New Roman" })] }),
];

const enDoc = new Document({
  styles: { default: { document: { run: { font: "Times New Roman", size: 22 } } } },
  sections: [{ properties: { page: { size: { width: A4_W, height: A4_H }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } }, footers: { default: foot }, children: enChildren }]
});

// ==================== CHINESE ====================
const cnChildren = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 4 }, children: [new TextRun({ text: "复现并扩展 UniPert-G2CP:", bold: true, size: 32, font: "SimSun" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 4 }, children: [new TextRun({ text: "一个 162 细胞系全基因组虚拟细胞计算平台", bold: true, size: 28, font: "SimSun" })] }),
  pc(""),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "卫铠睿\u00B9*, 詹李靖\u00B9, 祁灿阳\u00B9", bold: true, size: 26, font: "SimSun" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "\u00B9新疆医科大学临床医学部，乌鲁木齐 830011", size: 22, font: "SimSun" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: "预印本（中文版） \u00b7 2026年8月", size: 20, font: "SimSun", color: "666666" })] }),
  hr(CONTENT_W),
  h2("摘要"),
  p("背景：2026年发表于《Cell》的UniPert-G2CP框架首次实现了基因与药物双模态扰动的统一转录组预测，但仅覆盖5个细胞系与7,860种化合物。本文独立复现该架构并大幅扩展规模，区分自洽评测与真实留出评测。"),
  p("方法：独立复现ECFP4指纹药物编码器、ESM2-8M蛋白锚定基因编码器、对比学习与CPI对齐、两阶段迁移，扩展至32,039种药物\u00d7162个细胞系，输出12,328全基因组。采用扰动级完全留出核心评测，并用论文原始数据与公式同口径复刻。"),
  p("结果：基因敲除PCC=0.442；新药PCC=0.3047；强效应基因方向准确率73.8%；CPI富集因子EF=139（随机基线=1）；论文同口径SMD=1.302(论文1.85)；Mantel自洽\u03c1=0.852（原文同口径）；Tanimoto SMD=1.613(论文1.61,验证复刻正确)。"),
  p("结论：架构可扩展至32倍原规模（162细胞系、32,039化合物、全基因组），但性能对CPI训练词表高度敏感：机制聚类SMD较原文下降约30%（1.85→1.288），提示领域迁移瓶颈而非单纯的架构失效。平台已部署为交互式API。"),
  p(""),
  pb("关键词："), p("扰动预测；转录组学；对比学习；药物响应；CRISPR筛选；虚拟细胞"),
  hr(CONTENT_W),
  h2("1. 引言"),
  p("2026年Li等人在《Cell》发表UniPert-G2CP[1]，将基因与药物干预统一建模，被视为虚拟细胞的关键计算单元。原文覆盖4,994个基因与7,860个化合物，在5种癌细胞系上进行了系统评测。"),
  p("本文贡献：(1)独立全文复现并大幅扩展规模；(2)区分自洽/自洽评测与真实留出评测，提供多维度基准数字；(3)通过已知机制验证生物学合理性。"),
  h2("2. 方法（摘要）"),
  p("训练数据源自LINCS L1000、DepMap CRISPR、sciPlex3。CPI训练对19,161对，通过ChEMBL activities构建（<100\u00b5M）。架构遵循UniPert-G2CP：药物编码器为ECFP4\u2192512维线性；基因编码器为可训练embedding（4,994\u00d7512），前320维由ESM2-8M初始化；统一嵌入空间（NT-Xent对比学习+CPI对齐损失）；表型头为MLP（544维输入\u21921024\u219212,328全基因组输出）；两阶段迁移（DepMap预训练\u2192LINCS微调）。训练：AdamW（lr=3\u00d710\u207b\u2074），冻结模式40轮。关键澄清：基因编码器维度4,994是扰动词表——可以被敲除的基因集合（输入）；表型头输出12,328是响应词表——预测其表达变化的基因集合（输出）。两者不等价：表型头是通用MLP，将544维扰动嵌入（药物/基因嵌入+32维细胞系嵌入）经两个全连接层映射到12,328维全转录组输出。这一设计与GEARS、CPA等扰动模型的标准架构一致，不要求输入与输出基因维度相等。"),
  h2("3. 核心结果"),
  h3("3.1 扰动级完全留出"),
  tbl(["指标","数值","背景"],
    [["基因敲除PCC","0.442","GEARS系模型: 0.4-0.5"],["新药预测PCC","0.3047","原文未报该口径"],["方向准确率(全基因)","60.8%","随机基线50%"],["方向准确率(强效应前5%)","73.8%","GEARS系: ~70-75%"]]),
  h3("3.2 CPI富集"),
  tbl(["百分位","本文（随机基线=1）"],
    [["Top 0.5% (our dataset)","139"],["Top 1% (our dataset)","77"]]),
  h3("3.3 论文同口径复刻"),
  tbl(["指标","本文","[1]"],
    [["SMD(Cosine)","1.288","[1] 1.85"],["Tanimoto SMD(验证)","1.613","[1] 1.61"],["Mantel自洽\u03c1","0.852","[1] 0.43"],["细胞系数","162","[1] 5"]]),
  h3("3.4 单细胞系药物扰动分辨率"),
  p("我们在训练域内评估了单细胞系级别的预测能力：对每个细胞系分别计算扰动级留出的药物扰动PCC（120个有足够样本的细胞系）。单系PCC平均为0.359（中位0.362，范围-0.004至0.615；A549=0.362，MCF7=0.360）。该指标基于全基因组（12,328基因）在扰动级完全留出（训练中未见过的药物）条件下计算，直接对标原文sciPlex3的单细胞系评测。原文在sciPlex3上报告的PCC为0.98（A549/MCF7/K562），但该分数来自top-DEG子集且采用近域留出（药物见过仅换剂量/细胞系）；我们的单系PCC反映了全基因、未见扰动、更大且更异质的LINCS筛查库这一更严格的评测设定。"),
  h3("3.5 生物学验证"),
  p("以下四个案例为代表性阳性验证，并非全基因组系统验证，不排除假阳性预测的存在。"),
  tbl(["场景","预测","一致性"],
    [["CD36敲除xASC","PPARG\u2193, CEBPA\u2193","脂肪代谢 [7]"],["地塞米松xHA1E","TSC22D3\u2191","糖皮质激素靶 [8]"],["阿司匹林xPBMC-NK","PRF1\u2191,GZMB\u2191,GNLY\u2191","NK杀伤 [9]"],["二甲双胍xHEPG2","SLC7A5\u2193, PLIN2\u2193","mTORC1+脂滴 [10]"]]),
  p("实验研究已证实：siRNA沉默CD36可抑制3T3-L1前脂肪细胞的脂肪分化并降低PPARγ和C/EBPα蛋白表达 [7]；TSC22D3(GILZ)是地塞米松的经典诱导靶基因，介导糖皮质激素的抗增殖效应 [8]；阿司匹林可增强自然杀伤细胞的细胞毒活性 [9]；二甲双胍通过AMPK激活抑制mTORC1信号，与LAT1(SLC7A5)抑制剂联用可产生叠加的mTOR抑制效果 [10]。以上四项均与我们的模型预测方向一致。"),
  h2("4. 讨论"),
  h3("4.1 核心贡献"),
  p("本文做出四项贡献。第一，独立构建了UniPert-G2CP核心架构并扩展至32倍规模（162 vs. 5个细胞系）、全基因组输出（12,328基因），证明该架构可扩展至大规模场景，但性能对CPI词表域高度敏感（SMD较原文下降约30%，详见4.3节）。第二，指出原文评测框架中两个在评估泛化性能时需谨慎解读的设定：SMD指标受益于评测集（ChEMBL真实药物）的高指纹基线（原文ECFP4 Tanimoto SMD = 1.61，我们复刻为1.613）；对齐\u03c1指标比较的是分子嵌入与模型自身预测效应（自洽性测试），而非独立测量效应。第三，在原文的自洽Mantel评测口径下，我们取得\u03c1 = 0.852。原文在同一口径下报告0.43（基于不同PCL集），两者不可直接比较。第四，提供了原文未包含的定性生物学验证案例（CD36/ASC、地塞米松/HA1E、阿司匹林/PBMC-NK、二甲双胍/HEPG2）。"),
  p("在扰动级完全留出评测中，基因敲除预测PCC为0.442，新药预测PCC为0.3047，强效应基因（前5%）方向准确率为73.8%，与领域前沿（GEARS [6] 代表）同级。CPI富集因子（EF=139，显著高于随机基线）与自洽ρ（0.852）均基于各自数据集评测，不宜跨研究直接比较。"),
  h3("4.2 蛋白编码器选择说明"),
  p("与原文的一个关键架构差异是：我们采用轻量级ESM2-8M蛋白语言模型[4]为基因侧提供语义锚定，而原文使用ESM+MSA+蛋白相似图GNN消息传递的完整管线。这一选择兼顾实践约束与方法优势：（1）参数效率：ESM2-8M在蛋白功能预测与变异效应预测等下游任务上有竞争力表现，在同一自监督预训练框架（6500万条UniRef50序列的掩码语言建模）下，8M变体已捕获核心进化与结构信号，计算成本远低于同系列大模型（ESM2-35M/150M/650M）。（2）低维度本身是优势：320维输出为我们4,994个基因嵌入表提供了紧凑的语义先验——每个基因的训练观测有限，低维先验有助于避免过拟合，这一优势在高维编码器上会被放大。（3）锚定而非替代：我们不是用ESM特征直接替代可训练基因嵌入，而是采用锚定策略——基因嵌入表的前320维由ESM2-8M初始化并在训练中施加轻量L2正则，剩余维度和所有参数自由学习，表型头可自由发现基因→表型映射，在必要时覆盖甚至偏离ESM先验。（4）硬件兼容：8M参数量可在消费级GPU（RTX 3050 Ti, 4 GB）上正常运行，无需云端计算资源即可实现可复现的训练。我们坦承锚定策略构成对原文GNN蛋白编码器的简化，将MSA和蛋白相似图组件纳入基因编码器——既可作为嵌入表的附加输入，也可作为独立GNN编码器——是未来在有更大算力条件下的自然延伸方向。"),
  h3("4.3 SMD差距分析"),
  p("在同口径评测下，我们的机制聚类SMD为1.288，达到原文1.85的69.6%。差距的首要来源是化合物词表的互不重叠：原文的CPI训练集（53,963对，PubChem化合物）与我们的LINCS BRD词表（19,161对，ChEMBL化合物，已接近该词表下的物理上限）重叠率不足1%，无法直接共用。轻量级ESM2-8M蛋白锚定（未含原文的MSA与蛋白相似图GNN）也可能贡献部分差距。"),
  h3("4.4 局限"),
  p("(1)全基因方向准确率60.8%，不应直接作为实验结论——该平台宜作为候选筛选与假说生成工具。(2)蛋白编码器为简化版本。(3)CPI训练数据在现有化合物词表下已达物理上限。(4)SMD评测集反映已知药物的结构特征，与筛查库化合物存在系统性差异。(5)未实现单细胞分辨率。(6)原文未公开训练代码，本复现基于论文描述与公开评测仓库，可能存在实现差异。(7)32倍细胞系扩展带来的生物异质性提升本身也会拉低总体指标；全基因组输出（12,328基因）相比原研究的 landmark 基因预测难度更高，可能放大表观性能差距。"),
  h2("5. 结论"),
  p("我们独立构建并扩展UniPert-G2CP架构至162细胞系、32,039化合物、全基因组输出，建成可部署的虚拟细胞预测平台。架构可成功扩展，但性能对CPI训练词表高度敏感，揭示的是领域迁移瓶颈而非单纯的架构局限。本文提供了留出评测指标、论文同口径复刻与生物学验证案例作为透明参考。平台可作为扰动生物学研究的候选筛选工具免费获取。"),
  h2("数据与代码可用性"),
  p("所有训练数据均来自公开数据库：LINCS L1000 (GSE92742，level 5 moderated Z-scores)、DepMap CRISPR (23Q2，CERES评分)、sciPlex3、ChEMBL 37。额外19个GEO来源细胞系来自GSE61302（ASC）、GSE22886及GSE60235（免疫亚群：BM-PlasmaCell、PBMC-Bcell、PBMC-CD4T、PBMC-CD8T、PBMC-monocyte、PBMC-NK、PBMC-neutrophil、PBMC-plasmacytoidDC），以及HUVEC、iPSC、软骨细胞、胰岛、角质形成细胞、骨骼肌、视网膜色素上皮、虹膜色素上皮、脉络膜成纤维细胞和平滑肌细胞（主动脉、支气管、冠状动脉、肺动脉）等GEO系列，完整GSE编号见代码仓库README。自洽Mantel检验采用9,999次置换、余弦距离。原研究公开资源见GitHub [2]与Zenodo [3]。本复现工程的代码、模型权重（g2cp_full_cpi_v7.pt）与预处理数据集将发布于Zenodo/Figshare (DOI提交后分配) 及GitHub。API服务可按合理请求提供。"),
  h2("声明"),
  p("利益冲突：作者声明无竞争利益。"),
  p("经费资助：本研究未获得任何公共、商业或非营利部门资助机构的专项资助。"),
  p("伦理声明：本研究全部使用公开匿名数据集，不涉及人类或动物受试者，无需伦理审批。"),
  p("作者贡献：卫铠睿构思研究方案，实现核心架构与训练管线，完成全部计算实验与论文撰写。詹李靖开发交互式网页界面与可视化组件。祁灿阳参与评测基准构建与生物学验证数据整理。全体作者审阅并同意投稿。"),
  h2("参考文献"),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "[1] Li Y, et al. ", size: 22, font: "SimSun" }), new TextRun({ text: "Cell", italics: true, size: 22, font: "Times New Roman" }), new TextRun({ text: ". 2026.", size: 22, font: "SimSun" })] }),
  p("[2] lynn-1998. UniPert-G2CP_reproduce [Source code]. GitHub. 2026."),
  p("[3] Li Y, et al. UniPert-G2CP [Data set]. Zenodo. 2026. doi:10.5281/zenodo.20355906."),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "[4] Lin Z, et al. ", size: 22, font: "SimSun" }), new TextRun({ text: "Science", italics: true, size: 22, font: "Times New Roman" }), new TextRun({ text: ". 2023;379:1123-1130.", size: 22, font: "SimSun" })] }),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "[5] Subramanian A, et al. ", size: 22, font: "SimSun" }), new TextRun({ text: "Cell", italics: true, size: 22, font: "Times New Roman" }), new TextRun({ text: ". 2017;171:1437-1452.", size: 22, font: "SimSun" })] }),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "[6] Roohani Y, et al. ", size: 22, font: "SimSun" }), new TextRun({ text: "Nature Biotechnology", italics: true, size: 22, font: "Times New Roman" }), new TextRun({ text: ". 2023;42:927-936.", size: 22, font: "SimSun" })] }),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "[7] Gao H, et al. ", size: 22, font: "SimSun" }), new TextRun({ text: "Biochem Biophys Res Commun", italics: true, size: 22, font: "Times New Roman" }), new TextRun({ text: ". 2017;491:42-48.", size: 22, font: "SimSun" })] }),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "[8] Ayroldi E, et al. ", size: 22, font: "SimSun" }), new TextRun({ text: "J Clin Invest", italics: true, size: 22, font: "Times New Roman" }), new TextRun({ text: ". 2007;117:1605-1615.", size: 22, font: "SimSun" })] }),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "[9] Muss C, et al. ", size: 22, font: "SimSun" }), new TextRun({ text: "Dtsch Z Onkol", italics: true, size: 22, font: "Times New Roman" }), new TextRun({ text: ". 2002;34:140-145.", size: 22, font: "SimSun" })] }),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "[10] Ueno S, et al. ", size: 22, font: "SimSun" }), new TextRun({ text: "J Pharmacol Sci", italics: true, size: 22, font: "Times New Roman" }), new TextRun({ text: ". 2016;131:110-117.", size: 22, font: "SimSun" })] }),
];

const cnDoc = new Document({
  styles: { default: { document: { run: { font: "SimSun", size: 22 } } } },
  sections: [{ properties: { page: { size: { width: A4_W, height: A4_H }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } }, footers: { default: foot }, children: cnChildren }]
});

async function main() {
  const enBuf = await Packer.toBuffer(enDoc);
  fs.writeFileSync("biorxiv_english.docx", enBuf);
  console.log("EN Word: biorxiv_english.docx", (enBuf.length/1024).toFixed(0) + "KB");

  const cnBuf = await Packer.toBuffer(cnDoc);
  fs.writeFileSync("biorxiv_chinese.docx", cnBuf);
  console.log("CN Word: biorxiv_chinese.docx", (cnBuf.length/1024).toFixed(0) + "KB");
}
main().catch(e => { console.error(e); process.exit(1); });
