# Supplementary Table S21a. Four representative score-to-object tracking cases

From the full 12-item targeted audit (Table S21), this table selects 4 cases to show that source/accession,
processing level, named release, study-specific artifact, and exact processed object can be different recoverability
layers. **Cases 1–2 are the most direct examples of Level-5 build ambiguity; cases 3–4 only illustrate the
distinction between provenance layers and do not imply that these Level-3 studies should have reported
level5beta2020/dcic2021.**

| # | Study | Publicly reported level | Gives a build identifier at the same granularity as this paper's Level-5 comparison? | How this paper uses the case |
|---|---|---|---|---|
| 1 | DeepCE | Explicitly L1000 Level 5; public processed train/dev/test and code | No | **Direct case**: shows that "Level 5" itself is not a build identifier at the same granularity as `level5beta2020` / `dcic2021`. This paper does not conclude from it that the original study is irreproducible. |
| 2 | MiTCP | Explicitly GSE92742/GSE70138 and Level 5; preprocessed data archived | No | **Direct case**: even when the broad source and Level 5 are both clear, the paper-level descriptor and the exact processed build can still sit at different layers. |
| 3 | TranSiGen / XPert | Explicitly CMap LINCS Resource 2020, Level 3, with processed artifacts | Not applicable (their analysis is Level 3) | **Layer case**: shows that the named release, the processing level, and the study-specific processed artifact can each serve different recoverability functions; not used to accuse them of lacking a Level-5 build. |
| 4 | Dr.VAE | Explicitly CMap-L1000v1, Level 3 | Not applicable (their analysis is Level 3) | **Layer case**: shows that a named dataset/version is not the same as the exact metric-generating processed object meant here; whether finer identification is needed depends on the specific benchmark. |

**Interpretation boundary.** This audit is targeted context; it does not estimate a field-wide incidence rate, nor
does it assess the overall reproducibility quality of the audited papers. This paper uses it only to argue that when
a benchmark ecosystem genuinely contains multiple processed products usable as a reference, score auditability is
best bound directly to the actual scoring object, the commonly evaluable set, and the protocol.
