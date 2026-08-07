# -*- coding: utf-8 -*-
"""一次性补丁：TRANSFER_LIB 升级为 v3（每站带具体蛋白名）。"""
src = open('unipret/serve_api.py', encoding='utf-8').read()
start = src.index('    # 通路级联机制（v2：级间"怎么变构/传递"的具体文字，按通路）')
end = src.index('    # 表型终点具体描述（v2：基因表达调控后细胞最终变成什么样）')
new_lib = '''    # 通路级联机制（v3：每一站都是具体蛋白/分子名 + 该站机制）
    TRANSFER_LIB = {
        "jakstat": [
            {"n": "JAK（激酶）", "m": "受体二聚化启动 JAK 交叉磷酸化 → STAT 被磷酸化"},
            {"n": "pSTAT 二聚体", "m": "磷酸化 STAT 二聚化 → 转位入核"},
            {"n": "STAT-靶基因启动子", "m": "结合 GAS 启动子 → 上调免疫/增殖靶基因"}],
        "mapk": [
            {"n": "RAS-GTP", "m": "EGFR 活化 → 招募 GRB2-SOS → 催化 RAS 装载 GTP"},
            {"n": "RAF → MEK", "m": "RAS-GTP 招募 RAF → RAF 磷酸化激活 MEK"},
            {"n": "ERK（入核）", "m": "MEK 磷酸化 ERK → ERK 转位入核 → 激活 MYC/ELK1 增殖转录因子"}],
        "pi3k": [
            {"n": "PI3K / 膜上 PIP3", "m": "受体激活 PI3K → 质膜生成 PIP3"},
            {"n": "AKT", "m": "PIP3 招募 AKT → PDK1 磷酸化激活 AKT"},
            {"n": "mTORC1 / FOXO", "m": "AKT 磷酸化激活 mTORC1、抑制 FOXO → 促存活、促增殖、抑凋亡"}],
        "p53": [
            {"n": "p53（ATM/ATR 磷酸化）", "m": "DNA 损伤/应激 → ATM/ATR 磷酸化 p53 → p53 蛋白稳定"},
            {"n": "p53 四聚体", "m": "p53 四聚化 → 转录激活 p21/BAX"},
            {"n": "p21 → CDK；BAX → 线粒体", "m": "p21 抑制 CDK → 周期阻滞；BAX 促线粒体外膜透化 → 凋亡"}],
        "hdac": [
            {"n": "HDAC（组蛋白去乙酰化酶）", "m": "抑制 HDAC → 组蛋白乙酰化水平升高"},
            {"n": "染色质（开放状态）", "m": "乙酰化中和组蛋白正电荷 → 染色质解旋、转录因子可及性增加"},
            {"n": "抑癌基因启动子", "m": "抑癌基因重新表达 → 转录重编程"}],
        "cellcyc": [
            {"n": "CDK-cyclin 复合物", "m": "CDK 活性受抑 → RB 保持低磷酸化"},
            {"n": "E2F（被 RB 扣留）", "m": "低磷酸化 RB 结合并扣留 E2F → G1/S 基因无法转录"},
            {"n": "G1/S 检查点", "m": "周期阻滞于 G1 → 增殖停止"}],
        "apoptosis": [
            {"n": "BAX/BAK（线粒体孔）", "m": "BH3 信号激活 BAX/BAK → 线粒体外膜透化"},
            {"n": "细胞色素 c（释放）", "m": "细胞色素 c 释放到胞质 → 组装凋亡小体"},
            {"n": "半胱天冬酶级联", "m": "Caspase-9 激活 → Caspase-3 执行 → 细胞凋亡"}],
        "egfr": [
            {"n": "EGFR 二聚体（自磷酸化）", "m": "配体诱导受体二聚化 → 酪氨酸自磷酸化"},
            {"n": "GRB2-SOS → RAS", "m": "招募 GRB2/SOS → 激活 RAS"},
            {"n": "ERK + AKT", "m": "RAS→RAF→MEK→ERK 增殖级联 + PI3K-AKT 存活级联"}],
        "nfkb": [
            {"n": "IKK → I\u03baB", "m": "IKK 磷酸化 I\u03baB → I\u03baB 被泛素化降解"},
            {"n": "NF-\u03baB（入核）", "m": "NF-\u03baB 释放并转位入核"},
            {"n": "NF-\u03baB-炎症基因", "m": "转录促炎/抗凋亡基因（IL6、BCL2L1）"}],
        "dnarepair": [
            {"n": "ATM/ATR（损伤感应）", "m": "DNA 损伤被 ATM/ATR 感应激活"},
            {"n": "CHK1/2（检查点激酶）", "m": "检查点激酶被磷酸化激活"},
            {"n": "周期阻滞 + 修复复合物", "m": "阻滞细胞周期并招募修复复合物"}],
        "wnt": [
            {"n": "\u03b2-catenin（稳定）", "m": "Wnt 配体结合 Frizzled → \u03b2-catenin 不被降解而稳定"},
            {"n": "\u03b2-catenin（入核）", "m": "\u03b2-catenin 转位入核"},
            {"n": "TCF/LEF-靶基因", "m": "结合 TCF/LEF → 增殖/干性基因转录"}],
        "notch": [
            {"n": "\u03b3-\u5206\u6ccc\u9176\u5207\u5272 \u2192 NICD", "m": "Notch \u88ab \u03b3-\u5206\u6ccc\u9176\u5207\u5272\u91ca\u653e NICD"},
            {"n": "NICD（入核）", "m": "NICD 转位入核"},
            {"n": "CSL-\u9776\u57fa\u56e0", "m": "\u7ed3\u5408 CSL \u2192 \u5206\u5316\u547d\u8fd0\u51b3\u5b9a\u57fa\u56e0\u8f6c\u5f55"}],
        "tgfb": [
            {"n": "SMAD2/3（磷酸化）", "m": "TGF-\u03b2 \u7ed3\u5408\u53d7\u4f53 \u2192 SMAD2/3 \u88ab\u53d7\u4f53\u78f7\u9178\u5316"},
            {"n": "SMAD \u590d\u5408\u7269\uff08\u5165\u6838\uff09", "m": "pSMAD \u4e0e SMAD4 \u7ec4\u88c5\u590d\u5408\u7269\u5165\u6838"},
            {"n": "SBE-\u9776\u57fa\u56e0", "m": "\u7ed3\u5408 SBE \u2192 \u751f\u957f\u6291\u5236/EMT \u57fa\u56e0\u8f6c\u5f55"}],
        "er": [
            {"n": "ER\uff08\u914d\u4f53\u7ed3\u5408\u53d8\u6784\uff09", "m": "\u96cc\u6fc0\u7d20\u7ed3\u5408 ER \u914d\u4f53\u7ed3\u5408\u57df \u2192 \u53d7\u4f53\u6784\u8c61\u6539\u53d8\u3001\u70ed\u4e0c\u514b\u86cb\u767d\u89e3\u79bb"},
            {"n": "ER \u4e8c\u805a\u4f53\uff08\u5165\u6838\uff09", "m": "ER \u4e8c\u805a\u5316\u5e76\u8f6c\u4f4d\u5165\u6838"},
            {"n": "ERE-\u9776\u57fa\u56e0", "m": "\u7ed3\u5408\u96cc\u6fc0\u7d20\u5e94\u7b54\u5143\u4ef6 \u2192 \u589e\u6b96/\u5206\u6ccc\u57fa\u56e0\u8f6c\u5f55"}],
        "hif": [
            {"n": "HIF-\u03b1\uff08\u7a33\u5b9a\uff09", "m": "\u7f3a\u6c27 \u2192 HIF-\u03b1 \u812f\u6c28\u9178\u70be\u5316\u53d7\u6291 \u2192 \u86cb\u767d\u7a33\u5b9a"},
            {"n": "HIF-\u03b1/\u03b2 \u4e8c\u805a\u4f53\uff08\u5165\u6838\uff09", "m": "HIF-\u03b1 \u4e0e HIF-\u03b2 \u4e8c\u805a\u5316\u5165\u6838"},
            {"n": "HRE-\u9776\u57fa\u56e0", "m": "\u7ed3\u5408\u7f3a\u6c27\u5e94\u7b54\u5143\u4ef6 \u2192 \u8840\u7ba1\u65b0\u751f/\u7cd6\u9178\u89e3\u57fa\u56e0\u8f6c\u5f55"}],
        "rhogtp": [
            {"n": "RhoA/Rac\uff08\u88c5\u8f7d GTP\uff09", "m": "\u53d7\u4f53\u6fc0\u6d3b GEF \u2192 RhoA/Rac \u4ea4\u6362 GDP \u4e3a GTP"},
            {"n": "PAK/ROCK", "m": "GTP \u578b Rho \u6fc0\u6d3b\u4e0b\u6e38 PAK/ROCK"},
            {"n": "\u808c\u52a8\u86cb\u767d\u9aa8\u67b6", "m": "\u808c\u52a8\u86cb\u767d\u91cd\u6392 \u2192 \u8fc1\u79fb/\u6536\u7f29/\u5206\u88c2"}],
        "ampk": [
            {"n": "AMPK\uff08\u6fc0\u6d3b\uff09", "m": "AMP/ATP \u5347\u9ad8 \u2192 AMPK \u88ab LKB1/CaMKK2 \u78f7\u9178\u5316\u6fc0\u6d3b"},
            {"n": "AMPK-\u4ee3\u8c22\u9176", "m": "AMPK \u78f7\u9178\u5316\u4e0b\u6e38\u4ee3\u8c22\u9176\uff08ACC\u3001ULK1\uff09"},
            {"n": "\u5206\u89e3\u4ee3\u8c22\u7a0b\u5e8f", "m": "\u6291\u5236\u5408\u6210\u4ee3\u8c22\u3001\u4fc3\u8fdb\u5206\u89e3\u4ea7\u80fd"}],
        "epigen": [
            {"n": "\u8868\u89c2\u9176\uff08\u4fee\u9970\u6539\u53d8\uff09", "m": "\u8868\u89c2\u9057\u4f20\u9176\u6d3b\u6027\u6539\u53d8 \u2192 \u7ec4\u86cb\u767d/DNA \u4fee\u9970\u91cd\u7f16\u7a0b"},
            {"n": "\u67d3\u8272\u8d28\uff08\u72b6\u6001\u6539\u53d8\uff09", "m": "\u67d3\u8272\u8d28\u5f00\u653e/\u5173\u95ed\u72b6\u6001\u91cd\u7f16\u7a0b"},
            {"n": "\u57fa\u56e0\u8868\u8fbe\u8c31", "m": "\u57fa\u56e0\u8868\u8fbe\u8c31\u5927\u89c4\u6a21\u6539\u53d8"}],
        "tfdev": [
            {"n": "\u8f6c\u5f55\u56e0\u5b50\uff08\u6d3b\u6027\u6539\u53d8\uff09", "m": "\u8f6c\u5f55\u56e0\u5b50\u6d3b\u6027/\u4e30\u5ea6\u6539\u53d8"},
            {"n": "\u987a\u5f0f\u5143\u4ef6\uff08\u7ed3\u5408\u6539\u53d8\uff09", "m": "\u9776\u57fa\u56e0\u542f\u52a8\u5b50\u7ed3\u5408\u8c31\u6539\u53d8"},
            {"n": "\u5206\u5316\u7a0b\u5e8f", "m": "\u5206\u5316/\u5e94\u6fc0\u7a0b\u5e8f\u91cd\u7f16\u7a0b"}],
        "gpcrsig": [
            {"n": "G \u86cb\u767d\uff08\u6fc0\u6d3b\uff09", "m": "\u914d\u4f53\u7ed3\u5408 GPCR \u2192 G \u86cb\u767d \u03b1 \u4e9a\u57fa\u4ea4\u6362 GTP \u6fc0\u6d3b"},
            {"n": "\u7b2c\u4e8c\u4fe1\u4f7f\uff08cAMP/Ca\u00b2\u207a\uff09", "m": "\u817e\u82f7\u9178\u73af\u5316\u9176/\u78f7\u8102\u9176 C \u6d3b\u5316 \u2192 \u7b2c\u4e8c\u4fe1\u4f7f\u5347\u9ad8"},
            {"n": "\u4e0b\u6e38\u6fc0\u9176/\u79bb\u5b50\u901a\u9053", "m": "PKA/PKC/\u79bb\u5b50\u901a\u9053\u54cd\u5e94 \u2192 \u7ec6\u80de\u529f\u80fd\u6539\u53d8"}],
        "glycolysis": [
            {"n": "\u7cd6\u9178\u89e3\u9176\uff08\u6d3b\u6027\u6539\u53d8\uff09", "m": "\u7cd6\u9178\u89e3\u9176\u6d3b\u6027/\u8868\u8fbe\u6539\u53d8"},
            {"n": "\u4ee3\u8c22\u901a\u91cf\uff08\u91cd\u6392\uff09", "m": "\u8461\u8404\u7cd6\u4ee3\u8c22\u901a\u91cf\u91cd\u6392"},
            {"n": "\u4ea7\u80fd/\u5408\u6210\u9002\u5e94", "m": "\u7ec6\u80de\u80fd\u91cf\u4e0e\u751f\u7269\u5408\u6210\u9002\u5e94"}],
        "general": [
            {"n": "\u6270\u52a8\u86cb\u767d\u4fe1\u53f7", "m": "\u6270\u52a8\u86cb\u767d\u4fe1\u53f7\u8fdb\u5165\u7ec6\u80de"},
            {"n": "\u76f8\u4f3c\u86cb\u767d\u7f51\u7edc\uff08\u653e\u5927\uff09", "m": "\u901a\u8fc7\u76f8\u4f3c\u86cb\u767d\u7f51\u7edc\u9010\u7ea7\u653e\u5927"},
            {"n": "\u8f6c\u5f55\u7ec4\uff08\u91cd\u7f16\u7a0b\uff09", "m": "\u8f6c\u5f55\u7ec4\u91cd\u7f16\u7a0b \u2192 \u7ec6\u80de\u8868\u578b\u6539\u53d8"}],
    }
'''
src = src[:start] + new_lib + src[end:]
open('unipret/serve_api.py', 'w', encoding='utf-8').write(src)
import ast
ast.parse(src)
print('TRANSFER_LIB v3 写入成功，语法 OK')
