# -*- coding: utf-8 -*-
"""加 /cells 接口 + 前端细胞系动态加载"""
import ast

# 1) serve_g2cp.py 加 /cells
p = "serve_g2cp.py"
s = open(p, encoding="utf-8").read()
old = '@app.route("/health")'
new = '''@app.route("/cells")
def cells():
    return jsonify({"cells": cl_names, "n": len(cl_names)})


@app.route("/health")'''
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
ast.parse(s)
print("serve_g2cp: /cells 已加")

# 2) g2cp.html：cell 下拉改动态 + 初始化 fetch
p2 = "data/gui/g2cp.html"
s = open(p2, encoding="utf-8").read()
s = s.replace('<select id="cell"><option value="0">A375</option><option value="1">A549</option><option value="2">HT29</option><option value="3">MCF7</option><option value="4">PC3</option></select>',
              '<select id="cell"><option value="0">A375</option></select>')
s = s.replace('<select id="cellG"><option value="0">A375</option><option value="1">A549</option><option value="2">HT29</option><option value="3">MCF7</option><option value="4">PC3</option></select>',
              '<select id="cellG"><option value="0">A375</option></select>')
old3 = 'fetch(API+"/genes")'
new3 = '''fetch(API+"/cells").then(r=>r.json()).then(j=>{
  const cs=(j.cells||[]);
  const pick=cs.filter(c=>["A375","A549","HT29","MCF7","PC3","HCT116","K562","HELA","HEPG2","U2OS","MDAMB231","JURKAT","T47D","SKBR3","H1299","U251MG"].includes(c));
  const opts=(pick.length?pick:cs.slice(0,16)).map((c,i)=>`<option value="${i}">${esc(c)}</option>`).join("");
  document.getElementById("cell").innerHTML=opts+(cs.length>16?`<option value="">...共 ${cs.length} 系（/cells）</option>`:"");
  document.getElementById("cellG").innerHTML=opts;
}).catch(()=>{});
fetch(API+"/genes")'''
assert old3 in s
s = s.replace(old3, new3, 1)
open(p2, "w", encoding="utf-8").write(s)
print("g2cp.html: 动态细胞系已加")
