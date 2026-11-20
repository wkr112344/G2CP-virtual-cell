import csv, os, xml.etree.ElementTree as ET

INDIR = 'C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/figure_source_data'
OUTDIR = os.path.join(INDIR, 'xml')
os.makedirs(OUTDIR, exist_ok=True)

def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.reader(f))

def table_xml(rows, name):
    header = rows[0]
    root = ET.Element('table', name=name, nrow=str(len(rows)-1), ncol=str(len(header)))
    cols = ET.SubElement(root, 'columns')
    for c in header:
        ET.SubElement(cols, 'column').text = c
    data = ET.SubElement(root, 'rows')
    for row in rows[1:]:
        er = ET.SubElement(data, 'row')
        for i, c in enumerate(header):
            cell = ET.SubElement(er, 'c', name=c)
            cell.text = row[i] if i < len(row) else ''
    return ET.ElementTree(root)

def matrix_xml(rows, name):
    header = rows[0]
    rownames = [r[0] for r in rows[1:]]
    colnames = header[1:]
    root = ET.Element('matrix', name=name, nrow=str(len(rownames)), ncol=str(len(colnames)))
    rn = ET.SubElement(root, 'rownames')
    for n in rownames:
        ET.SubElement(rn, 'name').text = n
    cn = ET.SubElement(root, 'colnames')
    for n in colnames:
        ET.SubElement(cn, 'name').text = n
    data = ET.SubElement(root, 'data')
    for r in rows[1:]:
        er = ET.SubElement(data, 'row')
        for v in r[1:]:
            ET.SubElement(er, 'v').text = v
    return ET.ElementTree(root)

# (csv file, xml base name, kind)  kind: table | matrix
jobs = [
    ('TableS_seven_models.csv', 'TableS_seven_models', 'table'),
    ('DepMap_percell_PCC.csv', 'DepMap_percell_PCC', 'table'),
    ('DepMap_percell_delta_eval.csv', 'DepMap_percell_delta_eval', 'table'),
    ('TableS_disease_perdrug_scored.csv', 'TableS_disease_perdrug_scored', 'table'),
    ('DepMap_CERES_272x1244.csv', 'DepMap_CERES_272x1244', 'matrix'),
    ('DepMap_Chronos_272x1244.csv', 'DepMap_Chronos_272x1244', 'matrix'),
]

for fn, name, kind in jobs:
    rows = read_csv(os.path.join(INDIR, fn))
    tree = matrix_xml(rows, name) if kind == 'matrix' else table_xml(rows, name)
    ET.indent(tree)
    out = os.path.join(OUTDIR, name + '.xml')
    tree.write(out, encoding='utf-8', xml_declaration=True)
    print('wrote', out, os.path.getsize(out), 'bytes')
print('DONE')
