# Shared utilities: config loading, donor parsing, logging
import os, time, yaml
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # gtex_reference_sensitivity/

def load_config():
    with open(os.path.join(BASE, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)

def donor_id(sample_id: str) -> str:
    """GTEX-1117F-0226-SM-5GZZ7 -> GTEX-1117F"""
    return "-".join(sample_id.split("-")[:2])

def log(msg: str):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)

def p(*parts) -> str:
    return os.path.join(BASE, *parts)


def iter_numeric_csv_chunks(path, id_cols, block_size=1 << 29):
    """Streaming pyarrow CSV parsing (multithreaded parse + native float32 cast + zero-copy column access).
    yield (gid_np_str, vals_float32(nrows×ncols), sample_names|None)。
    id_cols: the leading ID column names; gid is the 2nd one (GCT uses Name, RSEM uses gene_id)."""
    import pyarrow as pa
    import pyarrow.csv as pac
    ro = pac.ReadOptions(block_size=block_size)
    po = pac.ParseOptions(delimiter="\t")
    with pac.open_csv(path, read_options=ro, parse_options=po) as reader:
        names = reader.schema.names
        assert names[:len(id_cols)] == list(id_cols), names[:4]
        sample_names = names[len(id_cols):]
        # cast only the numeric columns to float32; keep ID columns as string
        target = pa.schema([(nm, pa.string() if nm in id_cols else pa.float32())
                            for nm in names])
        for batch in reader:
            tbl = pa.Table.from_batches([batch]).cast(target)
            gid = tbl.column(id_cols[1]).to_numpy(zero_copy_only=False)
            cols = [tbl.column(i).combine_chunks() for i in range(len(id_cols), len(names))]
            try:
                arrs = [c.to_numpy(zero_copy_only=True) for c in cols]
            except pa.ArrowInvalid:          # fall back to a copy when nulls are present
                arrs = [c.to_numpy(zero_copy_only=False) for c in cols]
            vals = np.stack(arrs, axis=1)
            yield gid, vals, sample_names


def iter_tsv_gz_blocks(path, skip_lines=1, n_id_cols=2, id_index=1, block_comp=64 << 20):
    """Streaming zlib decompression + chunked np.loadtxt parsing (numpy 2.x C implementation, ~73 MB/s, no per-column overhead).
    yield (ids:list[str], vals:float32(nrows×ncols))。
    skip_lines: number of header lines to skip; n_id_cols: number of leading ID columns; id_index: which ID column to use as ids."""
    import io
    import zlib
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    skip = skip_lines
    f = open(path, "rb")
    buf = ""
    while True:
        b = f.read(block_comp)
        buf += d.decompress(b).decode("utf-8", "replace") if b else d.flush().decode("utf-8", "replace")
        lines = buf.split("\n")
        buf = lines.pop()                      # keep the trailing partial line for the next round
        proc = []
        for l in lines:
            if skip > 0:
                skip -= 1
                continue
            if not l:
                continue
            proc.append(l.split("\t", n_id_cols))
        if proc:
            ids = [pp[id_index] for pp in proc]
            rest = "\n".join(pp[-1] for pp in proc)
            vals = np.loadtxt(io.StringIO(rest), dtype=np.float32, delimiter="\t")
            if vals.ndim == 1:
                vals = vals.reshape(1, -1)
            yield ids, vals
        if not b:
            break
    f.close()
