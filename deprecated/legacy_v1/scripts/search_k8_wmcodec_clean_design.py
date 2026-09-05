#!/usr/bin/env python3
"""Search an exact-column-count K=8 design that WMCodec decodes 8/8 cleanly."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from registry import full_registry_bits, get_or_embed  # noqa: E402
from watermarks import detect  # noqa: E402

COUNTS=np.asarray([0,1,2,3,4,4,5,6,7,8,4,4,4,4,4,4],dtype=int)


def args():
    p=argparse.ArgumentParser()
    p.add_argument("--speaker",default="chinese:SSB0005")
    p.add_argument("--max-candidates",type=int,default=30)
    p.add_argument("--seed",type=int,default=20260830)
    p.add_argument("--output-dir",type=Path,default=ROOT/"data"/"k8_constructed_payload_search_20260829")
    return p.parse_args()


def bits_to_int(row): return int(sum(int(v)<<i for i,v in enumerate(row)))


def matrix(rng):
    while True:
        m=np.zeros((8,16),dtype=np.int8)
        for j,c in enumerate(COUNTS):
            if c: m[rng.choice(8,int(c),replace=False),j]=1
        if len({tuple(x) for x in m[:,:10]})<8: continue
        d16=[np.count_nonzero(m[i]!=m[j]) for i in range(8) for j in range(i)]
        d10=[np.count_nonzero(m[i,:10]!=m[j,:10]) for i in range(8) for j in range(i)]
        if min(d16)>=5 and min(d10)>=2: return m,min(d16),min(d10)


def main():
    a=args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(a.seed); registry=full_registry_bits("wmcodec")
    records=[]; best=(-1,None)
    for ci in range(a.max_candidates):
        m,h16,h10=matrix(rng); payloads=[bits_to_int(r) for r in m]
        rows=[]; exact=0
        for i,payload in enumerate(payloads):
            wav=get_or_embed("wmcodec",a.speaker,payload)
            _,_,hard=detect("wmcodec",wav,registry)
            decoded=bits_to_int(hard)
            ok=int(decoded==payload); exact+=ok
            rows.append({"index":i+1,"target":payload,"decoded":decoded,"exact":ok})
        rec={"candidate":ci,"payloads":payloads,"payload_bits":m.astype(int).tolist(),
             "ones_per_bit":m.sum(0).astype(int).tolist(),"min_hamming_16":int(h16),
             "min_hamming_10":int(h10),"clean_exact":exact,"members":rows}
        records.append(rec)
        tmp=a.output_dir/"search_progress.json.tmp"
        tmp.write_text(json.dumps({"seed":a.seed,"tested":len(records),"records":records},indent=2)+"\n")
        os.replace(tmp,a.output_dir/"search_progress.json")
        print(f"candidate={ci} exact={exact}/8 payloads={payloads}",flush=True)
        if exact>best[0]: best=(exact,rec)
        if exact==8:
            out=a.output_dir/"selected_design.json"
            out.write_text(json.dumps(rec,indent=2)+"\n")
            print(f"SUCCESS {out}",flush=True); return
    out=a.output_dir/"best_design.json"
    out.write_text(json.dumps(best[1],indent=2)+"\n")
    raise RuntimeError(f"no 8/8 design in {a.max_candidates}; best={best[0]}/8")


if __name__=="__main__": main()
