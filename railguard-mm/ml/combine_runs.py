from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser(description='Combine processed Rail-VIVID run tables while preserving run IDs')
    p.add_argument('inputs', nargs='+', type=Path)
    p.add_argument('--out', required=True, type=Path)
    a=p.parse_args()
    tables=[]
    for path in a.inputs:
        df=pd.read_csv(path)
        if 'run_id' not in df:
            df['run_id']=path.stem
        tables.append(df)
    out=pd.concat(tables,ignore_index=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(a.out,index=False)
    print(f'wrote {len(out)} windows from {len(tables)} runs to {a.out}')
if __name__=='__main__': main()
