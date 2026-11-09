"""Regenerate the paper-matched fixed-model partitions from packaged inputs.

Without --output-dir this only checks that the supplied manifests match the
declared recipe. No image pixels are changed and no model is constructed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import math
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str,parts)).encode()).digest()[:8],'little')


def build(cache_path):
    with np.load(cache_path,allow_pickle=False) as cache:
        sample_ids=cache['sample_ids'].astype(str)
        domains=cache['domains'].astype(str)
        materials=cache['materials'].astype(str)
        labels=cache['labels']
    if len(sample_ids)!=1701 or len(set(sample_ids))!=1701:
        raise ValueError('Expected the registered 1,701 known-class input identities')
    ordered=[dict(sample_id=s,domain=d,material=m,label=int(y)) for s,d,m,y in
             zip(sample_ids,domains,materials,labels)]
    manifests={'scan':[],'grouped':[]};counts=[]
    for protocol in ['scan','grouped']:
        for repeat in range(1,4):
            partition_seed=141+repeat;anchor_seed=241+repeat
            holdouts=['scan'] if protocol=='scan' else ['Al','Cu','Fe','Mo','Nb','Ni','Ti','Zn']
            for heldout in holdouts:
                fold=f'repeat_{repeat:02d}/'+('scan' if protocol=='scan' else f'holdout_{heldout}')
                roles={};eligible=defaultdict(list)
                for i,row in enumerate(ordered):
                    outer=(row['material']==heldout) if protocol=='grouped' else (
                        (row['domain']=='source' and row['material'] in ['Ag','Ti','V']) or
                        (row['domain']=='target' and row['material']=='Cu'))
                    if outer:roles[i]='outer_test'
                    else:eligible[row['domain'],row['label']].append(i)
                for (domain,label),idx in sorted(eligible.items()):
                    rng=np.random.default_rng(stable_seed(partition_seed,protocol,heldout,domain,label))
                    idx=rng.permutation(np.asarray(idx,dtype=int));n_val=len(idx)//10;n_test=len(idx)//10
                    for i in idx[:n_val]:roles[int(i)]='val'
                    for i in idx[n_val:n_val+n_test]:roles[int(i)]='internal_test'
                    for i in idx[n_val+n_test:]:roles[int(i)]='train'
                anchors=set()
                for label in range(3):
                    idx=[i for i,row in enumerate(ordered) if row['domain']=='target' and row['label']==label and roles[i]=='train']
                    rng=np.random.default_rng(stable_seed(anchor_seed,protocol,heldout,label))
                    n_anchor=min(len(idx),max(1,int(math.floor(.1*len(idx)+.5)))) if idx else 0
                    anchors.update(int(i) for i in rng.permutation(idx)[:n_anchor])
                fold_rows=[]
                for i,row in enumerate(ordered):
                    split=roles[i]
                    if split!='train':status='not_training';training_label=-1
                    elif row['domain']=='source':status='labeled_source';training_label=row['label']
                    elif i in anchors:status='anchor';training_label=row['label']
                    else:status='hidden';training_label=-1
                    fold_rows.append(dict(sample_id=row['sample_id'],domain=row['domain'],material=row['material'],
                        label=row['label'],split=split,training_label=training_label,anchor_status=status,
                        cache_index=i,repeat=repeat,heldout=heldout,fold=fold))
                manifests[protocol].extend(fold_rows)
                for domain in ['source','target']:
                    part=[r for r in fold_rows if r['domain']==domain]
                    counts.append(dict(protocol=protocol,fold=fold,repeat=repeat,domain=domain,
                        train=sum(r['split']=='train' for r in part),val=sum(r['split']=='val' for r in part),
                        internal_test=sum(r['split']=='internal_test' for r in part),outer_test=sum(r['split']=='outer_test' for r in part),
                        anchors=sum(r['anchor_status']=='anchor' for r in part),
                        hidden_training=sum(r['anchor_status']=='hidden' for r in part),
                        initialization_seed=41+repeat,partition_seed=partition_seed,anchor_seed=anchor_seed))
    return dict(scan=manifests['scan'],grouped=manifests['grouped'],partition_counts=counts)


def write(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=DATA/'cache/fixed_images.npz')
    parser.add_argument('--check',action='store_true',help='Read-only check (also the default when --output-dir is omitted)')
    parser.add_argument('--output-dir',type=Path,help='Explicit directory to write regenerated CSVs; omitted means read-only consistency check')
    args=parser.parse_args()
    if args.check and args.output_dir is not None:parser.error('--check and --output-dir cannot be combined')
    generated=build(args.cache)
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        for name,rows in generated.items():write(args.output_dir/(name+'.csv'),rows)
        print('Wrote partition CSVs to',args.output_dir.resolve())
    else:
        for name,rows in generated.items():
            with (DATA/'manifests/fixed'/(name+'.csv')).open(newline='') as stream:
                actual=list(csv.DictReader(stream))
            expected=[{k:str(v) for k,v in row.items()} for row in rows]
            if actual!=expected:raise ValueError('Registered manifest does not match the paper recipe: '+name)
        print('Passed: all fixed-model partition rows reproduce from packaged metadata and registered seeds.')


if __name__=='__main__':main()
