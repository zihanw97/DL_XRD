"""Verify packaged images, immutable CL roles and new fixed-model manifests.

This command performs no model construction, fitting or prediction. Add
--full to hash every raw TIFF; the default verifies cache/partition hashes
and every raw input's declared path, size and image count.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


DATA = Path(__file__).resolve().parent
ROOT = DATA.parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(16*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(full=False):
    manifest=json.loads((DATA/'cache/cache_manifest.json').read_text())
    for file,key in [('cache/xrd_dataset_cache.npz','destination_npz_sha256'),
                     ('cache/fixed_images.npz','fixed_images_npz_sha256'),
                     ('partitions/continual_partition.csv','continual_partition_sha256')]:
        require(sha(DATA/file)==manifest[key], 'Input hash mismatch: '+file)
    rows=read(DATA/'partitions/continual_partition.csv')
    require(len(rows)==1701, 'CL input count changed')
    require(len({r['sample_id'] for r in rows})==1701, 'Duplicate CL sample IDs')
    require(len({r['exact_float32_image_sha256'] for r in rows})==1701, 'Duplicate CL image fingerprints')
    require(sum(r['anchor_status']=='anchor' for r in rows)==127, 'Legacy CL source anchor identities/count changed')
    # Exact input pixel identities, independent of Python tensor serialization.
    with np.load(DATA/'cache/xrd_dataset_cache.npz',allow_pickle=False) as cache:
        require(set(cache.files)==set(manifest['keys']), 'Legacy data-cache keys changed')
        for key,info in manifest['keys'].items():
            require(list(cache[key].shape)==info['shape'] and str(cache[key].dtype)==info['dtype'],
                    'Data-cache shape/dtype changed: '+key)
        groups=defaultdict(list)
        for row in rows:groups[row['cache_prefix']].append(row)
        for prefix,part in groups.items():
            images,labels=cache[prefix+'_X'],cache[prefix+'_y']
            for row in part:
                i=int(row['cache_row'])
                require(int(labels[i])==int(row['label']), 'CL truth changed')
                require(hashlib.sha256(images[i].tobytes()).hexdigest()==row['exact_float32_image_sha256'],
                        'CL pixel identity changed: '+row['sample_id'])
    by_id={r['sample_id']:r for r in rows}
    with np.load(DATA/'cache/fixed_images.npz',allow_pickle=False) as cache:
        sample_ids=cache['sample_ids'].astype(str).tolist()
        fixed_images=cache['images']
        fixed_labels=cache['labels']
        fixed_domains=cache['domains'].astype(str)
        fixed_materials=cache['materials'].astype(str)
        require(len(sample_ids)==1701 and len(set(sample_ids))==1701, 'Fixed input identity count changed')
        require(fixed_images.shape==(1701,1,256,256) and fixed_images.dtype==np.float32,
                'Fixed input shape/dtype changed')
        for i,sample_id in enumerate(sample_ids):
            original=by_id[sample_id]
            require(int(fixed_labels[i])==int(original['label']), 'Fixed truth differs')
            require(fixed_domains[i]==original['domain'] and fixed_materials[i]==original['material'],
                    'Fixed sample metadata differs')
            require(hashlib.sha256(fixed_images[i].tobytes()).hexdigest()==original['exact_float32_image_sha256'],
                    'Fixed/CL input pixel mismatch: '+sample_id)
        labels=fixed_labels.copy()
        domains=fixed_domains.copy()
        materials=fixed_materials.copy()
    fold_counts={}
    fixed_protocol=json.loads((DATA/'manifests/fixed/protocol.json').read_text())
    require(fixed_protocol['requested_target_anchor_fraction']==.1, 'Requested fixed anchor budget changed')
    for protocol,expected_folds in [('scan',3),('grouped',24)]:
        require(sha(DATA/'manifests/fixed'/f'{protocol}.csv')==fixed_protocol['protocols'][protocol]['sha256'],
                'Fixed partition manifest hash changed: '+protocol)
        grouped=defaultdict(list)
        for row in read(DATA/'manifests/fixed'/f'{protocol}.csv'):grouped[row['fold']].append(row)
        require(len(grouped)==expected_folds, 'Fixed fold count changed')
        for fold,part in grouped.items():
            require(len(part)==1701 and len({r['cache_index'] for r in part})==1701, 'Missing/duplicate fixed rows')
            require(len({r['sample_id'] for r in part})==1701, 'Duplicate fixed input identity')
            target_train=Counter();target_anchors=Counter()
            for row in part:
                i=int(row['cache_index']); label=int(row['label']);split=row['split'];status=row['anchor_status']
                require(sample_ids[i]==row['sample_id'] and int(labels[i])==label and domains[i]==row['domain'] and materials[i]==row['material'], 'Fixed cache/manifest join mismatch')
                require(split in ['train','val','internal_test','outer_test'], 'Unknown fixed role')
                outer=(row['material']==row['heldout']) if protocol=='grouped' else (
                    (row['domain']=='source' and row['material'] in ['Ag','Ti','V']) or
                    (row['domain']=='target' and row['material']=='Cu'))
                require((split=='outer_test')==outer, 'Outer material leaked into development: '+fold)
                if split!='train':
                    require(status=='not_training' and int(row['training_label'])==-1, 'Nontraining labels exposed to optimizer')
                elif row['domain']=='source':
                    require(status=='labeled_source' and int(row['training_label'])==label, 'Source training labels changed')
                else:
                    target_train[label]+=1
                    require(status in ['anchor','hidden'], 'Bad target mask')
                    require(int(row['training_label'])==(label if status=='anchor' else -1), 'Hidden target truth exposed')
                    target_anchors[label]+=status=='anchor'
            for label,n in target_train.items():
                require(target_anchors[label]==min(n,max(1,int(math.floor(.1*n+.5)))), '10% target-anchor count changed')
        fold_counts[protocol]=len(grouped)
    raw=read(DATA/'metadata/raw_file_hashes.csv')
    require(len(raw)==1901, 'Raw input inventory changed')
    for row in raw:
        path=ROOT/row['path']
        require(path.is_file() and not path.is_symlink(), 'Missing or linked raw input: '+row['path'])
        require(path.stat().st_size==int(row['bytes']), 'Raw input size changed')
        if full:require(sha(path)==row['sha256'], 'Raw TIFF checksum changed: '+row['path'])
    require(len(list(DATA.rglob('*.tiff')))==1901, 'Undeclared raw TIFFs present')
    ood=read(DATA/'manifests/ood.csv')
    require(len(ood)==200 and Counter(r['category'] for r in ood)=={'noise_only':100,'SiC':100}, 'OOD manifest category counts changed')
    require(all((ROOT/r['image_path']).is_file() and r['sample_id'] not in by_id for r in ood), 'OOD inputs overlap known-class classification roles')
    return dict(status='passed',known_images=1701,raw_TIFFs=1901,OOD_inputs=200,
                fixed_folds=fold_counts,continual_source_anchors=127,
                full_raw_hash_check=bool(full),trained_or_predicted=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full',action='store_true',help='Hash every raw TIFF as well as caches and partition inputs')
    args=parser.parse_args()
    print(json.dumps(verify(args.full),indent=2))
