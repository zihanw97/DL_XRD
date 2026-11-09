"""Train matched fixed models, or validate the clean paper recipe without training."""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import torch
from fixed.data import read_manifest,validate_fold,ImageStore
from fixed.evaluation import export_selected
from fixed.ood import evaluate_ood
from fixed.training import train_selected
from fixed.config import validate_recipe

METHODS=['cnn','cnn_sngp','cnn_sngp_adapt']


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',choices=['scan','grouped'],default='scan')
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--manifest-metadata',type=Path,help='Registration JSON with requested anchor fraction and manifest/cache hashes')
    parser.add_argument('--cache',type=Path,default=ROOT/'data/cache/fixed_images.npz')
    parser.add_argument('--recipe',type=Path,default=Path(__file__).with_name('paper_recipe.json'))
    parser.add_argument('--fold',help='Exact fold ID; absent means every registered fold in the manifest')
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--train',action='store_true',help='Explicitly start training; otherwise only validate the plan')
    action.add_argument('--dry-run',action='store_true',help='Read and validate manifests without loading images or allocating a model')
    parser.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/fixed')
    parser.add_argument('--save-weights',action='store_true',help='Optional runtime export; no weights are included in this package')
    parser.add_argument('--ood-manifest',type=Path,default=ROOT/'data/manifests/ood.csv')
    parser.add_argument('--skip-ood',action='store_true',help='Skip post-selection OOD predictions; no effect on fitting')
    args=parser.parse_args(argv)
    manifest=args.manifest or ROOT/'data/manifests/fixed'/f'{args.protocol}.csv'
    manifest_metadata=args.manifest_metadata or manifest.parent/'protocol.json'
    config=validate_recipe(json.loads(args.recipe.read_text()))
    resolved_output=args.output.resolve()
    if resolved_output.is_relative_to(ROOT) and not resolved_output.is_relative_to(ROOT/'outputs'):
        raise ValueError('Runtime outputs inside this package must be under outputs/, never code or data')
    registered=json.loads(manifest_metadata.read_text())
    if registered.get('schema')!='fixed_manuscript_partitions_v1':
        raise ValueError('Unknown manifest registration schema')
    requested=registered['requested_target_anchor_fraction']
    if requested!=config['training']['requested_target_anchor_fraction']:
        raise ValueError('Recipe requested anchor fraction disagrees with the registered immutable masks')
    manifest_hash=digest(manifest)
    if manifest_hash!=registered['protocols'][args.protocol]['sha256']:
        raise ValueError('Manifest hash does not match its protocol registration')
    rows=read_manifest(manifest,args.fold)
    folds=sorted({r['fold'] for r in rows})
    if not folds:raise ValueError('No matching fold')
    plans=[validate_fold([r for r in rows if r['fold']==fold]) for fold in folds]
    if not args.cache.is_file():raise FileNotFoundError(args.cache)
    if args.train and not args.skip_ood and not args.ood_manifest.is_file():
        raise FileNotFoundError(args.ood_manifest)
    plan=dict(protocol=args.protocol,methods=args.methods,n_folds=len(folds),
        model_fits=len(folds)*len(args.methods),architecture=config['architecture'],
        training=config['training'],folds=plans,
        requested_target_anchor_fraction_registered=requested,
        manifest_registration=str(manifest_metadata),manifest_sha256_verified=True,
        ood_manifest=str(args.ood_manifest),ood_after_selection=not args.skip_ood,
        data_files_exist=True,training_started=args.train,
        historical_refined_results_reproduced=False)
    if not args.train:
        print(json.dumps(plan,indent=2));return plan
    cache_hash=digest(args.cache)
    if cache_hash!=registered['fixed_images']['sha256']:
        raise ValueError('Image cache hash does not match the manifest registration')
    store=ImageStore(args.cache);store.verify_rows(rows)
    for fold in folds:
        current=[r for r in rows if r['fold']==fold]
        repeat=int(current[0].get('repeat') or 1)
        seed=int(registered['seeds']['initialization'][repeat-1])
        for method in args.methods:
            output=args.output/args.protocol/fold/method
            output.mkdir(parents=True,exist_ok=True)
            if any(output.iterdir()):raise FileExistsError(f'Refusing to overwrite existing run: {output}')
            source=dict(recipe=config,manifest=str(manifest.resolve()),manifest_sha256=manifest_hash,
                manifest_registration=registered,
                cache=str(args.cache.resolve()),cache_sha256=cache_hash,fold=fold,
                seed=seed,method=method,supervision=validate_fold(current))
            (output/'configuration.json').write_text(json.dumps(source,indent=2)+'\n')
            def callback(row):
                with (output/'epochs.jsonl').open('a') as handle:handle.write(json.dumps(row)+'\n')
                print(f"{fold} {method} epoch {row['epoch']} valCE={row['validation_raw_ce']:.5g} EMA={row['validation_ema']:.5g}",flush=True)
            model,result=train_selected(method,store,current,config,seed=seed,device=args.device,epoch_callback=callback)
            (output/'selection_lock.json').write_text(json.dumps({k:v for k,v in result.items() if k!='history'},indent=2)+'\n')
            export_selected(model,store,current,output,device=args.device,batch_size=config['training']['batch_size'])
            if not args.skip_ood:
                evaluate_ood(model,args.ood_manifest,ROOT,output,device=args.device,batch_size=config['training']['batch_size'])
            (output/'training_result.json').write_text(json.dumps(result,indent=2)+'\n')
            if args.save_weights:torch.save(model.state_dict(),output/'selected_weights.pt')
            del model
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
    return plan


if __name__=='__main__':main()
