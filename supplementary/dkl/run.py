"""Portable entry point for the unchanged historical DKL control.

No historical weights are distributed. Training requires an explicitly supplied
SNGP encoder checkpoint; the default action only validates the data/recipe.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for folder in [ROOT, HERE]:
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_data(path):
    with np.load(path, allow_pickle=False) as archive:
        cache = {name: torch.from_numpy(archive[name].copy()) for name in archive.files}
    def pooled(split, domain):
        return TensorDataset(*(torch.cat([cache[f'{split}_{domain}_{block}_{field}']
                                          for block in (1, 2, 3)]) for field in ('X', 'y')))
    source = pooled('train', 'source')
    target = TensorDataset(cache['train_target_all_X_semi'], cache['train_target_all_y_semi'])
    assert len(source) == 602 and len(target) == 562
    assert int((target.tensors[1] >= 0).sum()) == 169
    assert int((target.tensors[1] == -1).sum()) == 393
    datasets = [source, target, pooled('val', 'source'), pooled('val', 'target')]
    return tuple(DataLoader(d, batch_size=32, shuffle=i < 2, num_workers=0,
                            generator=torch.Generator().manual_seed(43+i))
                 for i, d in enumerate(datasets))


@torch.no_grad()
def initialize(model, data, device):
    from training_utils import _evaluation_loader
    model.eval()
    features, labels, domains = [], [], []
    for domain, loader in enumerate(data[:2]):
        for x, y in _evaluation_loader(loader):
            features.append(model.features(x.to(device)).cpu())
            labels.append(y)
            domains.append(torch.full_like(y, domain))
    z = torch.cat(features).to(device)
    labels, domains = torch.cat(labels), torch.cat(domains)
    center = z.double().mean(0)
    scale = (z.double()-center).square().mean().sqrt().clamp_min(1e-4)
    model.configure_feature_transform(center, scale)
    return dict(scope='training inputs only', n=len(z), known_labels=int((labels >= 0).sum()),
                shared_feature_scale=float(scale),
                kernel=model.initialize_kernel_from_training(z, seed=42),
                inducing=model.initialize_inducing(z, labels, domains, seed=42))


@torch.no_grad()
def export_selected(model, cache_path, output, device):
    """Evaluate after validation selection; cached OOD inputs have no accuracy.

    Historical scan roles and full D4 material pools are retained. Stable IDs
    identify exact input cache rows; validation/test truth never enters fitting.
    """
    model.eval()
    metrics = {}
    with np.load(cache_path, allow_pickle=False) as archive:
        groups = {}
        for split in ['train', 'val', 'test']:
            for domain in ['source', 'target']:
                groups[f'D1_D3_{domain}_{split}'] = [f'{split}_{domain}_{b}' for b in (1, 2, 3)]
        groups['D4_source_full_holdout'] = [f'{s}_source_outlier' for s in ['train', 'val', 'test']]
        groups['D4_target_full_holdout'] = [f'{s}_target_4' for s in ['train', 'val', 'test']]
        groups['noise_cached_1s_subset'] = ['OOD_bad']
        groups['SiC_cached_1s_subset'] = ['OOD']
        for name, prefixes in groups.items():
            x = np.concatenate([archive[k+'_X'] for k in prefixes])
            y = np.concatenate([archive[k+'_y'] for k in prefixes])
            ids = np.concatenate([np.array([f'legacy_cache:{k}_X:{i:05d}' for i in range(len(archive[k+'_y']))]) for k in prefixes])
            features, probabilities = [], []
            for start in range(0, len(x), 32):
                z = model.features(torch.from_numpy(x[start:start+32]).to(device))
                p, _, _ = model.predictive(z, n_samples=128, seed=42)
                features.append(z.cpu().numpy()); probabilities.append(p.cpu().numpy())
            features, probabilities = np.concatenate(features), np.concatenate(probabilities)
            prediction = probabilities.argmax(1)
            known = not name.startswith(('noise_', 'SiC_'))
            np.savez_compressed(output/(name+'_predictions_prehead.npz'),
                                sample_ids=ids, features=features, probabilities=probabilities,
                                predicted_labels=prediction, true_labels=y if known else np.full(len(y), -1))
            metrics[name] = dict(n=len(y), correct=int((prediction==y).sum()) if known else None,
                                 accuracy=float((prediction==y).mean()) if known else None,
                                 probability_method='128 Sobol Gaussian-logit samples, seed42',
                                 selection_used=False)
    (output/'accuracy_metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT/'data/cache/xrd_dataset_cache.npz')
    parser.add_argument('--encoder-checkpoint', type=Path)
    parser.add_argument('--output', type=Path, default=HERE/'outputs/run_seed42')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    from dkl_training import effective_config, train_dkl
    cfg = effective_config(dict(seed=42))
    data = load_data(args.cache)
    plan = dict(training_started=args.train, configuration=cfg,
                training_counts=dict(source=602, target=562, anchors=169, hidden=393),
                target_training_anchor_fraction='historical 30%; separate from new fixed 10% recipe',
                encoder_checkpoint_required_for_training=True, historical_weights_included=False,
                architecture='original flatten CNN; SN dense512/256; three independent ARD-RBF variational GPs,64 inducing/class',
                predictive_probability='128 scrambled Sobol Gaussian-logit samples; seed42')
    if not args.train:
        print(json.dumps(plan, indent=2)); return plan
    if args.encoder_checkpoint is None or not args.encoder_checkpoint.is_file():
        parser.error('--train requires an explicitly supplied --encoder-checkpoint')
    output = args.output.resolve()
    if not output.is_relative_to(HERE) or output == HERE:
        raise ValueError('DKL output must be a new subdirectory under supplementary/dkl')
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError('Refusing to overwrite an existing DKL run; use --resume only for the same run')
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from models import SNGPWithCNN
    from dkl_model import CNNDeepKernelLearning
    encoder = SNGPWithCNN(1, 64, 3, rff_gamma=1/64, cnn_head='flatten', cnn_output_dim=2048)
    encoder.load_state_dict(torch.load(args.encoder_checkpoint, map_location='cpu', weights_only=True, mmap=True))
    device = torch.device(args.device)
    model = CNNDeepKernelLearning(encoder=encoder, num_inducing=64).to(device)
    del encoder
    output.mkdir(parents=True, exist_ok=True)
    plan['encoder_checkpoint_sha256'] = hashlib.sha256(args.encoder_checkpoint.read_bytes()).hexdigest()
    (output/'configuration.json').write_text(json.dumps(plan, indent=2)+'\n')
    if not args.resume:
        (output/'initialization.json').write_text(json.dumps(initialize(model, data, device), indent=2)+'\n')
    result = train_dkl(model, *data, device, output/'training', cfg, resume=args.resume)
    (output/'training_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    (output/'selection_lock.json').write_text(json.dumps(dict(
        selected_epoch=result['selected_epoch'], criterion='minimum predictive validation NLL',
        test_ood_or_geometry_used=False), indent=2)+'\n')
    export_selected(model, args.cache, output, device)
    return result


if __name__ == '__main__':
    main()
