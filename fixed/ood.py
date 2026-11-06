"""Post-selection noise/SiC predictions; OOD samples never update GP precision."""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score,roc_curve
import torch
import torch.nn.functional as F

from .training import forward_components
from data.preprocessing import load_image


def preprocess_tiff(path):
    return load_image(path)


def validation_threshold(confidence,rejection_fraction=.05):
    if not 0<=rejection_fraction<1 or not len(confidence):
        raise ValueError('Invalid validation rejection budget')
    return float(np.sort(confidence)[int(np.floor(rejection_fraction*len(confidence)))])


def detection_metrics(id_confidence,ood_confidence,threshold):
    labels=np.r_[np.zeros(len(id_confidence)),np.ones(len(ood_confidence))]
    uncertainty=1-np.r_[id_confidence,ood_confidence]
    fpr,tpr,_=roc_curve(labels,uncertainty)
    return dict(n_id=len(id_confidence),n_ood=len(ood_confidence),
        auroc=float(roc_auc_score(labels,uncertainty)),
        aupr_ood_positive=float(average_precision_score(labels,uncertainty)),
        fpr95=float(fpr[np.flatnonzero(tpr>=.95)[0]]),
        confidence_threshold=threshold,
        id_false_reject_rate=float(np.mean(id_confidence<threshold)),
        ood_false_accept_rate=float(np.mean(ood_confidence>=threshold)))


@torch.no_grad()
def evaluate_ood(model,manifest,root,output,*,device,batch_size=32):
    """Evaluate the same confirmed OOD pool for every selected fit.

    The ID reference is the pooled internal test (source+target), never the
    outer material holdout. A separate pooled validation set selects the
    threshold. OOD classes have no BCC/FCC/HCP ground truth or accuracy.
    """
    root,output=Path(root),Path(output)
    with Path(manifest).open(newline='') as handle:rows=list(csv.DictReader(handle))
    if not rows or not {'sample_id','category','image_path'}.issubset(rows[0]):
        raise ValueError('OOD manifest needs sample_id,category,image_path')
    if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Duplicate OOD sample IDs')
    model.eval();features=[];probabilities=[];variances=[];means=[]
    for start in range(0,len(rows),batch_size):
        current=rows[start:start+batch_size]
        images=[]
        for row in current:
            path=(root/row['image_path']).resolve()
            if not path.is_relative_to(root.resolve()):raise ValueError('OOD path escapes package')
            images.append(preprocess_tiff(path))
        images=torch.from_numpy(np.stack(images)).to(device)
        prehead,logits,phi=forward_components(model,images)
        variance=model.gp_head.predictive_variance(phi) if phi is not None else torch.zeros((len(logits),1),device=device)
        probs=F.softmax(logits/torch.sqrt(1+(np.pi/8)*variance),dim=1)
        features.append(prehead.cpu().numpy());means.append(logits.cpu().numpy())
        probabilities.append(probs.cpu().numpy());variances.append(variance.cpu().numpy())
    features,means,probabilities,variances=map(np.concatenate,[features,means,probabilities,variances])
    confidence=probabilities.max(1);categories=np.array([r['category'] for r in rows])
    np.savez_compressed(output/'ood_predictions_and_prehead.npz',prehead_features=features,
        mean_logits=means,probabilities=probabilities,gp_logit_variance=variances,
        sample_ids=np.array([r['sample_id'] for r in rows]),categories=categories,
        domain=np.array(['target']*len(rows)))
    with (output/'ood_predictions.csv').open('w',newline='') as handle:
        fields=['sample_id','category','predicted_label','confidence','ood_score','gp_logit_variance','probability_BCC','probability_FCC','probability_HCP']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for i,row in enumerate(rows):
            writer.writerow(dict(sample_id=row['sample_id'],category=row['category'],
                predicted_label=int(probabilities[i].argmax()),confidence=float(confidence[i]),
                ood_score=float(1-confidence[i]),
                gp_logit_variance=float(variances[i,0]) if hasattr(model,'gp_head') else '',
                probability_BCC=probabilities[i,0],probability_FCC=probabilities[i,1],probability_HCP=probabilities[i,2]))
    with np.load(output/'selected_predictions_and_prehead.npz',allow_pickle=False) as saved:
        reference_confidence=saved['probabilities'].max(1)
        validation=reference_confidence[saved['split']=='val']
        familiar=reference_confidence[saved['split']=='internal_test']
    report=dict(scope='Post-selection; no OOD accuracy; OOD inputs do not update model or GP precision',
        score='1-max predictive class probability; larger means more unfamiliar',
        reference='pooled source+target internal test',threshold_selection='pooled source+target validation; floor(0.05*N) rejection budget; accept confidence>=threshold, retain ties',
        categories={})
    if len(validation) and len(familiar):
        threshold=validation_threshold(validation)
        for category in sorted(set(categories)):
            report['categories'][category]=detection_metrics(familiar,confidence[categories==category],threshold)
    (output/'ood_detection.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
