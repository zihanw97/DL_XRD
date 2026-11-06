"""Post-selection evaluation and exact 256-D pre-head exports."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .data import loader
from .training import forward_components


def probability_metrics(log_probabilities, labels, bins=15):
    probability = np.exp(log_probabilities)
    confidence = probability.max(1)
    prediction = probability.argmax(1)
    one_hot = np.eye(3)[labels]
    complete_label_space=len(set(labels))==3
    accuracy = float(np.mean(prediction == labels))
    ece = 0.0
    for i in range(bins):
        mask = (confidence > i/bins) & (confidence <= (i+1)/bins)
        if mask.any():
            ece += mask.mean()*abs(np.mean(prediction[mask] == labels[mask])-confidence[mask].mean())
    return dict(n=len(labels), accuracy=accuracy,
        balanced_accuracy=float(balanced_accuracy_score(labels, prediction)) if complete_label_space else None,
        macro_f1=float(f1_score(labels, prediction, labels=[0,1,2], average="macro", zero_division=0)) if complete_label_space else None,
        nll=float(-log_probabilities[np.arange(len(labels)), labels].mean()),
        brier=float(np.square(probability-one_hot).sum(1).mean()), ece=float(ece),
        confusion_matrix=confusion_matrix(labels,prediction,labels=[0,1,2]).tolist())


@torch.no_grad()
def export_selected(model, store, rows, destination, *, device, batch_size=32):
    destination = Path(destination)
    model.eval()
    features, logits, variances, log_probs = [], [], [], []
    for images, _ in loader(store, rows, batch_size=batch_size):
        z, mean, phi = forward_components(model, images.to(device))
        variance = model.gp_head.predictive_variance(phi) if phi is not None else torch.zeros((len(mean),1),device=device)
        adjusted = mean / torch.sqrt(1.0+(np.pi/8)*variance)
        features.append(z.cpu().numpy());logits.append(mean.cpu().numpy())
        variances.append(variance.cpu().numpy());log_probs.append(F.log_softmax(adjusted,dim=1).cpu().numpy())
    features, logits, variances, log_probs = map(np.concatenate,[features, logits, variances, log_probs])
    truth = np.array([r["label"] for r in rows], dtype=np.int64)
    domains = np.array([r["domain"] for r in rows])
    splits = np.array([r["split"] for r in rows])
    probabilities = np.exp(log_probs)
    arrays = dict(prehead_features=features,mean_logits=logits,gp_logit_variance=variances,
        probabilities=probabilities,log_probabilities=log_probs,labels=truth,
        sample_ids=np.array([r["sample_id"] for r in rows]),domain=domains,split=splits,
        materials=np.array([r["material"] for r in rows]),
        training_labels=np.array([r["training_label"] for r in rows],dtype=np.int64))
    np.savez_compressed(destination/'selected_predictions_and_prehead.npz',**arrays)
    with (destination/'predictions.csv').open('w',newline='') as handle:
        fields=['sample_id','material','domain','split','true_label','training_label','predicted_label','confidence','gp_logit_variance','probability_BCC','probability_FCC','probability_HCP']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for i,row in enumerate(rows):
            writer.writerow(dict(sample_id=row['sample_id'],material=row['material'],domain=row['domain'],
                split=row['split'],true_label=truth[i],training_label=row['training_label'],
                predicted_label=int(probabilities[i].argmax()),confidence=float(probabilities[i].max()),
                gp_logit_variance=float(variances[i,0]) if hasattr(model,'gp_head') else '',
                probability_BCC=probabilities[i,0],probability_FCC=probabilities[i,1],probability_HCP=probabilities[i,2]))
    metrics={}
    for role in ['train','val','internal_test','outer_test']:
        for domain in ['source','target']:
            mask=(splits==role)&(domains==domain)
            if mask.any():
                value=probability_metrics(log_probs[mask],truth[mask])
                if len(set(truth[mask]))<3:
                    value['balanced_accuracy']=None
                    value['macro_f1']=None
                metrics[domain+'/'+role]=value
    validation=splits=='val'
    scaled=StandardScaler().fit_transform(features[validation])
    val_labels=truth[validation];val_domains=domains[validation]
    gaps=[]
    for label in range(3):
        ms=(val_labels==label)&(val_domains=='source');mt=(val_labels==label)&(val_domains=='target')
        if ms.any() and mt.any():gaps.append(float(np.linalg.norm(scaled[ms].mean(0)-scaled[mt].mean(0))))
    geometry=dict(pool='validation only; post-selection diagnostic, not used for fitting or selection',
        representation='raw256Dpre-head, then joint validation-fitted StandardScaler',
        class_silhouette=float(silhouette_score(scaled,val_labels)) if 1<len(set(val_labels))<len(val_labels) else None,
        mean_class_centroid_distance=float(np.mean(gaps)) if gaps else None)
    report=dict(metrics=metrics,validation_geometry=geometry,
        probability_method='analytical mean-field pi/8' if hasattr(model,'gp_head') else 'ordinary softmax',
        cnn_variance_placeholder=not hasattr(model,'gp_head'),
        experimental_training_accuracy='All training labels joined for reporting only, after selection lock; hidden labels never optimized.',
        test_results_used_for_selection=False)
    (destination/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
