"""Small numerical checks; never construct or train the full manuscript CNN."""
import math
import unittest
from unittest.mock import patch
from pathlib import Path
import json
from types import SimpleNamespace

import torch

from fixed.data import validate_fold
from fixed.losses import anchor_cross_entropy,anchor_prototype_alignment,fixed_objective,gaussian_mmd
from fixed.training import ValidationEMA,train_selected
from fixed.ood import validation_threshold,detection_metrics
from fixed.config import validate_recipe
import numpy as np


class FixedObjectiveTests(unittest.TestCase):
    def test_recipe_rejects_architecture_and_refinement_misdeclarations(self):
        load=lambda:json.loads((Path(__file__).parents[1]/'paper_recipe.json').read_text())
        validate_recipe(load())
        wrong=load();wrong['architecture']['cnn_head']='pool'
        with self.assertRaises(ValueError):validate_recipe(wrong)
        wrong=load();wrong['training']['cluster_refinement']=True
        with self.assertRaises(ValueError):validate_recipe(wrong)
        wrong=load();wrong['training']['optimizer']='AdamW'
        with self.assertRaises(ValueError):validate_recipe(wrong)
        wrong=load();wrong['training']['learning_rate']=float('nan')
        with self.assertRaises(ValueError):validate_recipe(wrong)
    def test_hidden_target_predictions_cannot_change_baseline_loss(self):
        zs=torch.tensor([[1.,0.],[0.,1.]])
        zt=torch.tensor([[.8,.2],[.2,.8]])
        ls=torch.tensor([[2.,0.,0.],[0.,2.,0.]])
        lt=torch.tensor([[2.,0.,0.],[0.,2.,0.]],requires_grad=True)
        labels=torch.tensor([0,1]);target_labels=torch.tensor([0,-1])
        total,_=fixed_objective(zs,ls,labels,zt,lt,target_labels,adapted=False)
        total.backward()
        self.assertTrue(torch.equal(lt.grad[1],torch.zeros(3)))
        self.assertGreater(float(lt.grad[0].abs().sum()),0)

    def test_prototypes_ignore_hidden_labels_and_use_raw_squared_gap(self):
        zs=torch.tensor([[1.,0.],[1.,0.]])
        zt=torch.tensor([[3.,0.],[1000.,1000.]])
        value=anchor_prototype_alignment(zs,torch.tensor([0,0]),zt,torch.tensor([0,-1]))
        self.assertEqual(float(value),4.)
        self.assertEqual(float(anchor_cross_entropy(torch.zeros(2,3),torch.tensor([-1,-1]))),0.)

    def test_mmd_normalizes_features_and_handles_empty_hidden_set(self):
        x=torch.tensor([[1.,0.],[0.,1.]])
        self.assertAlmostEqual(float(gaussian_mmd(x,10*x)),0.,places=6)
        self.assertEqual(float(gaussian_mmd(x,x[:0])),0.)

    def test_selection_uses_validation_ema_not_test_or_geometry(self):
        state=ValidationEMA(alpha=.2,minimum_improvement=0.)
        self.assertTrue(state.update(2.))
        self.assertTrue(state.update(1.))
        self.assertAlmostEqual(state.best,1.8)
        self.assertFalse(state.update(3.))
        self.assertEqual(state.bad_epochs,1)
        with self.assertRaises(FloatingPointError):state.update(math.nan)

    def test_roles_and_heldout_material_are_disjoint(self):
        rows=[]
        for domain in ['source','target']:
            for split in ['train','val']:
                rows.append(dict(fold='repeat_01/holdout_Cu',sample_id=domain+split,domain=domain,
                    material='Al',heldout='Cu',label=1,training_label=1 if split=='train' else -1,split=split))
        self.assertEqual(validate_fold(rows)['n'],4)
        leaked=[dict(r) for r in rows];leaked[0]['material']='Cu'
        with self.assertRaises(ValueError):validate_fold(leaked)
        duplicate=[dict(r) for r in rows];duplicate[1]['sample_id']=duplicate[0]['sample_id']
        with self.assertRaises(ValueError):validate_fold(duplicate)

    def test_ood_threshold_is_validation_only_and_retains_ties(self):
        validation=np.array([.6,.6,.7,.8,.9])
        threshold=validation_threshold(validation,rejection_fraction=.2)
        self.assertEqual(threshold,.6)
        result=detection_metrics(np.array([.8,.9]),np.array([.4,.5]),threshold)
        self.assertEqual(result['auroc'],1.)
        self.assertEqual(result['ood_false_accept_rate'],0.)
        self.assertEqual(result['id_false_reject_rate'],0.)

    def test_training_never_forwards_test_inputs_before_selection_lock(self):
        class TinyClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.project=torch.nn.Linear(2,2)
                self.classifier=torch.nn.Linear(2,3)

            def features(self,x):
                # Test rows are conspicuous sentinels. This synthetic model
                # checks access scope without allocating the huge real CNN.
                if bool((x>=100).any()):
                    raise AssertionError('Test data reached fitting/selection')
                return torch.relu(self.project(x))

        rows=[];images=[]
        for domain in ['source','target']:
            for role,n in [('train',3),('val',2),('outer_test',2)]:
                for i in range(n):
                    images.append([1000.,1000.] if role=='outer_test' else [float(i)/3,1.])
                    rows.append(dict(domain=domain,split=role,label=i%3,
                        training_label=i%3 if domain=='source' or i==0 else -1,
                        cache_index=len(images)-1))
        store=SimpleNamespace(images=torch.tensor(images))
        recipe=json.loads((Path(__file__).parents[1]/'paper_recipe.json').read_text())
        recipe['training'].update(max_epochs=2,minimum_epochs=1,patience=1,batch_size=2)
        with patch('fixed.training.build_model',lambda method,seed:TinyClassifier()):
            _,result=train_selected('cnn',store,rows,recipe,seed=42,device='cpu')
        self.assertTrue(result['selection_locked'])
        self.assertFalse(result['test_metrics_used_for_selection'])
        self.assertLessEqual(result['epochs_completed'],2)


if __name__=='__main__':unittest.main()
