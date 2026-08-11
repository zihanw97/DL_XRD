"""Model 2/3: CNN-SNGP (no domain adaptation).

Corresponds to the CNN-SNGP row in the paper's Table 3/4 (Sec. 4.1-4.2): same
CNN-SNGP architecture as run_cnn_sngp_adapt.py (spectrally-normalized
backbone + random-Fourier-feature GP head), trained the same way (source CE
+ labeled-target CE), but with the MMD domain-alignment loss turned off
(lambda_mmd=0, lambda_classwise_mmd=0). This isolates the effect of domain
adaptation on top of the SNGP uncertainty head.

Run data_prep.py first to produce xrd_dataset_cache.pt.

Usage:
    python run_cnn_sngp.py

    # To skip training and just evaluate an existing checkpoint instead
    # (e.g. the legacy HTMAX one -- same SNGPWithCNN architecture):
    SKIP_TRAINING=1 python run_cnn_sngp.py
    SKIP_TRAINING=1 PRETRAINED_CHECKPOINT=/path/to/model.pth python run_cnn_sngp.py
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.nn as nn
import torch.optim as optim

from datasets import load_cache_and_build_loaders
from evaluation import (
    compute_confidence_from_model_outputs,
    compute_uncertainty_from_model_outputs_sngp,
    latent_quality_report,
    opentsne_plot_all_latents_by_class,
    overall_accuracy_summary,
    plot_tsne_latents,
    run_predictions_across_loaders,
)
from models import SNGPWithCNN
from training import train_domain_adapt_semi_supervised

STUDENT_PATH = "cnn_sngp_student.pth"

INPUT_CHANNELS = 1
RFF_DIM = 64
OUTPUT_DIM = 3
LR = 5e-5
NUM_EPOCHS = 800

# Set PRETRAINED_CHECKPOINT to an existing .pth file to load it directly and
# skip training entirely (e.g. the legacy HTMAX checkpoint below, which has
# the same SNGPWithCNN(input_channels=1, rff_dim=64, output_dim=3)
# architecture as this script). Override via env var without editing the
# file, e.g.:
#   PRETRAINED_CHECKPOINT=/path/to/model.pth python run_cnn_sngp.py
PRETRAINED_CHECKPOINT = os.environ.get(
    "PRETRAINED_CHECKPOINT",
    "/mnt/a/tgf4743/migrate/HTMAX/saved_models/sngp_cnn_model_base_combined.pth",
)
SKIP_TRAINING = os.environ.get("SKIP_TRAINING", "0") == "1"


def load_model(input_channels, rff_dim, output_dim, model_path=STUDENT_PATH):
    model = SNGPWithCNN(input_channels=input_channels, rff_dim=rff_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def main():
    print("CUDA Available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = load_cache_and_build_loaders()

    if SKIP_TRAINING:
        print(f"SKIP_TRAINING=1 -- loading pretrained checkpoint from {PRETRAINED_CHECKPOINT}")
        model = load_model(INPUT_CHANNELS, RFF_DIM, OUTPUT_DIM, model_path=PRETRAINED_CHECKPOINT)
        model = model.to(device)
    else:
        # ---- Define & train the model ----
        model = SNGPWithCNN(input_channels=INPUT_CHANNELS, rff_dim=RFF_DIM, output_dim=OUTPUT_DIM).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(list(model.parameters()), lr=LR)

        model = train_domain_adapt_semi_supervised(
            model,
            train_source_loader=loaders["train_source_loader_all"],
            train_target_loader=loaders["train_target_loader_all_semi"],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_epochs=NUM_EPOCHS,

            # Main losses -- no domain-alignment loss (that's what separates
            # this from run_cnn_sngp_adapt.py)
            source_weight=1.0,
            lambda_target_supervised=1.0,
            lambda_mmd=0.0,
            lambda_classwise_mmd=0.0,
            lambda_pseudo=0.1,

            # Semi-supervised target settings
            use_pseudo_labels=True,
            use_classwise_mmd=False,
            pseudo_confidence_threshold=0.90,
            unlabeled_value=-1,
            num_classes=3,

            # Early stopping
            early_stop=True,
            patience=30,
            min_delta=1e-4,
            smooth_alpha=0.2,
            warmup_epochs=5,

            # Saving
            save_model_path=STUDENT_PATH,
        )

        # ---- Reload best checkpoint for evaluation ----
        model = load_model(INPUT_CHANNELS, RFF_DIM, OUTPUT_DIM)
        model = model.to(device)

    # ---- Predictions across all loaders ----
    results = run_predictions_across_loaders(model, loaders, device)

    print("\n" + "=" * 70)
    print("Overall accuracy on the whole dataset")
    print("=" * 70)
    overall_accuracy_summary(results)

    print("\n" + "=" * 70)
    print("Latent-space visualization (t-SNE)")
    print("=" * 70)
    plot_tsne_latents(results)

    print("\n" + "=" * 70)
    print("Uncertainty & OOD detection (SNGP predictive uncertainty)")
    print("=" * 70)
    id_uncertainty = compute_uncertainty_from_model_outputs_sngp(model, loaders["train_source_loader_all"], device)
    ood_uncertainty_test = compute_uncertainty_from_model_outputs_sngp(model, loaders["test_source_loader_all"], device)
    print(f"ID (train_source_all) uncertainty:      mean={id_uncertainty.mean():.4f}")
    print(f"OOD (test_source_all) uncertainty:       mean={ood_uncertainty_test.mean():.4f}")

    ood_uncertainty_exp = compute_uncertainty_from_model_outputs_sngp(model, loaders["OOD_data_loader"], device)
    print(f"OOD (OOD_data_loader) uncertainty:       mean={ood_uncertainty_exp.mean():.4f}")

    ood_uncertainty_bad = compute_uncertainty_from_model_outputs_sngp(model, loaders["OOD_bad_data_loader"], device)
    print(f"OOD (OOD_bad_data_loader) uncertainty:   mean={ood_uncertainty_bad.mean():.4f}")

    print("\n" + "=" * 70)
    print("Confidence analysis")
    print("=" * 70)
    confidences = compute_confidence_from_model_outputs(model, loaders["train_source_loader_all"], device)
    print("Mean predictive confidence:", confidences.mean())
    print("Std predictive confidence:", confidences.std())

    id_confidences = compute_confidence_from_model_outputs(model, loaders["test_source_loader_1"], device)
    ood_confidences = compute_confidence_from_model_outputs(model, loaders["test_source_loader_outlier"], device)
    print(f"ID confidence:  mean={id_confidences.mean():.4f} std={id_confidences.std():.4f}")
    print(f"OOD confidence: mean={ood_confidences.mean():.4f} std={ood_confidences.std():.4f}")

    print("\n" + "=" * 70)
    print("Latent-space quality metrics (silhouette & source/target alignment)")
    print("=" * 70)
    latent_quality_report(results)

    print("\n" + "=" * 70)
    print("Final latent-space visualization (openTSNE)")
    print("=" * 70)
    opentsne_plot_all_latents_by_class(
        results,
        save_path="opentsne_latents_cnn_sngp.png",
        class_names={0: "BCC", 1: "FCC", 2: "HCP"},
        tsne_embedding=None,
        tsne_seed=1,
        perplexity=20.0,
        indices=(1, 2, 3),
    )


if __name__ == "__main__":
    main()
