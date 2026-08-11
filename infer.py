"""Standalone inference on one or more XRD images with a pretrained checkpoint.

No dataset cache, training loop, or GPU cluster required -- just a checkpoint
and an image (2D XRD detector pattern, .tif/.tiff/.png/...). Applies the same
preprocessing used at training time (outlier clipping + min-max normalization
+ resize to 256x256) and prints the predicted crystal structure
(BCC/FCC/HCP) with confidence, and, for the SNGP models, the predictive
uncertainty (std).

Usage:
    python infer.py --checkpoint saved_models/cnn_sngp_adapt.pth --arch sngp image1.tiff image2.tiff
    python infer.py --checkpoint saved_models/cnn_baseline.pth --arch plain image1.tiff
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.transform import resize

from datasets import clean_and_normalize_image
from models import CNNWithPlainHead, SNGPWithCNN

CLASS_NAMES = {0: "BCC", 1: "FCC", 2: "HCP"}
IMG_SIZE = 256


def load_image(path):
    img = np.array(Image.open(path))
    img = clean_and_normalize_image(img)
    img = resize(img, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
    return img.astype(np.float32)


def build_model(arch, input_channels=1, rff_dim=64, output_dim=3):
    if arch == "sngp":
        return SNGPWithCNN(input_channels=input_channels, rff_dim=rff_dim, output_dim=output_dim)
    if arch == "plain":
        return CNNWithPlainHead(input_channels=input_channels, output_dim=output_dim)
    raise ValueError(f"Unknown --arch {arch!r}, expected 'sngp' or 'plain'")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", help="Path(s) to XRD image file(s) (.tif/.tiff/.png/...)")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pth state_dict")
    parser.add_argument(
        "--arch",
        choices=["sngp", "plain"],
        default="sngp",
        help="'sngp' for cnn_sngp.pth / cnn_sngp_adapt.pth / continual_stage*.pth "
        "(SNGPWithCNN); 'plain' for cnn_baseline.pth (CNNWithPlainHead). Default: sngp",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    model = build_model(args.arch)
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    images = np.stack([load_image(p) for p in args.images])[:, None, :, :]  # (N, 1, 256, 256)
    x = torch.from_numpy(images).to(device)

    with torch.no_grad():
        _latent, mean_logits, std = model(x)
        probs = F.softmax(mean_logits, dim=-1)
        confidences, preds = probs.max(dim=-1)

    for path, pred, conf, s in zip(args.images, preds.tolist(), confidences.tolist(), std.squeeze(-1).tolist()):
        line = f"{path}: {CLASS_NAMES[pred]} (confidence={conf:.4f}"
        if args.arch == "sngp":
            line += f", predictive_std={s:.4f}"
        line += ")"
        print(line)


if __name__ == "__main__":
    main()
