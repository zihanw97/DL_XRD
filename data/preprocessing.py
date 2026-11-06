"""Image preprocessing used by the manuscript's original CNN architecture.

Native TIFF dimensions are preserved until the final direct resize. No crop,
log transform, geometric registration, azimuthal integration or augmentation
is applied.
"""
from pathlib import Path
import csv
import hashlib

import numpy as np
from PIL import Image
from skimage.transform import resize


def clean_and_normalize_image(image, quantile=0.999):
    """Upper-clip at the per-image quantile, then min-max scale in float64."""
    image=np.asarray(image).astype(float)
    if image.ndim!=2:
        raise ValueError('Expected one 2D detector-count image')
    cutoff=np.quantile(image,quantile)
    clipped=np.clip(image,a_min=None,a_max=cutoff)
    # A few simulated TIFFs contain +inf peaks; the original upper clip
    # intentionally removes those when the 99.9th percentile is finite.
    if not np.isfinite(clipped).all():
        raise ValueError('Detector image remains nonfinite after upper clipping')
    return (clipped-clipped.min())/(clipped.max()-clipped.min()+1e-8)


def preprocess_image(image):
    """Return a single-channel float32 image with shape [1, 256, 256]."""
    normalized=clean_and_normalize_image(image,quantile=0.999)
    resized=resize(normalized,(256,256),anti_aliasing=True)
    return np.asarray(resized,dtype=np.float32)[None,:,:]


def load_image(path):
    """Read one TIFF with PIL and apply the exact fixed-model preprocessing."""
    with Image.open(Path(path)) as handle:
        original=np.array(handle)
    return preprocess_image(original)


def rebuild_fixed_cache(output_path):
    """Rebuild a new fixed-input NPZ from raw TIFFs, checking all 1,701 hashes.

    The immutable role/anchor inputs are not altered. An existing output is
    never overwritten. A different library version can change interpolation
    arithmetic; a hash mismatch is reported before any NPZ is written.
    """
    data=Path(__file__).resolve().parent
    root=data.parent
    output=Path(output_path).resolve()
    if output.exists():
        raise FileExistsError('Choose a new input-cache output path: '+str(output))
    with (data/'metadata/sample_inventory.csv').open(newline='') as stream:
        rows=sorted((r for r in csv.DictReader(stream) if r['domain'] in ['source','target']),
                    key=lambda r:r['sample_id'])
    with (data/'partitions/continual_partition.csv').open(newline='') as stream:
        expected={r['sample_id']:r['exact_float32_image_sha256'] for r in csv.DictReader(stream)}
    if len(rows)!=1701 or len(expected)!=1701:
        raise ValueError('Expected exactly the registered known-class inventory')
    images=[]
    for row in rows:
        image=load_image(root/row['sample_id'])
        if hashlib.sha256(image.tobytes()).hexdigest()!=expected[row['sample_id']]:
            raise ValueError('Preprocessing differs from the registered pixels: '+row['sample_id'])
        images.append(image)
    phases={'BCC':0,'FCC':1,'HCP':2}
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(output,images=np.stack(images),
             labels=np.array([phases[r['phase']] for r in rows],dtype=np.int64),
             sample_ids=np.array([r['sample_id'] for r in rows]),
             domains=np.array([r['domain'] for r in rows]),
             materials=np.array([r['material'] for r in rows]))
    return output


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild-fixed-cache',type=Path,required=True,
                        help='New NPZ output path; rebuild only input images/metadata from raw TIFFs')
    args=parser.parse_args()
    print(rebuild_fixed_cache(args.rebuild_fixed_cache))
