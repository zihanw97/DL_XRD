# Input data

Raw TIFFs and the two NPZ caches are distributed as assets of the
`manuscript-clean-inputs-20261106` release. From the `DL_XRD_R1` package root, run
`python data/download_inputs.py` to restore them at the paths below and verify
their checksums. These release assets are the data for this R1 package;
they are stored in GitHub Releases rather than duplicated as Git blobs.
Input manifests, acquisition metadata, checksums, and split
assignments are versioned with the code. The older `data-v1` release and its
checkpoints are a separate, historical artifact set.

This folder contains only model inputs, immutable sample metadata and partition definitions. No trained weights, optimizer states, predictions, learned embeddings or previous result summaries are included.

The 1,901 TIFF inputs comprise 901 simulated, 800 known-class experimental, 100 noise-only and 100 SiC images. Simulated images are 1024 by 1024; experimental images are 1062 by 1028. Original acquisition instructions are included for the 20 experimental exposure folders. The available Al2O3 and unidentified OOD groups were not used in the requested comparison and are excluded. Duplicate copies of the noise/SiC images are not added.

`cache/xrd_dataset_cache.npz` is the original image/label data cache converted from a tensor serialization without changing any array value. Its 0-D `batch_size` array is an input setting. It contains no network or optimizer state. Some legacy combined-data masks remain for exact cache preservation; neither the fixed nor continual pipeline uses those old masks. `cache/fixed_images.npz` contains the same 1,701 known-class images in stable sample-ID order, plus labels, sample IDs, domains and materials. Use `allow_pickle=False` when reading either file.

`partitions/continual_partition.csv` is the unchanged legacy source partition with 127 of 642 target-training labels. The continual loader verifies its hash and applies a requested 10% training-label mask in memory: 64 of 642 labels (9.97% after rounding), selected as a deterministic subset of those original anchors. Counts use round-half-up per canonical block/class, with SHA256 ranking and seed 242 as specified in `continual/README.md` and `continual/recipe_provenance.json`. Every strategy, seed and order shares the same effective mask; registered run-profile CSVs contain it explicitly, and material-disjoint remapping retains the same anchor IDs. The 79 target-validation labels remain separate. These roles must not be replaced by the fixed-model masks.

`manifests/fixed/scan.csv` and `grouped.csv` are NEW paper-matched fixed-model partitions shared by CNN, CNN-SNGP and CNN-SNGP-adapt. They use requested 10% experimental training anchors. Within each eligible domain/class, floor(10%) is assigned to validation, floor(10%) to internal testing and the rest to training. Training anchors are sampled only from target-training rows, with classwise round-half-up counts (at least one per nonempty class). Nontraining `training_label` fields are -1; `label` is reserved for evaluation truth. No geometry- or test-result-based partition selection is used.

The scan protocol excludes simulated Ag/Ti/V and experimental Cu as separate outer tests. The grouped protocol holds each of Al/Cu/Fe/Mo/Nb/Ni/Ti/Zn out in both domains, then applies the within-development split. Three repeats use initialization seeds 42/43/44, partition seeds 142/143/144 and anchor seeds 242/243/244. Every model receives the same manifest row identities. Exact counts are in `manifests/fixed/partition_counts.csv`. These new fixed manifests are not the historical 30%-anchor/8:1-development experiments from the prior manuscript results.

`manifests/ood.csv` identifies all 200 confirmed noise/SiC scans for inference after model selection is locked. These inputs have no BCC/FCC/HCP reference label and are excluded from optimization, validation and GP precision updates.

`metadata/sample_inventory.csv` and `acquisition_metadata.csv` report the verified measurement hierarchy. Each known material has one owner-confirmed specimen, shared between its 1 s and 3 s exposures; this is eight known specimen groups, not 800 independent specimens. Scan positions come from acquisition instructions; coordinate units and independent session boundaries are unverified. SiC polytype/crystallinity are not inferred from its two exposure folders.

`cache/cache_manifest.json` records unchanged source-input hashes, shapes and legacy partition policies; `continual/recipe_provenance.json` records the active 10% continual training mask. `metadata/raw_file_hashes.csv` provides SHA256 for every included TIFF. Files are independent copies, never references to the development workspace.

From the package root, run `python data/verify_inputs.py --full` to verify every raw TIFF and all source cache/partition identities without constructing a model. Run `python continual/verify_inputs.py` to verify the effective 64-label continual training mask and stage scopes. Run `python data/prepare_fixed_manifests.py --check` to regenerate the fixed partitions in memory and compare all rows. An explicit `--output-dir PATH` writes regenerated CSVs to a chosen directory; the default check does not change inputs.

The shared `data.preprocessing.load_image` function implements the original preprocessing. `python -m data.preprocessing --rebuild-fixed-cache NEW_PATH.npz` rebuilds a new fixed-image input cache directly from the included TIFFs and verifies every image against the immutable float32 fingerprints before writing. It never overwrites an existing output. The supplied cache was preserved from the original input tensors; 52 representative TIFFs across all material identities and both domains independently reproduced those cached float32 arrays bitwise with this preprocessing implementation. A few raw simulated TIFFs contain positive infinite peaks; the original percentile upper clipping removes these before resizing.
