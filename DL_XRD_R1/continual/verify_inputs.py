"""Verify all preserved continual inputs and the plan without model training."""
from collections import Counter
import json
import sys

sys.dont_write_bytecode = True
import data_setup as data
import launch_suite


def main():
    scopes = data.load_scopes(data.DEFAULT_ASSIGNMENTS, data.DEFAULT_CACHE)
    rows = data.read_assignments(data.DEFAULT_ASSIGNMENTS)
    data.validate_assignments(data.material_disjoint_assignments(rows))
    assert len(scopes) == 24
    assert len(rows) == len({r['sample_id'] for r in rows}) == 1701
    assert len({r['exact_float32_image_sha256'] for r in rows}) == 1701
    counts = Counter((r['domain'], r['split']) for r in rows)
    assert [counts['source', split] for split in data.SPLITS] == [723, 89, 89]
    assert [counts['target', split] for split in data.SPLITS] == [642, 79, 79]
    for order in data.ORDERS.values():
        for stage in range(1, 5):
            for method in launch_suite.METHODS:
                _, _, training_ids, validation_ids, forbidden = data.stage_loaders(
                    scopes, order, stage, method)
                assert not set(training_ids) & set(validation_ids)
                assert not (set(training_ids) | set(validation_ids)) & set(forbidden)
    print(json.dumps(dict(status='passed', scopes=len(scopes), images=len(rows),
        image_fingerprints_verified=True, anchor_count=data.EXPECTED_ANCHOR_COUNT,
        requested_target_anchor_fraction=data.REQUESTED_TARGET_ANCHOR_FRACTION,
        realized_target_anchor_fraction=data.EXPECTED_ANCHOR_COUNT / counts['target', 'train'],
        target_validation_labels=79,
        jobs_planned=len(launch_suite.jobs()), stage_scope_checks=72,
        source_counts={s:counts['source',s] for s in data.SPLITS},
        target_counts={s:counts['target',s] for s in data.SPLITS},
        training_started=False, result_files_created=False, checkpoints_created=False), indent=2))


if __name__ == '__main__':
    main()
