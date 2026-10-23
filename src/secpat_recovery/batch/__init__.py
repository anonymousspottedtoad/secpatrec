"""Batch runs: recover every record of a manifest, score a batch folder.

``recover_batch.run`` is ``secpat recover --manifest-path …`` and
``evaluate_batch.run`` is ``secpat evaluate --manifest-path … --batch-path …``;
``secpat ablation run`` runs both once per arm. Scoring reads the ground truth
from a local snapshot only. See ``README.md`` beside this file for the stages.
"""
