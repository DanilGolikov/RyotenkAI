"""Mac-side data layer (preview / validation).

Dataset *loading* lives in :mod:`ryotenkai_pod.trainer.data_loaders` —
the loader runs on the pod next to the trainer. This package only
hosts Mac-side concerns: dataset previews used by the Web UI and the
validation framework that runs before the pipeline kicks off.
"""
