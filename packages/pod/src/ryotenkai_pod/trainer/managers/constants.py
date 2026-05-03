"""Constants for training managers (DataBuffer, DataLoader, etc.)."""

# HuggingFace datasets API
HF_SPLIT_TRAIN = "train"

# Pipeline state JSON keys (serialization)
KEY_STATUS = "status"
KEY_STARTED_AT = "started_at"
KEY_COMPLETED_AT = "completed_at"
KEY_PHASES = "phases"

# Checkpoint naming
CHECKPOINT_FINAL_DIR = "checkpoint-final"

# Run ID generation: timestamp format length (includes milliseconds)
RUN_ID_TIMESTAMP_LEN = 20

# Checkpoint cleanup: rough size estimate per checkpoint (MB)
CHECKPOINT_SIZE_ESTIMATE_MB = 500
