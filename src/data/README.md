# Data Module

Dataset loading and validation for LLM training. Supports local JSONL files and Hugging Face Hub with native TRL formatting.

## Structure

```
src/data/
├── __init__.py
├── constants.py
├── loaders/
│   ├── base.py                  # BaseDatasetLoader (base class)
│   ├── json_loader.py           # JsonDatasetLoader — local JSON/JSONL files
│   ├── hf_loader.py             # HuggingFaceDatasetLoader — HF Hub datasets
│   ├── multi_source_loader.py   # MultiSourceDatasetLoader — per-phase routing
│   └── factory.py               # DatasetLoaderFactory — auto-select by source_type
└── validation/                  # Plugin-based dataset validation (see validation/README.md)
```

## Dataset Loaders

### DatasetLoaderFactory (recommended)

```python
from src.data.loaders import DatasetLoaderFactory

factory = DatasetLoaderFactory(config)
loader = factory.create_for_dataset(dataset_config)
dataset = loader.load(source)
```

### JsonDatasetLoader — local files

```python
from src.data.loaders import JsonDatasetLoader

loader = JsonDatasetLoader(config)
result = loader.load_for_phase(phase_config)
dataset = result.unwrap()

dataset = loader.load("data/train.jsonl")
```

### HuggingFaceDatasetLoader — HF Hub

```python
from src.data.loaders import HuggingFaceDatasetLoader

loader = HuggingFaceDatasetLoader(config)
dataset = loader.load("tatsu-lab/alpaca", split="train")
```

### MultiSourceDatasetLoader — multi-phase training

Routes loading to the right loader per phase based on `source_type` in config.

```python
from src.data.loaders import MultiSourceDatasetLoader

loader = MultiSourceDatasetLoader(config)
result = loader.load_for_phase(phase_config)
dataset = result.unwrap()
```

## Supported Formats

TRL SFTTrainer natively supports:

### Messages (ChatML) — for SFT/CoT

```json
{
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"}
  ]
}
```

### Text — for CPT (continual pre-training)

```json
{"text": "Raw text for language modeling..."}
```

### DPO Preference — for DPO/ORPO

```json
{
  "prompt": "Write a poem",
  "chosen": "Good response here",
  "rejected": "Bad response here"
}
```

### GRPO/SAPO — for online RL

```json
{
  "prompt": "Solve: 2+2=",
  "completion": "4"
}
```

## Validation

Plugin-based dataset validation runs before training via the `DatasetValidator` pipeline stage.

See [validation/README.md](./validation/README.md) for the full plugin system docs, including custom plugin creation, secrets (`DTST_*`), and configuration.

## Related Files

| File | Role |
|------|------|
| `src/pipeline/stages/dataset_validator.py` | Validation orchestrator |
| `src/utils/config.py` | `DatasetConfig`, `PipelineConfig` |
| `src/config/CONFIG_REFERENCE.md` | Full configuration documentation |
