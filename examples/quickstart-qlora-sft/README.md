# Quickstart: QLoRA SFT on Qwen2.5-0.5B

Train a small language model using QLoRA (4-bit) on RunPod in under 10 minutes.

This example fine-tunes [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
on a HelixQL NL→Query task using 400 training samples, then deploys and evaluates the result.

**Estimated cost**: ~$0.05 on RunPod Community Cloud (NVIDIA RTX A4000, ~$0.22/hr).

---

## Prerequisites

| Requirement | Where to get it |
|---|---|
| Python 3.12+ | [python.org](https://www.python.org/downloads/) |
| RunPod account | [runpod.io](https://www.runpod.io/) |
| RunPod API key | [Settings → API Keys](https://www.runpod.io/console/user/settings) |
| SSH key registered in RunPod | [Settings → SSH Keys](https://www.runpod.io/console/user/settings) |
| HuggingFace token (read access) | [Settings → Tokens](https://huggingface.co/settings/tokens) |

## Step 1: Install RyotenkAI

```bash
git clone https://github.com/ryotenkai/ryotenkai.git
cd ryotenkai
./setup.sh
```

## Step 2: Configure secrets

```bash
cp .env.example secrets.env
```

Edit `secrets.env` and fill in:

```env
RUNPOD_API_KEY=your_runpod_api_key
HF_TOKEN=your_huggingface_read_token
```

## Step 3: Review the config

Open `examples/quickstart-qlora-sft/pipeline_config.yaml` and search for `TODO` comments.
At minimum you need to:

1. Set `providers.runpod.connect.ssh.key_path` to your SSH private key path
2. Verify Docker image tags are up to date (check [Docker Hub](https://hub.docker.com/u/ryotenkai))

## Step 4: Validate config

```bash
ryotenkai config-validate --config examples/quickstart-qlora-sft/pipeline_config.yaml
```

## Step 5: Run the pipeline

```bash
python -m src.pipeline.orchestrator examples/quickstart-qlora-sft/pipeline_config.yaml
```

The pipeline will:

1. **Validate** the training dataset (min samples, diversity, empty ratio)
2. **Provision** an A4000 GPU pod on RunPod
3. **Train** Qwen2.5-0.5B with QLoRA for 3 epochs (~5 min)
4. **Retrieve** the trained LoRA adapter
5. **Deploy** a vLLM inference server with the merged model
6. **Evaluate** syntax validity on a held-out eval set
7. **Clean up** the pod automatically

## What's in the box

```
examples/quickstart-qlora-sft/
├── pipeline_config.yaml              # Pipeline config (edit TODOs)
├── datasets/
│   ├── training/
│   │   ├── train.jsonl               # 400 SFT samples (NL → HelixQL)
│   │   └── eval.jsonl                # 49 eval samples (training split)
│   └── evaluation/
│       └── helixql_eval.jsonl        # 31 held-out samples (post-training eval)
└── README.md                         # This file
```

### Dataset format

Each line in the JSONL files is a chat-format sample:

```json
{
  "messages": [
    {"role": "system", "content": "You are a HelixQL generator..."},
    {"role": "user", "content": "Schema + natural language request"},
    {"role": "assistant", "content": "QUERY GetUsers(...) => ..."}
  ]
}
```

## Optional: MLflow tracking

To track experiments with MLflow, start the local stack first:

```bash
./docker/mlflow/start.sh
```

Then set `experiment_tracking.mlflow.enabled: true` in the config.
MLflow UI will be available at http://localhost:5002.

## Optional: LLM-as-a-judge evaluation

To enable the Cerebras LLM judge plugin:

1. Get an API key from [Cerebras](https://cloud.cerebras.ai/)
2. Add `EVAL_CEREBRAS_API_KEY=your_key` to `secrets.env`
3. Set `evaluators.plugins[judge_main].enabled: true` in the config

## Troubleshooting

| Problem | Solution |
|---|---|
| `RUNPOD_API_KEY not set` | Check that `secrets.env` exists and contains the key |
| SSH connection timeout | Verify your SSH key is registered in RunPod settings |
| OOM during training | Reduce `per_device_train_batch_size` to 4 |
| Config validation fails | Run `ryotenkai config-validate --config ...` for details |
