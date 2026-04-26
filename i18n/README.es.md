<p align="center">
  <img src="../docs/logo_final.png" alt="RyotenkAI" width="400">
</p>
<h1 align="center">RyotenkAI</h1>

<p align="center">
  Fine-tuning declarativo para LLM.<br>
  Proporciona una configuracion YAML y tus datasets, y RyotenkAI orquesta validacion, aprovisionamiento de GPU, entrenamiento, despliegue de inferencia, evaluacion y reportes.
</p>

<p align="center">
  <a href="../README.md">🇬🇧 English</a> |
  <a href="README.ru.md">🇷🇺 Русский</a> |
  <a href="README.ja.md">🇯🇵 日本語</a> |
  <a href="README.zh-CN.md">🇨🇳 简体中文</a> |
  <a href="README.ko.md">🇰🇷 한국어</a> |
  🇪🇸 Español |
  <a href="README.he.md">🇮🇱 עברית</a>
</p>

<p align="center">
  <a href="#inicio-rapido">Inicio Rapido</a> ·
  <a href="#como-funciona">Como Funciona</a> ·
  <a href="#estrategias-de-entrenamiento">Estrategias</a> ·
  <a href="#proveedores-gpu">Proveedores</a> ·
  <a href="#sistemas-de-plugins">Plugins</a> ·
  <a href="#configuracion">Configuracion</a>
</p>

<p align="center">
  <a href="https://discord.gg/QqDM2DbY">
    <img src="https://img.shields.io/badge/Discord-Unirse%20a%20la%20comunidad-5865F2?logo=discord&logoColor=white" alt="Unirse a Discord">
  </a>
  <br>
  <img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/%F0%9F%A4%97_Transformers-4.x-FFD21E" alt="Transformers">
  <img src="https://img.shields.io/badge/TRL-multi--strategy-blueviolet" alt="TRL">
  <img src="https://img.shields.io/badge/PEFT-LoRA%20%7C%20QLoRA-8A2BE2" alt="PEFT">
  <br>
  <img src="https://img.shields.io/badge/vLLM-inference-00ADD8?logo=v&logoColor=white" alt="vLLM">
  <img src="https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white" alt="MLflow">
  <img src="https://img.shields.io/badge/Docker-containerized-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/RunPod-cloud_GPU-673AB7" alt="RunPod">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License: MIT">
</p>

---

## Que es RyotenkAI

RyotenkAI es un control plane declarativo para fine-tuning de LLM. Describes el workflow en YAML, proporcionas datasets, y RyotenkAI ejecuta todo el ciclo de vida: validacion del dataset, aprovisionamiento de GPU, entrenamiento multifase, recuperacion del modelo, despliegue de inferencia, evaluacion y reportes de experimentos en MLflow.

| Workflow manual de fine-tuning | RyotenkAI |
|---|---|
| Revisar manualmente la calidad del dataset | Validacion basada en plugins: format, duplicates, length, diversity |
| Entrar por SSH al GPU y ejecutar scripts | Un solo comando: aprovisiona GPU, despliega entrenamiento y monitorea |
| Esperar y confiar en que funcione | Monitoreo en tiempo real: metricas GPU, loss curves, deteccion de OOM |
| Descargar pesos manualmente | Recupera adapters automaticamente, hace merge de LoRA y publica en HF Hub |
| Levantar un servidor de inferencia aparte | Despliega un endpoint vLLM con health checks |
| Probar salidas a mano | Evaluacion basada en plugins: syntax, semantic match, LLM-as-judge |
| Escribir notas en un documento | Seguimiento de experimentos en MLflow + reporte Markdown generado |

---

## Como Funciona

### Flujo del pipeline

```text
YAML Config
    │
    ▼
┌─────────────────┐
│ Dataset Validator│  Valida la calidad de los datos antes del entrenamiento
│ (plugin system)  │  min_samples, diversity, format, custom
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GPU Deployer    │  Aprovisiona computo (SSH o RunPod API)
│                  │  Despliega training container + codigo
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Training Monitor  │  Sigue el proceso, analiza logs y detecta OOM
│                  │  GPU metrics, loss curves, health checks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Retriever  │  Descarga adapters / merged weights
│                  │  Opcionalmente publica en HuggingFace Hub
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Inference Deployer│  Inicia vLLM server en Docker
│                  │  Health checks, OpenAI-compatible API
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Evaluator  │  Ejecuta plugins de evaluacion sobre el endpoint activo
│ (plugin system)  │  syntax, semantic match, LLM judge, custom
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Report Generator  │  Recolecta todos los datos desde MLflow
│ (plugin system)  │  Genera un reporte de experimento en Markdown
└─────────────────┘
```

### Ejecucion del entrenamiento

```text
Pipeline (control plane)              GPU Provider (single_node / RunPod)
         │                                         │
    SSH / API ──────────────────────────► Docker container
         │                                   │
    rsync code ─────────────────────────►  /workspace/
         │                                   │
    start training ─────────────────────►  accelerate launch train.py
         │                                   │
    monitor ◄───────────── logs, markers, GPU metrics
         │                                   │
    retrieve artifacts ◄────── adapters, checkpoints, merged weights
```

### Cadena de estrategias de entrenamiento (dentro del GPU container)

Entrenamiento multifase con gestion automatica de estado, recuperacion ante OOM y checkpoints:

```text
run_training(config.yaml)
  │
  ├── MemoryManager.auto_configure()     Detecta el tier de GPU y define umbrales de VRAM
  │     └── GPUPreset: margin, critical%, max_retries
  │
  ├── load_model_and_tokenizer()         Carga el modelo base (sin PEFT todavia)
  │     └── MemoryManager: snapshot antes/despues, limpieza de CUDA cache
  │
  ├── DataBuffer.init_pipeline()         Inicializa el seguimiento de estado
  │     └── pipeline_state.json          Estados de fases y rutas de checkpoint
  │     └── phase_0_sft/                 Directorios de salida por fase
  │     └── phase_1_dpo/
  │
  └── ChainRunner.run(strategies)        Ejecuta la cadena de fases
        │
        │   Para cada fase (por ejemplo CPT → SFT → DPO):
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │  PhaseExecutor.execute(phase_idx, phase, model, buffer) │
  │                                                         │
  │  1. buffer.mark_phase_started(idx)                      │
  │     └── Guardado atomico del estado en pipeline_state.json
  │                                                         │
  │  2. StrategyFactory.create(phase.strategy_type)         │
  │     ├── SFTStrategy     (messages → instruction tuning) │
  │     ├── DPOStrategy     (chosen/rejected → alignment)   │
  │     ├── ORPOStrategy    (preference → odds ratio)       │
  │     ├── GRPOStrategy    (reward-guided RL)              │
  │     ├── SAPOStrategy    (self-aligned preference)       │
  │     └── CPTStrategy     (raw text → domain adaptation)  │
  │                                                         │
  │  3. dataset_loader.load_for_phase(phase)                │
  │     └── strategy.validate_dataset + prepare_dataset     │
  │                                                         │
  │  4. TrainerFactory.create_from_phase(...)               │
  │     ├── strategy.get_trainer_class() → TRL Trainer      │
  │     ├── Mezcla hyperparams: global ∪ phase overrides    │
  │     ├── Crea PEFT config (LoRA / QLoRA / AdaLoRA)       │
  │     ├── Conecta callbacks (MLflow, GPU metrics)         │
  │     └── Envuelto en MemoryManager.with_memory_protection│
  │                                                         │
  │  5. trainer.train()                                     │
  │     └── MemoryManager.with_memory_protection            │
  │           ├── Monitorea uso de VRAM                     │
  │           ├── Ante OOM → aggressive_cleanup + retry     │
  │           └── max_retries segun el preset del GPU tier  │
  │                                                         │
  │  6. Guarda checkpoint-final                             │
  │     ├── buffer.mark_phase_completed(metrics)            │
  │     └── buffer.cleanup_old_checkpoints(keep_last=2)     │
  │                                                         │
  └─────────────────────┬───────────────────────────────────┘
                        │
                        ▼  el modelo pasa a la siguiente fase en memoria
                        │
               ┌────────┴────────┐
               │  Siguiente fase? │
               │  idx < total    │──── No ──► devolver modelo entrenado
               └────────┬────────┘
                        │ Yes
                        ▼
                 (repetir PhaseExecutor)
```

### DataBuffer - gestion de estado entre fases

```text
DataBuffer
  │
  ├── Pipeline State (pipeline_state.json)
  │     {
  │       "status": "running",
  │       "phases": [
  │         { "strategy": "sft", "status": "completed", "checkpoint": "phase_0_sft/checkpoint-final" },
  │         { "strategy": "dpo", "status": "running",   "checkpoint": null }
  │       ]
  │     }
  │
  ├── Phase Directories
  │     output/
  │     ├── phase_0_sft/
  │     │   ├── checkpoint-500/     (intermedio, se limpia automaticamente)
  │     │   ├── checkpoint-1000/    (intermedio, se limpia automaticamente)
  │     │   └── checkpoint-final/   (se conserva, sirve de entrada para la siguiente fase)
  │     └── phase_1_dpo/
  │         └── checkpoint-final/
  │
  ├── Resume Logic
  │     En caso de fallo o reinicio:
  │       1. load_state() → encuentra la primera fase no completada
  │       2. get_model_path_for_phase(idx) → checkpoint-final anterior
  │       3. Carga adapters PEFT sobre el modelo base
  │       4. get_resume_checkpoint(idx) → checkpoint intermedio de la fase (si existe)
  │       5. Continua el entrenamiento desde donde se detuvo
  │
  └── Cleanup
         cleanup_old_checkpoints(keep_last=2)
         Elimina directorios intermedios checkpoint-N/ y conserva checkpoint-final
```

### MemoryManager - proteccion frente a OOM en GPU

```text
MemoryManager.auto_configure()
  │
  ├── Detecta GPU: nombre, VRAM, compute capability
  │     ├── RTX 4060  (8GB)  → consumer_low   tier
  │     ├── RTX 4090  (24GB) → consumer_high  tier
  │     ├── A100      (80GB) → datacenter     tier
  │     └── Unknown          → safe fallback
  │
  ├── GPUPreset por tier:
  │     margin_mb:    VRAM reservada como margen (512-4096 MB)
  │     critical_pct: umbral para activar OOM recovery (85-95%)
  │     warning_pct:  umbral para log de warning (70-85%)
  │     max_retries:  cantidad de reintentos automaticos (1-3)
  │
  └── with_memory_protection(operation):
         ┌─────────────────────────────┐
         │  Attempt 1                  │
         │  ├── Verificar margen de VRAM
         │  ├── Ejecutar operation     │
         │  └── Success → return       │
         │                             │
         │  OOM detected?              │
         │  ├── aggressive_cleanup()   │
         │  │   ├── gc.collect()       │
         │  │   ├── torch.cuda.empty_cache()
         │  │   └── Limpiar gradients  │
         │  ├── Registrar evento OOM en MLflow
         │  └── Reintentar (hasta max) │
         │                             │
         │  Se agotaron los reintentos?│
         │  └── OOMRecoverableError    │
         └─────────────────────────────┘
```

### Flujo de evaluacion

```text
EvaluationRunner
  1. Carga JSONL eval dataset → lista de (question, expected_answer, metadata)
  2. Recoge respuestas del modelo via vLLM endpoint → list[EvalSample]
  3. Para cada plugin habilitado (ordenado por prioridad):
       result = plugin.evaluate(samples)
  4. Agrega resultados → RunSummary (passed/failed, metrics, recommendations)
```

### Flujo de generacion de reportes

```text
ryotenkai runs report <run_dir>
  │
  ▼
MLflow ──► obtiene runs, metrics, artifacts, configs
  │
  ▼
Construye el modelo del reporte (phases, issues, timeline)
  │
  ▼
Ejecuta plugins (cada uno renderiza una seccion)
  │
  ▼
Renderiza Markdown → experiment_report.md
  │
  └── vuelve a registrarlo en MLflow como artifact
```

---

## Inicio Rapido

### Setup con un solo comando

```bash
git clone https://github.com/DanilGolikov/ryotenkai.git
cd ryotenkai
bash setup.sh
source .venv/bin/activate
```

### Configuracion

1. Edita `secrets.env` con tus API keys (RunPod, HuggingFace)
2. Copia y personaliza la configuracion de ejemplo

```bash
cp src/config/pipeline_config.yaml my_config.yaml
# Edita my_config.yaml con tu modelo, dataset y provider
```

### Ejecucion

```bash
# Validar la configuracion
ryotenkai config validate --config my_config.yaml

# Iniciar el pipeline completo
ryotenkai run start --config my_config.yaml

# O ejecutar el entrenamiento localmente (para desarrollo)
ryotenkai run start --local --config my_config.yaml
```

### TUI interactiva

```bash
ryotenkai tui
```

La TUI ofrece un panel navegable para revisar runs, inspeccionar estados por etapa y monitorear pipelines en vivo.

---

## Configuracion

RyotenkAI usa un unico archivo de configuracion YAML (schema v7). Secciones principales:

```yaml
model:
  name: "Qwen/Qwen2.5-0.5B-Instruct"

training:
  type: qlora                    # qlora | lora | adalora | full
  provider: single_node          # single_node | runpod
  strategies:
    - strategy_type: sft
      hyperparams: { epochs: 3 }
    - strategy_type: dpo
      hyperparams: { epochs: 1 }

datasets:
  default:
    source_hf:
      train_id: "your-org/dataset"

providers:
  single_node:
    connect:
      ssh: { alias: pc }
    training:
      workspace_path: /home/user/workspace
      docker_image: "ryotenkai/ryotenkai-training-runtime:latest"

mlflow:
  tracking_uri: "http://localhost:5002"
  experiment_name: ryotenkai
```

Referencia completa de configuracion: [`../src/config/CONFIG_REFERENCE.md`](../src/config/CONFIG_REFERENCE.md)

---

## Estrategias de entrenamiento

RyotenkAI soporta entrenamiento multifase con strategy chaining. Las strategies definen **que** entrenar; los adapters (LoRA, QLoRA, AdaLoRA, Full FT) definen **como** hacerlo.

| Estrategia | Senal | Caso de uso |
|------------|-------|-------------|
| **CPT** (Continued Pre-Training) | raw text | Inyectar conocimiento de dominio |
| **SFT** (Supervised Fine-Tuning) | instruction → response pairs | Enseniar al modelo el formato de una tarea |
| **CoT** (Chain-of-Thought) | reasoning traces | Mejorar el razonamiento paso a paso |
| **DPO** (Direct Preference Optimization) | chosen vs rejected pairs | Alinear con preferencias humanas |
| **ORPO** (Odds Ratio Preference Optimization) | chosen vs rejected pairs | Alignment sin reward model separado |
| **GRPO** (Group Relative Policy Optimization) | reward-guided RL | Reinforcement learning a partir de rewards |
| **SAPO** (Self-Aligned Preference Optimization) | chosen vs rejected + self-alignment | Mejora del aprendizaje por preferencias |

Las estrategias se pueden encadenar: `CPT → SFT → DPO` se ejecuta de forma secuencial, y cada fase construye sobre el checkpoint anterior. Toda la cadena se configura completamente en YAML.

---

## Proveedores GPU

Los proveedores gestionan el aprovisionamiento de GPU para entrenamiento e inferencia. Training e inference usan interfaces de proveedor separadas.

| Provider | Tipo | Training | Inference | Como se conecta |
|----------|------|----------|-----------|-----------------|
| **single_node** | Local | SSH a tu servidor con GPU | vLLM sobre Docker via SSH | alias en `~/.ssh/config` o host/port/key explicitos |
| **RunPod** | Nube | Pod via GraphQL API | Provision de volume + pod | API key en `secrets.env` |

### single_node

Acceso SSH directo a una maquina con GPU. El pipeline despliega un Docker container con el training runtime, sincroniza codigo, ejecuta el entrenamiento y recupera artifacts, todo via SSH. La inferencia despliega un vLLM container en el mismo host.

Caracteristicas: deteccion automatica de GPU (`nvidia-smi`), health checks, limpieza del workspace.

### RunPod

GPU en la nube via RunPod API. El pipeline crea un pod con el tipo de GPU solicitado, espera a que SSH este listo, ejecuta el entrenamiento y opcionalmente elimina el pod al finalizar. Para inferencia, aprovisiona un volume persistente y un pod separado.

Caracteristicas: spot instances, multiples tipos de GPU, auto-cleanup (`cleanup.auto_delete_pod`).

---

## Sistemas de plugins

RyotenkAI tiene tres sistemas de plugins, todos con el mismo patron: decorador `@register`, auto-discovery y secrets aislados por namespace mediante `secrets.env`.

### Validacion de datasets

Valida datasets antes de que comience el entrenamiento. Los plugins revisan format, quality, diversity y restricciones especificas del dominio. Es la primera etapa del pipeline: si la validacion falla, el entrenamiento no empieza.

Secrets namespace: `DTST_*` - Docs: [`../src/data/validation/README.md`](../src/data/validation/README.md)

### Evaluacion

Evalua la calidad del modelo despues del entrenamiento contra un endpoint vLLM activo. Los plugins ejecutan verificaciones deterministicas (syntax, semantic match) y scoring LLM-as-judge. Los resultados alimentan el reporte del experimento.

Secrets namespace: `EVAL_*` - Docs: [`../src/evaluation/plugins/README.md`](../src/evaluation/plugins/README.md)

### Generacion de reportes

Genera reportes de experimentos a partir de datos de MLflow. Cada plugin renderiza una seccion del documento Markdown (header, summary, metrics, issues, etc.). El reporte final se registra de nuevo en MLflow como artifact.

Docs: [`../src/reports/plugins/README.md`](../src/reports/plugins/README.md)

Todos los sistemas de plugins admiten plugins personalizados: implementa la clase base, usa `@register`, y el pipeline los descubrira automaticamente.

---

## Integracion con MLflow

Inicia el stack de MLflow:

```bash
make docker-mlflow-up
```

Accede a la UI en `http://localhost:5002`. Todos los pipeline runs se rastrean con metricas, artifacts y snapshots de configuracion.

---

## Imagenes Docker

| Imagen | Proposito |
|--------|-----------|
| `ryotenkai/ryotenkai-training-runtime` | CUDA + PyTorch + dependencias para entrenamiento |
| `ryotenkai/inference-vllm` | Runtime de inferencia vLLM (serve + merge deps + SSH) |

Puedes construirlas localmente o publicarlas en Docker Hub. Consulta [`../docker/training/README.md`](../docker/training/README.md) y [`../docker/inference/README.md`](../docker/inference/README.md).

---

## Referencia CLI

| Comando | Descripcion |
|---------|-------------|
| `ryotenkai run start --config <path>` | Ejecuta el training pipeline completo |
| `ryotenkai run start --local --config <path>` | Ejecuta entrenamiento local (sin GPU remota) |
| `ryotenkai dataset validate --config <path>` | Ejecuta solo la validacion del dataset |
| `ryotenkai config validate --config <path>` | Verificaciones estaticas pre-flight |
| `ryotenkai info --config <path>` | Muestra configuracion del pipeline y del modelo |
| `ryotenkai tui [run_dir]` | Lanza la TUI interactiva |
| `ryotenkai runs inspect <run_dir>` | Inspecciona un directorio de run |
| `ryotenkai runs ls [dir]` | Lista todos los runs con resumen |
| `ryotenkai runs logs <run_dir>` | Muestra el pipeline log de un run |
| `ryotenkai runs status <run_dir>` | Monitoreo en vivo de un pipeline en ejecucion |
| `ryotenkai runs diff <run_dir>` | Compara configuracion entre intentos |
| `ryotenkai runs report <run_dir>` | Genera reporte de experimento MLflow |
| `ryotenkai version` | Muestra informacion de version |

---

## Terminal UI (TUI)

RyotenkAI incluye una interfaz de terminal integrada para monitorear e inspeccionar training runs:

```bash
ryotenkai tui             # explorar todos los runs
ryotenkai tui <run_dir>   # abrir un run especifico
```

**Runs list** - vista general de todos los pipeline runs con status, duration y config name:

<p align="center">
  <img src="../docs/screenshots/tui_runs_list.png" alt="TUI Runs List" width="800">
</p>

**Run detail** - entra en cualquier run para ver stages, timing, outputs y validation results:

<p align="center">
  <img src="../docs/screenshots/tui_run_detail.png" alt="TUI Run Detail" width="800">
</p>

**Evaluation answers** - revisa las salidas del modelo junto a las respuestas esperadas:

<p align="center">
  <img src="../docs/screenshots/tui_eval_answers.png" alt="TUI Evaluation Answers" width="800">
</p>

La TUI ofrece pestanias de **Details**, **Logs**, **Inference**, **Eval** y **Report**: todo lo necesario para entender un training run sin salir de la terminal.

---

## Desarrollo

### Setup

```bash
bash setup.sh
source .venv/bin/activate
```

### Tests

```bash
make test          # todos los tests
make test-unit     # solo unit tests
make test-fast     # omite slow tests
make test-cov      # con coverage
```

### Linting

```bash
make lint          # revisar
make format        # auto-format
make fix-all       # auto-fix
```

### Pre-commit

Los hooks de pre-commit se ejecutan automaticamente. Para correrlos manualmente:

```bash
make pre-commit
```

---

## Estructura del proyecto

```text
ryotenkai/
├── src/
│   ├── config/          # Configuration schemas (Pydantic v2)
│   ├── pipeline/        # Orchestration and stage implementations
│   ├── training/        # Estrategias de entrenamiento y orchestration
│   ├── providers/       # GPU providers (single_node, RunPod)
│   ├── evaluation/      # Plugins de evaluacion del modelo
│   ├── data/            # Manejo de datasets y plugins de validacion
│   ├── reports/         # Plugins de generacion de reportes
│   ├── tui/             # Terminal UI (Textual)
│   ├── utils/           # Utilidades compartidas
│   └── tests/           # Suite de tests
├── docker/
│   ├── training/        # Training runtime Docker image
│   ├── inference/       # Inference Docker images
│   └── mlflow/          # Stack de MLflow (docker-compose)
├── scripts/             # Utility scripts
├── docs/                # Documentacion y diagramas
├── setup.sh             # Setup en un solo comando
├── Makefile             # Comandos de desarrollo
└── pyproject.toml       # Metadatos del paquete y configuracion de herramientas
```

## Comunidad

Unete al servidor de Discord para soporte, conversacion sobre roadmap, compartir configuraciones y hablar sobre workflows de fine-tuning:

[discord.gg/QqDM2DbY](https://discord.gg/QqDM2DbY)

## Contribuir

Consulta [`../CONTRIBUTING.md`](../CONTRIBUTING.md).

## Licencia

[MIT](../LICENSE) © Golikov Daniil
