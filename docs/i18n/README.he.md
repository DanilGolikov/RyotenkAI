<p align="center">
  <img src="../logo_final.png" alt="RyotenkAI" width="400">
</p>
<h1 align="center">RyotenkAI</h1>

<p align="center">
  fine-tuning הצהרתי עבור LLM.<br>
  מספקים קובץ YAML ודאטהסטים, ו-RyotenkAI מתזמר אימות, הקצאת GPU, אימון, פריסת inference, הערכה והפקת דוחות.
</p>

<p align="center">
  <a href="../../README.md">🇬🇧 English</a> |
  <a href="README.ru.md">🇷🇺 Русский</a> |
  <a href="README.ja.md">🇯🇵 日本語</a> |
  <a href="README.zh-CN.md">🇨🇳 简体中文</a> |
  <a href="README.ko.md">🇰🇷 한국어</a> |
  <a href="README.es.md">🇪🇸 Español</a> |
  🇮🇱 עברית
</p>

<p align="center">
  <a href="#התחלה-מהירה">התחלה מהירה</a> ·
  <a href="#איך-זה-עובד">איך זה עובד</a> ·
  <a href="#אסטרטגיות-אימון">אסטרטגיות</a> ·
  <a href="#ספקי-gpu">ספקים</a> ·
  <a href="#מערכות-פלאגינים">פלאגינים</a> ·
  <a href="#תצורה">תצורה</a>
</p>

<p align="center">
  <a href="https://discord.gg/QqDM2DbY">
    <img src="https://img.shields.io/badge/Discord-הצטרפו%20לקהילה-5865F2?logo=discord&logoColor=white" alt="הצטרפו ל-Discord">
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

## מה זה RyotenkAI

RyotenkAI הוא control plane הצהרתי עבור fine-tuning של LLM. אתם מתארים את ה-workflow ב-YAML, מספקים דאטהסטים, ו-RyotenkAI מבצע את מחזור החיים המלא: אימות דאטהסט, הקצאת GPU, אימון רב-שלבי, שליפת מודל, פריסת inference, הערכה ודיווח ניסויים ב-MLflow.

| workflow ידני של fine-tuning | RyotenkAI |
|---|---|
| בדיקה ידנית של איכות הדאטהסט | אימות מבוסס plugins: format, duplicates, length, diversity |
| התחברות ב-SSH ל-GPU והרצת סקריפטים | פקודה אחת: מקצה GPU, פורסת אימון ומנטרת |
| להמתין ולקוות שהכל יעבוד | ניטור בזמן אמת: GPU metrics, loss curves, זיהוי OOM |
| הורדה ידנית של משקלים | שליפה אוטומטית של adapters, מיזוג LoRA ופרסום ל-HF Hub |
| הקמה נפרדת של inference server | פריסה של vLLM endpoint עם health checks |
| בדיקה ידנית של הפלט | הערכה מבוססת plugins: syntax, semantic match, LLM-as-judge |
| כתיבת הערות במסמך | מעקב ניסויים ב-MLflow + דוח Markdown שנוצר אוטומטית |

---

## איך זה עובד

### זרימת ה-pipeline

```text
YAML Config
    │
    ▼
┌─────────────────┐
│ Dataset Validator│  מאמת איכות נתונים לפני האימון
│ (plugin system)  │  min_samples, diversity, format, custom
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GPU Deployer    │  מקצה משאבי חישוב (SSH או RunPod API)
│                  │  פורס training container + קוד
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Training Monitor  │  עוקב אחרי התהליך, מנתח לוגים ומזהה OOM
│                  │  GPU metrics, loss curves, health checks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Retriever  │  מוריד adapters / merged weights
│                  │  אופציונלית מפרסם ל-HuggingFace Hub
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Inference Deployer│  מפעיל vLLM server בתוך Docker
│                  │  Health checks, OpenAI-compatible API
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Evaluator  │  מריץ plugins של הערכה על endpoint פעיל
│ (plugin system)  │  syntax, semantic match, LLM judge, custom
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Report Generator  │  אוסף את כל הנתונים מ-MLflow
│ (plugin system)  │  מרנדר דוח ניסוי ב-Markdown
└─────────────────┘
```

### ביצוע האימון

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

### שרשרת אסטרטגיות האימון (בתוך GPU container)

אימון רב-שלבי עם ניהול מצב אוטומטי, התאוששות מ-OOM ו-checkpointing:

```text
run_training(config.yaml)
  │
  ├── MemoryManager.auto_configure()     מזהה את ה-GPU tier ומגדיר ספי VRAM
  │     └── GPUPreset: margin, critical%, max_retries
  │
  ├── load_model_and_tokenizer()         טוען את המודל הבסיסי (עדיין בלי PEFT)
  │     └── MemoryManager: snapshot לפני/אחרי, ניקוי CUDA cache
  │
  ├── DataBuffer.init_pipeline()         מאתחל מעקב מצב
  │     └── pipeline_state.json          מצבי phases ונתיבי checkpoint
  │     └── phase_0_sft/                 ספריות פלט לכל phase
  │     └── phase_1_dpo/
  │
  └── ChainRunner.run(strategies)        מריץ את שרשרת השלבים
        │
        │   עבור כל phase (למשל CPT → SFT → DPO):
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │  PhaseExecutor.execute(phase_idx, phase, model, buffer) │
  │                                                         │
  │  1. buffer.mark_phase_started(idx)                      │
  │     └── שמירת מצב אטומית אל pipeline_state.json        │
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
  │     ├── מיזוג hyperparams: global ∪ phase overrides     │
  │     ├── יצירת PEFT config (LoRA / QLoRA / AdaLoRA)      │
  │     ├── חיבור callbacks (MLflow, GPU metrics)           │
  │     └── עטיפה ב-MemoryManager.with_memory_protection    │
  │                                                         │
  │  5. trainer.train()                                     │
  │     └── MemoryManager.with_memory_protection            │
  │           ├── ניטור שימוש ב-VRAM                       │
  │           ├── בעת OOM → aggressive_cleanup + retry      │
  │           └── max_retries לפי GPU tier preset           │
  │                                                         │
  │  6. שמירת checkpoint-final                              │
  │     ├── buffer.mark_phase_completed(metrics)            │
  │     └── buffer.cleanup_old_checkpoints(keep_last=2)     │
  │                                                         │
  └─────────────────────┬───────────────────────────────────┘
                        │
                        ▼  המודל עובר ל-phase הבא בזיכרון
                        │
               ┌────────┴────────┐
               │  phase הבא?      │
               │  idx < total    │──── No ──► החזרת מודל מאומן
               └────────┬────────┘
                        │ Yes
                        ▼
                 (חזרה על PhaseExecutor)
```

### DataBuffer - ניהול מצב בין phases

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
  │     │   ├── checkpoint-500/     (checkpoint ביניים, מנוקה אוטומטית)
  │     │   ├── checkpoint-1000/    (checkpoint ביניים, מנוקה אוטומטית)
  │     │   └── checkpoint-final/   (נשמר ומשמש כקלט ל-phase הבא)
  │     └── phase_1_dpo/
  │         └── checkpoint-final/
  │
  ├── Resume Logic
  │     בעת crash/restart:
  │       1. load_state() → איתור ה-phase הראשון שלא הושלם
  │       2. get_model_path_for_phase(idx) → checkpoint-final הקודם
  │       3. טעינת PEFT adapters על המודל הבסיסי
  │       4. get_resume_checkpoint(idx) → checkpoint אמצעי של ה-phase (אם יש)
  │       5. המשך אימון מהנקודה שבה הופסק
  │
  └── Cleanup
         cleanup_old_checkpoints(keep_last=2)
         מסיר ספריות checkpoint-N/ ביניים ושומר על checkpoint-final
```

### MemoryManager - הגנה מפני GPU OOM

```text
MemoryManager.auto_configure()
  │
  ├── זיהוי GPU: שם, VRAM, compute capability
  │     ├── RTX 4060  (8GB)  → consumer_low   tier
  │     ├── RTX 4090  (24GB) → consumer_high  tier
  │     ├── A100      (80GB) → datacenter     tier
  │     └── Unknown          → safe fallback
  │
  ├── GPUPreset לכל tier:
  │     margin_mb:    מרווח VRAM שמור (512-4096 MB)
  │     critical_pct: סף להפעלת OOM recovery (85-95%)
  │     warning_pct:  סף ל-warning logs (70-85%)
  │     max_retries:  מספר ניסיונות חוזרים אוטומטיים (1-3)
  │
  └── with_memory_protection(operation):
         ┌─────────────────────────────┐
         │  Attempt 1                  │
         │  ├── בדיקת מרווח VRAM       │
         │  ├── הרצת operation         │
         │  └── Success → return       │
         │                             │
         │  OOM detected?              │
         │  ├── aggressive_cleanup()   │
         │  │   ├── gc.collect()       │
         │  │   ├── torch.cuda.empty_cache()
         │  │   └── ניקוי gradients    │
         │  ├── רישום אירוע OOM ב-MLflow
         │  └── Retry (עד max)         │
         │                             │
         │  כל הניסיונות נכשלו?        │
         │  └── OOMRecoverableError    │
         └─────────────────────────────┘
```

### זרימת הערכה

```text
EvaluationRunner
  1. טעינת JSONL eval dataset → רשימת (question, expected_answer, metadata)
  2. איסוף תשובות מודל דרך vLLM endpoint → list[EvalSample]
  3. עבור כל plugin מופעל (ממויין לפי עדיפות):
       result = plugin.evaluate(samples)
  4. אגרגציית תוצאות → RunSummary (passed/failed, metrics, recommendations)
```

### זרימת יצירת דוחות

```text
ryotenkai runs report <run_dir>
  │
  ▼
MLflow ──► שליפת runs, metrics, artifacts, configs
  │
  ▼
בניית מודל דוח (phases, issues, timeline)
  │
  ▼
הרצת plugins (כל plugin מרנדר מקטע אחד)
  │
  ▼
רינדור Markdown → experiment_report.md
  │
  └── נשמר שוב ב-MLflow כ-artifact
```

---

## התחלה מהירה

### setup בפקודה אחת

```bash
git clone https://github.com/DanilGolikov/ryotenkai.git
cd ryotenkai
bash setup.sh
source .venv/bin/activate
```

### תצורה

1. ערכו את `secrets.env` והוסיפו API keys (RunPod, HuggingFace)
2. העתיקו והתאימו את קובץ התצורה לדוגמה

```bash
cp src/config/pipeline_config.yaml my_config.yaml
# ערכו את my_config.yaml עם המודל, הדאטהסט והגדרות ה-provider שלכם
```

### הרצה

```bash
# אימות התצורה
ryotenkai config validate --config my_config.yaml

# הרצת ה-pipeline המלא
ryotenkai run start --config my_config.yaml

# או הרצת אימון מקומי (לצרכי פיתוח)
ryotenkai run start --local --config my_config.yaml
```

### TUI אינטראקטיבי

```bash
ryotenkai tui
```

ה-TUI מספק dashboard ניווטי לעיון ב-runs, בדיקת מצבי stages וניטור pipelines חיים.

---

## תצורה

RyotenkAI משתמש בקובץ YAML יחיד (schema v7). הסעיפים המרכזיים:

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

מסמך תצורה מלא: [`../src/config/CONFIG_REFERENCE.md`](../../src/config/CONFIG_REFERENCE.md)

---

## אסטרטגיות אימון

RyotenkAI תומך באימון רב-שלבי באמצעות strategy chaining. ה-strategies מגדירות **מה** לאמן; ה-adapters (LoRA, QLoRA, AdaLoRA, Full FT) מגדירים **איך** לאמן.

| אסטרטגיה | סיגנל | שימוש |
|----------|-------|-------|
| **CPT** (Continued Pre-Training) | raw text | הזרקת ידע תחומי |
| **SFT** (Supervised Fine-Tuning) | instruction → response pairs | ללמד את המודל את פורמט המשימה |
| **CoT** (Chain-of-Thought) | reasoning traces | שיפור reasoning צעד-אחר-צעד |
| **DPO** (Direct Preference Optimization) | chosen vs rejected pairs | alignment להעדפות אנושיות |
| **ORPO** (Odds Ratio Preference Optimization) | chosen vs rejected pairs | alignment בלי reward model נפרד |
| **GRPO** (Group Relative Policy Optimization) | reward-guided RL | reinforcement learning מבוסס reward |
| **SAPO** (Self-Aligned Preference Optimization) | chosen vs rejected + self-alignment | שיפור preference learning |

אפשר לשרשר אסטרטגיות: `CPT → SFT → DPO` רץ ברצף, כאשר כל phase נבנה על ה-checkpoint של השלב הקודם. כל השרשרת ניתנת להגדרה מלאה ב-YAML.

---

## ספקי GPU

הספקים אחראים על הקצאת GPU לאימון ול-inference. גם ל-training וגם ל-inference יש provider interfaces נפרדים.

| Provider | סוג | Training | Inference | אופן חיבור |
|----------|-----|----------|-----------|------------|
| **single_node** | מקומי | SSH לשרת ה-GPU שלכם | vLLM ב-Docker דרך SSH | alias ב-`~/.ssh/config` או host/port/key מפורשים |
| **RunPod** | ענן | Pod דרך GraphQL API | הקצאת volume + pod | API key ב-`secrets.env` |

### single_node

גישה ישירה ב-SSH למכונה עם GPU. ה-pipeline פורס Docker container עם training runtime, מסנכרן קוד, מריץ אימון ומחזיר artifacts, הכל דרך SSH. ה-inference פורס vLLM container על אותו host.

יכולות: זיהוי GPU אוטומטי (`nvidia-smi`), health checks, ניקוי workspace.

### RunPod

GPU בענן דרך RunPod API. ה-pipeline יוצר pod עם סוג ה-GPU המבוקש, ממתין ל-SSH readiness, מריץ אימון, ובמידת הצורך מוחק את ה-pod בסיום. עבור inference הוא מקצה persistent volume ו-pod נפרד.

יכולות: spot instances, סוגי GPU מרובים, auto-cleanup (`cleanup.auto_delete_pod`).

---

## מערכות פלאגינים

ל-RyotenkAI יש שלוש מערכות plugins, וכולן פועלות באותו דפוס: decorator מסוג `@register`, גילוי אוטומטי, ו-secrets מבודדים לפי namespace דרך `secrets.env`.

### אימות דאטהסטים

מאמת דאטהסטים לפני תחילת האימון. ה-plugins בודקים format, quality, diversity ומגבלות תחומיות. זהו השלב הראשון ב-pipeline, ואם האימות נכשל, האימון לא יתחיל.

Secrets namespace: `DTST_*` - Docs: [`../src/data/validation/README.md`](../src/data/validation/README.md)

### הערכה

מעריך את איכות המודל לאחר האימון מול live vLLM endpoint. ה-plugins מריצים בדיקות דטרמיניסטיות (syntax, semantic match) ו-LLM-as-judge scoring. התוצאות מוזנות לדוח הניסוי.

Secrets namespace: `EVAL_*` - Docs: [`../src/evaluation/plugins/README.md`](../src/evaluation/plugins/README.md)

### הפקת דוחות

מייצר דוחות ניסוי מתוך נתוני MLflow. כל plugin מרנדר מקטע אחד במסמך Markdown (header, summary, metrics, issues ועוד). הדוח הסופי נרשם בחזרה ל-MLflow כ-artifact.

Docs: [`../src/reports/plugins/README.md`](../src/reports/plugins/README.md)

כל מערכות ה-plugins תומכות גם ב-plugins מותאמים אישית: מממשים את מחלקת הבסיס, מוסיפים `@register`, וה-pipeline יגלה אותם אוטומטית.

---

## אינטגרציה עם MLflow

הפעילו את סטאק MLflow:

```bash
make docker-mlflow-up
```

ממשק ה-UI זמין ב-`http://localhost:5002`. כל ה-pipeline runs מנוטרים יחד עם metrics, artifacts ו-config snapshots.

---

## Docker Images

| Image | מטרה |
|-------|------|
| `ryotenkai/ryotenkai-training-runtime` | CUDA + PyTorch + dependencies לצורכי אימון |
| `ryotenkai/inference-vllm` | vLLM inference runtime (serve + merge deps + SSH) |

אפשר לבנות מקומית או לפרסם ל-Docker Hub. ראו [`../docker/training/README.md`](../../docker/training/README.md) ו-[`../docker/inference/README.md`](../../docker/inference/README.md).

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `ryotenkai run start --config <path>` | מריץ את training pipeline המלא |
| `ryotenkai run start --local --config <path>` | מריץ אימון מקומי (ללא GPU מרוחק) |
| `ryotenkai dataset validate --config <path>` | מריץ רק אימות דאטהסט |
| `ryotenkai config validate --config <path>` | בדיקות pre-flight סטטיות לתצורה |
| `ryotenkai info --config <path>` | מציג תצורת pipeline ומודל |
| `ryotenkai tui [run_dir]` | מפעיל TUI אינטראקטיבי |
| `ryotenkai runs inspect <run_dir>` | בודק ספריית run |
| `ryotenkai runs ls [dir]` | מציג את כל ה-runs עם סיכום |
| `ryotenkai runs logs <run_dir>` | מציג pipeline log של run מסוים |
| `ryotenkai runs status <run_dir>` | ניטור חי של pipeline רץ |
| `ryotenkai runs diff <run_dir>` | השוואת config בין ניסיונות |
| `ryotenkai runs report <run_dir>` | יצירת דוח ניסוי ב-MLflow |
| `ryotenkai version` | הצגת מידע על גרסה |

---

## Terminal UI (TUI)

RyotenkAI כולל ממשק טרמינל מובנה לניטור ולבדיקה של training runs:

```bash
ryotenkai tui             # עיון בכל ה-runs
ryotenkai tui <run_dir>   # פתיחת run מסוים
```

**Runs list** - סקירה של כל pipeline runs עם status, duration ו-config name:

<p align="center">
  <img src="../docs/screenshots/tui_runs_list.png" alt="TUI Runs List" width="800">
</p>

**Run detail** - כניסה לכל run כדי לראות stages, timing, outputs ו-validation results:

<p align="center">
  <img src="../docs/screenshots/tui_run_detail.png" alt="TUI Run Detail" width="800">
</p>

**Evaluation answers** - סקירת פלטי המודל לצד expected answers:

<p align="center">
  <img src="../docs/screenshots/tui_eval_answers.png" alt="TUI Evaluation Answers" width="800">
</p>

ה-TUI מספק לשוניות **Details**, **Logs**, **Inference**, **Eval** ו-**Report** - כל מה שצריך כדי להבין training run מבלי לעזוב את הטרמינל.

---

## פיתוח

### Setup

```bash
bash setup.sh
source .venv/bin/activate
```

### Tests

```bash
make test          # כל הבדיקות
make test-unit     # unit tests בלבד
make test-fast     # דילוג על slow tests
make test-cov      # עם coverage
```

### Linting

```bash
make lint          # בדיקה
make format        # עיצוב אוטומטי
make fix-all       # תיקון אוטומטי
```

### Pre-commit

Pre-commit hooks רצים אוטומטית. להרצה ידנית:

```bash
make pre-commit
```

---

## מבנה הפרויקט

```text
ryotenkai/
├── src/
│   ├── config/          # Configuration schemas (Pydantic v2)
│   ├── pipeline/        # Orchestration and stage implementations
│   ├── training/        # אסטרטגיות אימון ו-orchestration
│   ├── providers/       # GPU providers (single_node, RunPod)
│   ├── evaluation/      # plugins להערכת מודל
│   ├── data/            # טיפול בדאטהסטים ו-plugins לאימות
│   ├── reports/         # plugins להפקת דוחות
│   ├── tui/             # Terminal UI (Textual)
│   ├── utils/           # utilities משותפים
│   └── tests/           # חבילת הבדיקות
├── docker/
│   ├── training/        # Training runtime Docker image
│   ├── inference/       # Inference Docker images
│   └── mlflow/          # MLflow stack (docker-compose)
├── scripts/             # Utility scripts
├── docs/                # תיעוד ודיאגרמות
├── setup.sh             # setup בפקודה אחת
├── Makefile             # פקודות פיתוח
└── pyproject.toml       # מטא-דאטה של החבילה והגדרות כלים
```

## קהילה

הצטרפו לשרת ה-Discord כדי לקבל תמיכה, לדון ב-roadmap, לשתף configs ולדבר על workflows של fine-tuning:

[discord.gg/QqDM2DbY](https://discord.gg/QqDM2DbY)

## תרומה

ראו [`../CONTRIBUTING.md`](../../CONTRIBUTING.md).

## רישיון

[MIT](../../LICENSE) © Golikov Daniil
