# Presets

**What they are.** Curated starter pipeline configs users can drop into their project. Each preset is a folder with a `manifest.toml` describing it and a `preset.yaml` carrying the actual config fragment.

The catalog endpoint [`GET /config/presets`](../../src/api/routers/config.py) lists every preset from this directory; the web UI `PresetDropdown` + `PresetPreviewModal` let users preview the diff and apply the preset to their current config.

## Minimal layout

```
community/presets/<preset_id>/
├── manifest.toml
└── preset.yaml           # the actual config fragment users load
```

A zip archive is also supported (`community/presets/<name>.zip`), unpacked to `community/.cache/<sha256>/` on first access.

## `manifest.toml` format (v1 — current)

```toml
[preset]
id = "02-medium"               # stable identifier; defines folder order via digit prefix
name = "≤ 10B"                 # shown in the dropdown; falls back to id
description = """
One or two sentences shown under the preset name. Tell the user WHEN to pick this —
model size, GPU class, intended use — not WHAT's inside (the diff shows that).
"""
size_tier = "medium"           # small | medium | large — drives UI chips
version = "1.0.0"

[preset.entry_point]
file = "preset.yaml"           # relative path to the YAML body
```

No other fields are read today. Anything beyond this is ignored by the loader (but will be used by the advanced contract — see below).

## `preset.yaml` conventions

- **Leave `providers: {}` empty.** Users pick their own provider in a separate Settings panel after loading the preset. Hard-coding a provider forces users to reconfigure every time.
- **Use realistic placeholders for `datasets.*`** — the current frontend replaces the user's config wholesale, so a placeholder path (`./datasets/train.jsonl`) is the least-bad default. Call this out in `description`.
- **Don't** include `evaluation` / `reports` blocks unless the preset is *specifically* about evaluation/reporting — otherwise the user's own setup gets stomped.
- Prefer small, focused YAML — a preset is a *starting point*, not a manual.

## Current apply policy (v1)

The frontend's `ConfigTab` loads a preset by replacing the entire form value with the parsed YAML (`setFormValue(parsed)`). This is blunt: **every top-level key present in `preset.yaml` overwrites the user's config, and keys absent from the preset are removed.**

Practical consequences you must document in your preset:

- If `datasets:` is in `preset.yaml`, the user's existing `datasets` is replaced.
- If `providers: {}` is in `preset.yaml`, the user's providers block is wiped.
- The `PresetPreviewModal` shows a field-level diff against the current form so users can confirm before clicking Apply.

A v2 contract with explicit `[preset.scope]` / `[preset.requirements]` / `[preset.placeholders]` is planned to replace this blunt overwrite with a declared scope + requirements check; see **Advanced manifest fields** below.

## Referencing a preset from the UI / API

```
GET /api/v1/config/presets
→ [{ name, display_name, description, yaml, size_tier }, …]
```

The frontend renders the list as a dropdown. On click it opens `PresetPreviewModal` which computes `deepDiff(currentForm, parsedPresetYaml)` and asks the user to confirm. Applying writes the parsed YAML into the form and marks it dirty.

There is no CLI import step — dropping the folder into `community/presets/` is all that's required. On the next API request the loader picks it up.

## Testing a preset locally

1. Put `community/presets/<id>/manifest.toml` + `preset.yaml` in place.
2. Restart the backend (or call `catalog.reload()` in a shell).
3. Hit `GET /api/v1/config/presets` — the preset should show up with correct `display_name` / `description` / `size_tier`.
4. In the UI, click the preset, review the diff, apply, then run `POST /api/v1/config/validate` on the saved config to make sure the preset produces a valid pipeline.

## Advanced manifest fields — coming in a separate task

The planned v2 contract extends the manifest with three optional blocks:

```toml
[preset.scope]                         # what the preset owns vs leaves alone
replaces  = ["model", "training"]
preserves = ["datasets", "providers", "evaluation", "reports"]

[preset.requirements]                  # what the user's environment must provide
hub_models         = ["meta-llama/Llama-3.1-70B-Instruct"]
provider_kind      = ["runpod"]
required_plugins   = []
min_vram_gb        = 80

[preset.placeholders]                  # fields the user must fill after applying
"datasets.default.source_local.local_paths.train" = "Path to your training JSONL"
```

**None of these are read yet.** They will be wired into a new `POST /config/presets/{id}/preview` endpoint that returns a structured diff, a requirements check, and placeholder hints — at which point the frontend can render the three-section modal ("Changes / Preserved / Requirements / Things you must fill"). Authoring these fields early costs nothing; they're ignored by v1.

## Sharing

Drop the folder into `community/presets/` or ship a zip. The loader handles both.

## Examples in this directory

| Folder | `size_tier` | Target hardware |
|---|---|---|
| `01-small/` | `small` | Consumer GPUs (models up to ~1 B params) |
| `02-medium/` | `medium` | Single A100/A40 (models in the 2–10 B sweet spot) |
| `03-large/` | `large` | Multi-GPU / big-memory card (13 B+) |
