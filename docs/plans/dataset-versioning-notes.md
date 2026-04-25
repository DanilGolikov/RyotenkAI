# Dataset versioning — research notes & mini-plan (v2)

> Не часть текущего PR. Отдельный этап (Phase D) с явным PR
> когда понадобится инспектировать историю / откатывать датасеты /
> сравнивать версии между запусками pipeline.
>
> **v2 update:** прежний draft (см. git history) предлагал copy-on-save в
> `.datasets/snapshots/`. Это ломается на GB-scale данных. Ниже — переработанный
> подход с content-addressable storage (CAS) в стиле DVC: hash → один blob
> на одну версию данных, дедупликация автоматически.

## Ресерч (TL;DR best practices 2024-2025)

- **«Version everything»** — индустриальный канон: Git для кода, **DVC**
  (dvc.org) для данных, **MLflow Model Registry** для моделей.
- **DVC внутри** — content-addressable storage (CAS): `dvc add foo.jsonl` →
  считается sha256 → файл копируется в `.dvc/cache/<sha[0:2]>/<sha[2:]>` →
  workspace-файл заменяется на hardlink/reflink, в репо коммитится `.dvc`
  pointer. На `dvc checkout` link восстанавливается. Один blob на
  одно содержимое — даже если 10 снапшотов идентичны, на диске лежит ровно
  один файл.
- **Reflink (CoW)** — APFS (macOS) поддерживает `clonefile()`, Btrfs/XFS —
  `FICLONE` ioctl. Это zero-cost copy: новая ссылка на те же inode-блоки,
  а write-on-modify создаёт новые блоки только для изменённых страниц.
- **Hardlink** — нулевая стоимость, но same inode → редактирование одного
  файла мутирует все ссылки. Mitigation: cache-blob — read-only (`chmod 0444`).
- **LakeFS / Pachyderm** — git-like для data lake. Промышленный
  оверкилл для одиночного локального workspace.
- **MLflow artifact logging** — per-run snapshot, но не первоклассный
  versioning.

## Решение для RyotenkAI: lightweight CAS, без DVC CLI

### Disk layout

```
<project>/
  .datasets/
    cache/                                    # content-addressable storage
      ab/cdef0123…/                           # file keyed by sha256[0:2]/[2:]
      e3/ba2c0917…/
    snapshots/
      <dataset_key>/
        history.json                          # append-only list of refs
```

`history.json` shape:
```json
{
  "version": 1,
  "dataset_key": "sft_dataset",
  "entries": [
    {
      "ts": "2026-04-25T10:30:00Z",
      "split": "train",
      "sha256": "abcdef…",
      "size_bytes": 4823945123,
      "row_count": 12_500_000,
      "message": "Initial import",
      "link_kind": "reflink"
    },
    {
      "ts": "2026-04-26T08:15:00Z",
      "split": "train",
      "sha256": "abcdef…",                 // unchanged → same blob
      "size_bytes": 4823945123,
      "row_count": 12_500_000,
      "message": "no-op snapshot",
      "link_kind": "reflink"
    },
    {
      "ts": "2026-04-26T19:42:00Z",
      "split": "train",
      "sha256": "ff8210…",                 // changed → new blob
      "size_bytes": 4824001772,
      "row_count": 12_500_212,
      "message": "Added 212 examples",
      "link_kind": "hardlink"
    }
  ]
}
```

Правила:
- Один blob лежит в cache **один раз**, независимо от количества ссылок.
- При совпадении sha256 — append-only manifest entry без новой записи на диск.
- Linking strategy (в порядке предпочтения):
  1. `reflink` — `os.copy_file_range` (Linux Btrfs/XFS) / `clonefile()` (macOS APFS)
  2. `hardlink` — `os.link()` (любая POSIX FS, same volume)
  3. `copy` — `shutil.copy2()` (cross-volume или fallback). Логируется warning'ом.
- Cache blob помечается read-only (`chmod 0444`), чтобы случайная мутация workspace через hardlink не повредила другие версии.

### Backend

**Новый модуль `src/data/versioning/cas.py`** — content-addressable store:

```python
class DatasetCAS:
    def __init__(self, project_root: Path) -> None: ...

    def stat(self, sha256: str) -> CacheEntry | None: ...

    def has(self, sha256: str) -> bool: ...

    def ingest(
        self,
        source: Path,
        progress: Callable[[int, int | None], None] | None = None,
    ) -> CacheEntry:
        """Stream-hash `source` (1 MB chunks), then either:
        - return existing entry if hash already cached;
        - link `source` into cache (reflink → hardlink → copy);
        - return CacheEntry with link_kind."""

    def restore(self, sha256: str, target: Path, link_kind: Literal["reflink","hardlink","copy"] = "reflink") -> None: ...

    def evict(self, lru_cap_bytes: int) -> int:
        """Drop cache blobs not referenced by any manifest. LRU by mtime."""
```

**Новый модуль `src/data/versioning/snapshots.py`** — manifest API:

```python
class DatasetSnapshotStore:
    def snapshot(self, dataset_key: str, source: Path, *, split: str, message: str = "") -> SnapshotEntry: ...

    def list(self, dataset_key: str) -> list[SnapshotEntry]: ...

    def diff_summary(self, dataset_key: str, sha_a: str, sha_b: str) -> DiffSummary:
        """Stream both blobs, count added/removed/changed rows. Caps row-level
        diff at 5 000 entries (UI doesn't render more anyway)."""

    def restore(self, dataset_key: str, sha256: str, target: Path) -> None: ...
```

**Endpoints (`src/api/routers/datasets.py`):**

- `POST /api/v1/projects/{id}/datasets/{key}/snapshot` body `{message?, split: "train"|"eval"}` → `SnapshotEntry`. Async streaming (Server-Sent Events for hash-progress).
- `GET /api/v1/projects/{id}/datasets/{key}/snapshots` → `{entries: [...]}`.
- `GET /api/v1/projects/{id}/datasets/{key}/snapshots/{sha}/preview?offset=N&limit=M` — paginated read of a frozen snapshot.
- `GET /api/v1/projects/{id}/datasets/{key}/diff?from={sha_a}&to={sha_b}` → `DiffSummary` (row-level for files <100 MB; head+tail+counts for larger).
- `POST /api/v1/projects/{id}/datasets/{key}/restore` body `{sha256, target_path?}` → restore с reflink, по умолчанию в `<workspace>/<key>.<sha[:8]>.jsonl` (не перезаписывает source без явного `target_path == source`).

### Frontend

- Новый sub-route `/projects/:id/datasets/:key/history` (master-detail).
- В `DatasetDetail` header → кнопка **«History»** → переход на sub-route.
- `DatasetHistoryPanel.tsx` — список snapshot entries (timestamp, sha[:8], size, rows, message). Каждая строка — Compare / Restore / Preview.
- `DatasetDiffView.tsx` — переиспользует `web/src/lib/lineDiff.ts` для row-level diff. При больших diff (>5K rows) — показывает summary («+212 rows, -12 rows, 47 modified, full diff truncated»).
- На «Snapshot now» — модалка с message-input + progress-bar (SSE).

### MLflow integration (bonus)

`src/pipeline/orchestrator.py` при старте run пишет:
```python
mlflow.set_tags({
    f"dataset.{key}.sha256": cas.stat(...)["sha256"][:16],
    f"dataset.{key}.row_count": ...,
})
```

Это связывает MLflow experiment с конкретной версией данных без зависимости от DVC.

### LRU cache management

- Лимит конфигурируется env `RYOTENKAI_DATASET_CACHE_GB` (default 20).
- При превышении: список blobs, отсортированных по `atime`, **исключая** те что в активных manifest entries → удалить старейшие до возврата под лимит.
- Trigger: один раз в час (background task) + manually via endpoint.

## Risks (3-iteration)

### Iteration 1
| # | Risk | Mitigation |
|---|---|---|
| R1 | Hash 50 GB файла занимает минуты, блокирует API | Async streaming + SSE progress; UI показывает прогресс-бар. Hash в worker thread (не event loop) |
| R2 | Hardlink через пользовательскую правку мутирует cache blob | `chmod 0444` на cache → write fail; FE-edit always tmp+rename → новый sha → новый blob |
| R3 | Cross-volume linking falls back к full copy | Detect once at init via `os.statvfs`; warn user; for >1 GB файлов запросить confirmation перед copy |

### Iteration 2
| # | Risk | Mitigation |
|---|---|---|
| R4 | Manifest race (две одновременные snapshot операции) | Atomic write через `os.replace` + advisory `fcntl.flock` на manifest |
| R5 | Source файл удалён, manifest sha остаётся | GC sweep: для всех blobs проверяет referenced-set из всех manifest'ов; orphan = unlink |
| R6 | Diff на 50 GB файлах — невозможно загрузить в память | Stream-diff с capped row-window; UI показывает summary только для файлов >100 MB |

### Iteration 3
| # | Risk | Mitigation |
|---|---|---|
| R7 | Symlink/hardlink через FUSE (NFS, sshfs) — может не работать | Detect via test-link на init; fallback к copy с warning |
| R8 | Privacy: sha-fingerprint утекает в logs / MLflow | Sha — content hash, тот же риск что у самого файла. Не share с external system без подтверждения |
| R9 | Cache-corruption (диск ошибки) | Validate on read: при checkout проверяем sha2(blob) vs key; mismatch → manifest помечает entry как corrupted, restore запрещён |

## Что выходит из MVP / Phase D
- DVC CLI integration — отложено
- LakeFS / S3 remote storage — отложено
- Cross-project dataset registry — отложено (strict per-project scoping)
- Auto-snapshot на каждый pipeline run — только по явному кликам Snapshot now (контролируемо)
- BLAKE3 (быстрее sha256, no FIPS): нет в stdlib, лишняя dep, sha256 — стандарт DVC

## Источники

- [DVC content-addressable storage explained — apxml.com](https://apxml.com/courses/data-versioning-experiment-tracking/chapter-2-versioning-data-dvc/introducing-dvc)
- [DVC checkout reference](https://doc.dvc.org/command-reference/checkout)
- [DVC cache management](https://mintlify.com/treeverse/dvc/commands/cache)
- [Content-addressable storage — Wikipedia](https://en.wikipedia.org/wiki/Content-addressable_storage)
- [reflink / clonefile / FICLONE — kernel.org](https://man7.org/linux/man-pages/man2/ioctl_ficlonerange.2.html)
