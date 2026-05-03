"""
Offline integration test for EvaluationRunner.

Loads the REAL eval dataset, mocks model answers (no RunPod / no vLLM),
runs helixql_syntax + cerebras_judge plugins, saves answers.md + reports.

Output goes to runs/test_eval_runner_offline/ for visual inspection.

Run:
    python3 -m pytest src/tests/integration/test_eval_runner_offline.py -v -s
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

EVAL_DATASET = Path(__file__).resolve().parents[3] / "data" / "eval" / "helixql_eval.jsonl"

pytestmark = pytest.mark.skipif(
    not EVAL_DATASET.exists(),
    reason=f"Eval dataset not found: {EVAL_DATASET}",
)


# ---------------------------------------------------------------------------
# Mock inference client
# ---------------------------------------------------------------------------

class _MockInferenceClient:
    """
    Returns a mix of answers to exercise all paths:
      bucket 0 (every 3rd)  → correct HelixQL (expected answer)
      bucket 1              → helixschema block (what the real model did)
      bucket 2              → empty string (garbage)
    """

    def __init__(self, dataset_rows: list[dict]) -> None:
        self._rows = dataset_rows
        self._idx = 0

    def generate(self, prompt: str) -> str:
        row = self._rows[self._idx] if self._idx < len(self._rows) else {}
        bucket = self._idx % 3
        self._idx += 1

        if bucket == 0:
            return row.get("expected_answer", "")

        if bucket == 1:
            question = row.get("question", "")
            if "```helixschema" in question:
                start = question.index("```helixschema")
                end = question.index("```", start + 3) + 3
                return question[start:end]
            return "```helixschema\nUnknown {}\n```"

        return ""  # bucket 2 — empty/garbage


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def _load_rows() -> list[dict]:
    rows = []
    with EVAL_DATASET.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_eval_config(*, cerebras_enabled: bool = False):
    from src.config.evaluation.schema import (
        EvaluationConfig,
        EvaluationDatasetConfig,
        EvaluatorPluginConfig,
        EvaluatorsConfig,
    )

    return EvaluationConfig(
        enabled=True,
        dataset=EvaluationDatasetConfig(path=str(EVAL_DATASET)),
        save_answers_md=True,
        evaluators=EvaluatorsConfig(plugins=[
            EvaluatorPluginConfig(
                id="syntax_check",
                plugin="helixql_syntax",
                enabled=True,
                save_report=True,
                params={},
                thresholds={"min_valid_ratio": 0.5},
            ),
            EvaluatorPluginConfig(
                id="judge_main",
                plugin="cerebras_judge",
                enabled=cerebras_enabled,
                save_report=True,
                params={
                    "model": "gpt-oss-120b",
                    "max_samples": 5,
                    "temperature": 0.0,
                    "max_tokens": 512,
                    "max_retries": 1,
                },
                thresholds={"min_mean_score": 0.6},
            ),
        ]),
    )


INSPECT_DIR = Path("runs/test_eval_runner_offline")


def _run_with_dir(out_dir: Path):
    """Run EvaluationRunner with mock client, writing output to out_dir."""
    import sys

    import src.utils.logger as logger_mod_real
    logger_mod = sys.modules.get("src.utils.logger", logger_mod_real)

    original = getattr(logger_mod, "_run_log_dir", None)
    logger_mod._run_log_dir = out_dir
    try:
        from src.evaluation.runner import EvaluationRunner
        rows = _load_rows()
        client = _MockInferenceClient(rows)
        cfg = _build_eval_config(cerebras_enabled=False)
        runner = EvaluationRunner(eval_config=cfg, secrets_resolver=None)
        return runner.run(client), out_dir
    finally:
        logger_mod._run_log_dir = original


# ---------------------------------------------------------------------------
# Fixture: run once, share across tests in class
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class")
def run_result(tmp_path_factory, monkeypatch_session=None):
    """Run EvaluationRunner once per class with mock client."""
    tmp_path = tmp_path_factory.mktemp("eval_offline")
    return tmp_path


@pytest.fixture()
def summary_and_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import sys

    from src.evaluation.runner import EvaluationRunner

    # _run_log_dir is a module-level variable in src.utils.logger
    logger_mod = sys.modules.get("src.utils.logger")
    if logger_mod is None:
        import src.utils.logger as logger_mod
    monkeypatch.setattr(logger_mod, "_run_log_dir", tmp_path)

    rows = _load_rows()
    client = _MockInferenceClient(rows)
    cfg = _build_eval_config(cerebras_enabled=False)

    runner = EvaluationRunner(eval_config=cfg, secrets_resolver=None)
    summary = runner.run(client)
    return summary, tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAnswersMd:
    def test_file_created(self, summary_and_dir) -> None:
        _, out = summary_and_dir
        assert (out / "evaluation" / "answers.md").exists()

    def test_has_all_task_sections(self, summary_and_dir) -> None:
        _, out = summary_and_dir
        content = (out / "evaluation" / "answers.md").read_text()
        rows = _load_rows()
        for i in range(1, len(rows) + 1):
            assert f"### Task {i}" in content

    def test_raw_helixschema_preserved(self, summary_and_dir) -> None:
        """Bucket-1 answers are raw helixschema blocks — must appear verbatim."""
        _, out = summary_and_dir
        content = (out / "evaluation" / "answers.md").read_text()
        assert "```helixschema" in content

    def test_no_strip_on_expected_answer(self, summary_and_dir) -> None:
        """Expected answers from dataset must appear verbatim (no strip/manipulation)."""
        _, out = summary_and_dir
        content = (out / "evaluation" / "answers.md").read_text()
        assert "RETURN items" in content

    def test_print_first_3_tasks(self, summary_and_dir, capsys) -> None:
        _, out = summary_and_dir
        text = (out / "evaluation" / "answers.md").read_text()
        # Print first 3 tasks for visual check
        parts = text.split("\n---\n")
        print("\n\n" + "="*60 + "\nanswers.md (first 3 tasks)\n" + "="*60)
        print("\n---\n".join(parts[:3]))


class TestHelixqlSyntaxReport:
    def test_report_file_created(self, summary_and_dir) -> None:
        _, out = summary_and_dir
        assert (out / "evaluation" / "helixql_syntax_report.md").exists()

    def test_no_markdown_table(self, summary_and_dir) -> None:
        _, out = summary_and_dir
        content = (out / "evaluation" / "helixql_syntax_report.md").read_text()
        # Table rows have 3+ pipe-separated columns — metrics line uses | as separator but has no columns
        table_lines = [
            ln for ln in content.splitlines()
            if ln.startswith("|") or (ln.count("|") >= 3 and not ln.startswith("**"))
        ]
        assert table_lines == [], f"Found table rows: {table_lines[:2]}"

    def test_has_task_sections(self, summary_and_dir) -> None:
        _, out = summary_and_dir
        content = (out / "evaluation" / "helixql_syntax_report.md").read_text()
        assert "### Task 1" in content
        assert "### Model answer" in content
        assert "### Expected answer" in content

    def test_valid_and_invalid_both_present(self, summary_and_dir) -> None:
        """~33% answers are correct → report must contain both passing and failing verdicts."""
        _, out = summary_and_dir
        content = (out / "evaluation" / "helixql_syntax_report.md").read_text()
        assert "**Score:** OK" in content
        assert "**Score:** FAILED" in content

    def test_invalid_has_reason(self, summary_and_dir) -> None:
        _, out = summary_and_dir
        content = (out / "evaluation" / "helixql_syntax_report.md").read_text()
        assert "Comment" in content

    def test_valid_ratio_is_partial(self, summary_and_dir) -> None:
        summary, _ = summary_and_dir
        result = summary.plugin_results["syntax_check"]
        ratio = result.metrics["valid_ratio"]
        assert 0.0 < ratio < 1.0, f"Expected partial ratio, got {ratio}"
        print(f"\n[helixql_syntax] valid_ratio={ratio:.2%}")

    def test_print_report(self, summary_and_dir, capsys) -> None:
        _, out = summary_and_dir
        text = (out / "evaluation" / "helixql_syntax_report.md").read_text()
        parts = text.split("\n---\n\n")
        print("\n\n" + "="*60 + "\nhelixql_syntax_report.md (header + first 3 tasks)\n" + "="*60)
        print("\n---\n\n".join(parts[:4]))


class TestSampleCount:
    def test_all_samples_collected(self, summary_and_dir) -> None:
        summary, _ = summary_and_dir
        rows = _load_rows()
        assert summary.sample_count == len(rows), (
            f"Expected {len(rows)} samples, got {summary.sample_count}"
        )


# ---------------------------------------------------------------------------
# Cerebras live (skipped without API key)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[3] / "secrets.env").exists(),
    reason="secrets.env not found",
)
class TestCerebrasJudgeLive:
    """
    Runs cerebras_judge against 5 real samples with mock model answers.
    Skipped automatically if EVAL_CEREBRAS_API_KEY is missing from secrets.env.
    """

    def _has_cerebras_key(self) -> bool:
        secrets = (Path(__file__).resolve().parents[3] / "secrets.env").read_text()
        return "EVAL_CEREBRAS_API_KEY" in secrets and "EVAL_CEREBRAS_API_KEY=" in secrets

    def test_cerebras_report_created(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        if not self._has_cerebras_key():
            pytest.skip("EVAL_CEREBRAS_API_KEY not in secrets.env")

        import sys

        from src.config.secrets import load_secrets
        from src.evaluation.plugins.secrets import SecretsResolver
        from src.evaluation.runner import EvaluationRunner

        logger_mod = sys.modules.get("src.utils.logger")
        if logger_mod is None:
            import src.utils.logger as logger_mod
        monkeypatch.setattr(logger_mod, "_run_log_dir", tmp_path)

        secrets = load_secrets()
        resolver = SecretsResolver(secrets)

        rows = _load_rows()[:5]  # only 5 samples
        client = _MockInferenceClient(rows)
        cfg = _build_eval_config(cerebras_enabled=True)
        # Cap max_samples at 5
        next(p for p in cfg.evaluators.plugins if p.id == "judge_main").params["max_samples"] = 5

        runner = EvaluationRunner(eval_config=cfg, secrets_resolver=resolver)
        summary = runner.run(client)

        report = tmp_path / "evaluation" / "cerebras_judge_report.md"
        assert report.exists(), "cerebras_judge_report.md not created"
        content = report.read_text()
        assert "# cerebras_judge Report" in content
        print("\n\n" + "="*60 + "\ncerebras_judge_report.md\n" + "="*60)
        print(content)


# ---------------------------------------------------------------------------
# Visual inspect: writes to runs/test_eval_runner_offline/ (persistent)
# ---------------------------------------------------------------------------

def test_generate_inspect_files() -> None:
    """
    Generates answers.md + helixql_syntax_report.md + cerebras_judge_report.md
    into runs/test_eval_runner_offline/evaluation/ for direct inspection in IDE.

    cerebras_judge runs if EVAL_CEREBRAS_API_KEY is present in secrets.env,
    otherwise only syntax plugin runs.
    """
    import shutil
    import sys

    import src.utils.logger as logger_mod_real

    secrets_path = Path(__file__).resolve().parents[3] / "secrets.env"
    has_cerebras = (
        secrets_path.exists()
        and "EVAL_CEREBRAS_API_KEY=" in secrets_path.read_text()
    )

    out_dir = INSPECT_DIR
    eval_dir = out_dir / "evaluation"
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    logger_mod = sys.modules.get("src.utils.logger", logger_mod_real)
    original = getattr(logger_mod, "_run_log_dir", None)
    logger_mod._run_log_dir = out_dir

    try:
        from src.evaluation.runner import EvaluationRunner

        rows = _load_rows()
        cfg = _build_eval_config(cerebras_enabled=has_cerebras)

        if has_cerebras:
            from src.config.secrets import load_secrets
            from src.evaluation.plugins.secrets import SecretsResolver
            secrets = load_secrets()
            resolver = SecretsResolver(secrets)
            # Cap cerebras at 10 samples to save quota
            next(p for p in cfg.evaluators.plugins if p.id == "judge_main").params["max_samples"] = 10
        else:
            resolver = None

        client = _MockInferenceClient(rows)
        runner = EvaluationRunner(eval_config=cfg, secrets_resolver=resolver)
        runner.run(client)
    finally:
        logger_mod._run_log_dir = original

    print(f"\n✓ Files written under {eval_dir.resolve()}/")
    for f in sorted(eval_dir.iterdir()):
        print(f"  - {f.name}  ({f.stat().st_size} bytes)")
    print("\nOpen in IDE:")
    for f in sorted(eval_dir.iterdir()):
        print(f"  {f.resolve()}")
