# Testing Strategy Research: How Hard-to-Test Systems Test Themselves (2024-2026)

Research input for RyotenkAI testing-strategy proposal. Five reference systems
analyzed: Kubernetes, Unreal Engine, Grafana, Argo Workflows, and Terraform/Pulumi
providers (the cloud-provider plugin pair best mirrors RunPod-as-a-provider).
Each lists concrete techniques with the underlying principle. Final section
maps the patterns onto an ML pipeline orchestrator (orchestrator + RunPod +
trainer subprocess + vLLM + React UI).

---

## 1. Kubernetes (kubelet / scheduler / controller-manager)

The canonical "complex distributed system" testing playbook. K8s ships
~14 different categories of tests, gated separately.

1. **Two-tier suite split: `test/e2e` vs `test/e2e_node`** — `test/e2e` runs
   as a generic client against any cluster (used for conformance);
   `test/e2e_node` runs *on the same node as a kubelet instance* so it can
   poke kubelet directly. Principle: collocate the test with the component
   when the component is the SUT, otherwise treat the cluster as a black box.
   ([k8s e2e framework][1], [sig-testing e2e-tests][2])

2. **Ginkgo v2 + Gomega with `Eventually`/`Consistently`** — distributed-system
   assertions are time-windowed: `Eventually(condition).Should(BeTrue())` polls
   until satisfied; `Consistently` checks it stays true. `framework.ExpectNoError`
   wraps errors with stack traces and step names from `ginkgo.By`. Principle:
   never assert on point-in-time state in async systems; assert on convergence.
   ([e2e-tests devel guide][2], [ginkgo with-in-k8s][1])

3. **Conformance labels + focus/skip filters** — every test is labelled
   `[Conformance]`, `[Slow]`, `[Serial]`, `[Disruptive]`, `[Feature:X]`,
   `[sig-name]`. CI runs different label subsets in different jobs (presubmit
   blocking vs nightly). Principle: a *single* test corpus, sliced by metadata
   into different latency/reliability budgets. ([e2e-tests labels][2])

4. **kind / kubernetes-in-docker as the substrate** — every "node" is a
   Docker container running systemd+kubelet, bootstrapped with kubeadm.
   Spins up a multi-node cluster in <60s. Used for both presubmit e2e
   (`pull-kubernetes-e2e-kind`) and Cluster-API E2E (`NewKindClusterProvider`).
   Principle: the real binary, fake hardware — get production code paths
   without paying for VMs. ([kind project][3], [Cluster API e2e][4])

5. **ClusterLoader2 (CL2)** — declarative scalability harness. YAML
   describes "I want 10k pods, 2k services, 5 daemonsets, this throughput" and
   CL2 hits SLOs for pod-startup latency, API latency, etc. Release-blocking.
   Principle: scale is a first-class test target with explicit SLO contracts,
   not a "we'll see how it goes". ([clusterloader2 README][5],
   [load-test config][6])

6. **Chaos-Mesh / LitmusChaos / kube-monkey** — CRD-defined experiments
   (`PodChaos`, `NetworkChaos`, `IOChaos`, `StressChaos`) injected via eBPF
   and sidecar; namespace whitelist + RBAC controls blast radius.
   `kube-monkey/mtbf` = mean-time-between-failure label per app. Probes
   (HTTP/Prometheus/k8s) validate steady-state during chaos.
   Principle: failure injection lives *next to* the code, not in a separate
   QA repo, so resilience is a design property, not a phase. ([Chaos Mesh][7],
   [Litmus probes][8], [kube-monkey][9])

7. **Prow + TestGrid + go.k8s.io/triage** — every PR runs hundreds of jobs
   across many clusters; TestGrid renders pass/fail history as a grid;
   `triage` aggregates failures by stack-trace fingerprint across the fleet.
   Persistent flakes are quarantined by adding `[Flaky]` to the test name —
   they still run, but only on a flake-only board. Merge-blocking flakes are
   P0 with quick turnaround required. Principle: separate the *test result*
   from the *test verdict*; flakes get tagged, not deleted, so signal is
   preserved while CI stays green. ([flaky-tests guide][10])

8. **`-poll-progress-after=30s` + SIGUSR1 progress reports** — long e2e tests
   periodically dump a progress report; you can SIGINFO the running test
   to get a "where am I stuck" trace. Principle: long tests must be
   *observable while running*, not just on failure. ([e2e best practices][2])

[1]: https://www.kubernetes.dev/blog/2023/04/12/e2e-testing-best-practices-reloaded/
[2]: https://github.com/kubernetes/community/blob/master/contributors/devel/sig-testing/e2e-tests.md
[3]: https://kind.sigs.k8s.io/
[4]: https://main.cluster-api.sigs.k8s.io/developer/core/testing
[5]: https://github.com/kubernetes/perf-tests/tree/master/clusterloader2
[6]: https://github.com/kubernetes/perf-tests/blob/master/clusterloader2/testing/load/config.yaml
[7]: https://chaos-mesh.org/
[8]: https://litmuschaos.io/
[9]: https://github.com/asobti/kube-monkey
[10]: https://github.com/kubernetes/community/blob/master/contributors/devel/sig-testing/flaky-tests.md

---

## 2. Unreal Engine (and the wider AAA-game testing landscape)

Games face the same problem ML pipelines do: an enormous combinatorial state
space, expensive runs, non-determinism, and "did it look right" is part of
correctness.

1. **Automation Test Framework (the base layer)** — C++ macro library
   (`IMPLEMENT_SIMPLE_AUTOMATION_TEST`), with `ADD_LATENT_AUTOMATION_COMMAND`
   for multi-frame async tests. Every higher-level framework wraps this.
   Principle: a single low-level test bus, multiple ergonomic facades.
   ([UE 5.7 Automation framework][11], [Andrew Fray 2025 topography][12])

2. **Functional Tests + Spec + CQTest** — three styles on top of Automation:
   blueprint-authored functional tests, BDD-style Spec with BeforeEach/AfterEach,
   and CQTest (Lyra) for fluent gameplay tests. Principle: pick the abstraction
   that matches the code's authoring layer (BP designers vs C++ engineers).
   ([Andrew Fray][12])

3. **Gauntlet** — orchestrates *Unreal sessions* (e.g. "5 clients + 1 server
   on 3 platforms") across devices. Not a test framework; a *session runner*
   that boots binaries, parses logs, scrapes /Saved, manages ports. Uses
   `TestController` puppeteers on the game side. Principle: the orchestration
   layer is its own thing, separate from assertions — same idea as
   Argo-running-tests-as-workflows. ([UE 5.7 Gauntlet][13])

4. **Screenshot Comparison Tool** — built into the Automation Framework:
   capture screenshot at a known camera, diff against checked-in golden,
   flag visual deltas. Principle: when the *output is pixels*, the assertion
   is also pixels. ([UE Screenshot Tool][14])

5. **Determinism via FixedFrameStep + Substepping** — force `DeltaTime`
   so physics, animation, and AI run identically across runs. Required for
   replay-based regression tests. Principle: regression suites need
   determinism *engineered in*, not hoped for. ([Duality.ai determinism][15])

6. **CSV Profiler + `stat unit` regression CI** — `csvprofile start/stop`
   captures Game/Render/GPU ms-per-frame to CSV; `CsvToSvg` renders trends.
   A "fixed camera flythrough" gives a deterministic but dynamic scene
   you can re-run nightly and trend frame time per resolution. Principle:
   *performance is a regression*, gated like correctness, with explicit
   budgets in milliseconds (not FPS). ([Unreal Art Opt][16],
   [Puget benchmark methodology][17])

7. **AI bot / monkey-tester agents (Modl.ai, Regression Games)** — LLM-driven
   exploration agents that wander the game, capturing logs/screenshots/perf
   on every anomaly. Modl.ai works engine-agnostic; Regression Games is
   Unity-only. Principle: where the state space is too large to enumerate,
   *generate* test trajectories with an agent and use anomaly detection as
   the oracle. ([Modl.ai][18], [Regression Games][19])

8. **Replay/record-replay regression** — record a player session, replay
   deterministically against new builds, diff outcomes. Project Malmo and
   GAGI demonstrate the pattern; in shipping titles it's typical to record
   golden replays from QA and re-run in CI. ([academic survey][20])

[11]: https://dev.epicgames.com/documentation/en-us/unreal-engine/automation-test-framework-in-unreal-engine
[12]: https://andrewfray.wordpress.com/2025/04/09/the-topography-of-unreal-test-automation-in-2025/
[13]: https://dev.epicgames.com/documentation/en-us/unreal-engine/gauntlet-automation-framework-in-unreal-engine
[14]: https://dev.epicgames.com/documentation/en-us/unreal-engine/screenshot-comparison-tool-in-unreal-engine
[15]: https://www.duality.ai/blog/game-engines-determinism
[16]: https://unrealartoptimization.github.io/book/process/measuring-performance/
[17]: https://www.pugetsystems.com/blog/2023/05/19/unreal-engine-testing-methodolgies/
[18]: https://modl.ai/
[19]: https://www.regression.gg/
[20]: https://arxiv.org/pdf/1906.00317

---

## 3. Grafana (the panel/data-source combinatorial explosion)

Grafana's problem: arbitrary user-defined dashboards × dozens of datasources
× plugin SDK × frontend rendering. Solved with a layered SDK-contract test
pyramid.

1. **Backend: Go + testify, split into `make test-go-unit` /
   `make test-go-integration`** — unit tests use `-short` and a 30-min
   timeout; integration tests have per-DB variants
   (`test-go-integration-postgres`). Test infra in `pkg/tests/testsuite` and
   `pkg/tests/testinfra`. Principle: classic pyramid — unit fast, integration
   gated by docker-compose backends. ([Grafana plugin SDK Go][21])

2. **Plugin SDK contract tests** — every backend plugin implements a fixed
   set of interfaces (`QueryDataHandler`, `CheckHealthHandler`, `StreamHandler`,
   `InstanceDisposer`, `CallResourceHandler`). Tests construct a real
   `backend.QueryDataRequest`/`DataSourceInstanceSettings`, call
   `datasource.QueryData(ctx, req)`, assert on the response. The SDK *is*
   the test surface. Principle: lock the contract, let plugins evolve
   freely behind it. ([Grafana plugin SDK][21], [SDK go pkg][22])

3. **Cypress for plugin E2E** — `@grafana/plugin-e2e` ships a Cypress lib
   that boots a local Grafana, installs the plugin, and exercises real UI
   flows. Plugins drop a `cypress.config.ts` + spec files. Principle: ship
   the harness *as a library* so every plugin tests the same way.
   ([Grafana e2e cypress forum][23])

4. **Storybook + Chromatic** — Grafana publishes a Storybook for design-system
   components. Chromatic auto-converts every story to a visual regression
   test; PRs are badged with a "UI Tests" check that must pass before merge.
   Principle: every component variation already exists as a story — get
   visual regression for free by pointing a tool at the existing artifact.
   ([Storybook visual tests][24], [Chromatic for Storybook][25])

5. **k6 browser tests for cross-environment performance** — Grafana Labs
   uses k6 (their own product) to run browser-based E2E + perf tests against
   Grafana Cloud and report into Grafana itself. Principle: dogfood and
   collapse "E2E test" + "synthetic monitor" into one artifact.
   ([Grafana k6 browser tips 2024][26])

6. **CI shard parallelism** — backend Go suite is sharded across multiple
   CI jobs to keep wall-clock low. Principle: parallel by *test* not by
   *file*; long tests don't dominate one shard.

[21]: https://oneuptime.com/blog/post/2026-01-30-grafana-backend-plugins/view
[22]: https://pkg.go.dev/github.com/grafana/grafana-plugin-sdk-go/backend/datasource
[23]: https://community.grafana.com/t/e2e-plugin-testing-via-cypress/111374
[24]: https://storybook.js.org/docs/writing-tests/visual-testing
[25]: https://www.chromatic.com/storybook
[26]: https://grafana.com/blog/2024/11/21/5-tips-to-write-better-browser-tests-for-performance-testing-and-synthetic-monitoring/

---

## 4. Argo Workflows (DAG semantics + retry/error distinction + K8s controllers)

Closest analogue for "long-running, retry-heavy, controller-driven pipeline
on opaque infrastructure".

1. **`test/e2e` Go suite gated by build tags** — `make test-api`,
   `make test-cli`, `make test-functional`, etc. Build tags
   (`//go:build api`) on each file map to a Make target so CI runs subsets in
   parallel; locally you `make TestArtifactServer` for one. Principle: tag
   tests by *infrastructure dependency*, not by name, so CI parallelism is
   a property of metadata. ([Argo running locally][27], [Makefile][28])

2. **`make start PROFILE=mysql AUTH_MODE=client API=true`** — the dev loop
   runs the *real* controller + server against a local k3d/kind cluster,
   tests submit real workflow CRs via kubectl, then introspect with
   `kubectl get wf` and controller logs. Principle: e2e is "run prod, look
   at it" — the more the test resembles a user, the more it catches.
   ([running locally][27])

3. **Failure-vs-Error retry semantics encoded as test fixtures** — Argo
   distinguishes *Failure* (main container exits non-zero) from *Error*
   (init/wait/controller bug). Each `retryStrategy.retryPolicy`
   (OnFailure / OnError / Always / OnTransientError) has dedicated e2e
   tests in `test/e2e/expectedfailures/parallelism-dag-failure.yaml`.
   Principle: when retry semantics are part of the contract, every branch
   needs a deliberate fixture. ([Argo retries docs][29],
   [Failure vs Error discussion][30])

4. **`expr`-based retry with `lastRetry.exitCode` / `lastRetry.duration` /
   `lastRetry.message`** — retry is configurable as an expression evaluated
   per attempt, AND-ed with retryPolicy. Tests must cover the *expression
   evaluator*, not just the retry counter. Principle: when policy becomes
   a tiny DSL, that DSL needs its own unit tests. ([retries doc][29])

5. **`hack/db` — fake-data injectors for local repro** — a Go CLI under
   `hack/` generates archived workflows in the dev DB so you can repro state
   that's expensive to reach naturally (e.g. retry storms). Principle:
   bypass the slow path to specific states with seeded fixtures.

6. **k3d image-import inner-loop** — Makefile detects k3d, builds `argosay`
   test images locally, imports via `k3d image import`. Avoids registry
   round-trip. Principle: the test image must be *the* image you're
   testing, with zero registry latency. ([Makefile][28])

7. **Stress testing as kubectl apply** — `kubectl apply -f
   test/stress/massive-workflow.yaml` exercises the controller under load.
   Principle: load tests live next to functional tests, in the same repo,
   in the same yaml dialect.

[27]: https://argo-workflows.readthedocs.io/en/latest/running-locally/
[28]: https://github.com/argoproj/argo-workflows/blob/main/Makefile
[29]: https://argo-workflows.readthedocs.io/en/latest/retries/
[30]: https://github.com/argoproj/argo-workflows/discussions/10429

---

## 5. Terraform & Pulumi providers (the cloud-provider plugin = RunPod analogue)

This is the closest-fit reference for RyotenkAI's RunPod provider: an
external API, opaque from the user's side, costly to hit, with eventual
consistency and quotas.

1. **`terraform-plugin-testing` framework with `TestStep` lifecycle** —
   acceptance tests run real `plan`/`apply`/`refresh`/`destroy` cycles via a
   plugin server attached in-process to `go test`. Principle: the test runner
   *becomes* the orchestrator — same code path, no double-implementation.
   ([Terraform plugin testing][31])

2. **go-vcr cassette replay (Google + Okta providers)** — provider tests
   record HTTP interactions to YAML cassettes on first run, replay
   deterministically afterwards. Custom `Configure` overrides swap the HTTP
   client at test init. Costs drop ~100x; flakes from upstream API hiccups
   disappear. Principle: tape-record once, replay forever — but cassettes
   age and drift; mark "needs re-record" in CI. ([VCR integration issue][32],
   [Google provider VCR utils][33])

3. **Pulumi `providertest` with `grpc.json` invoke replay** — records at
   the *gRPC provider protocol* layer instead of HTTP. `ReplayInvokes` swaps
   the provider server with a recorded one. Principle: record at whichever
   layer is *most stable* — for Pulumi that's gRPC; for HTTP-first APIs
   (RunPod) it's HTTP. ([Pulumi providertest][34])

4. **Mock-based unit testing (`pulumi.runtime.setMocks` /
   `Mocks` interface)** — for fast inner loop, intercept all external calls
   in-memory; mocks return synthetic IDs/ARNs. Principle: two-tier — fast
   mocks for inner loop, recorded acceptance for outer loop, real API only
   nightly. ([Pulumi unit testing][35])

5. **Sandbox provider scaffolding** — `terraform-provider-sandbox` is a
   purpose-built dummy provider used to test the testing framework itself.
   Principle: meta-tests need their own meta-fixture.

6. **Property/integration: `TestCheckResourceAttr` golden assertions** —
   tests assert on the resource state diff after each `TestStep`, not on
   intermediate API calls. Principle: assert on the *observable contract*
   (the state file), not on *how it got there*.

[31]: https://developer.hashicorp.com/terraform/plugin/testing
[32]: https://github.com/hashicorp/terraform-plugin-testing/issues/190
[33]: https://github.com/hashicorp/terraform-provider-google/blob/main/google/acctest/vcr_utils.go
[34]: https://github.com/pulumi/providertest
[35]: https://www.pulumi.com/docs/iac/guides/testing/unit/

---

## Cross-cutting techniques (worth highlighting separately)

- **Property-based testing with Hypothesis** — `@given(strategies)` generates
  thousands of inputs; shrinks to minimal failing case. `fuzz_one_input()`
  bridges to coverage-guided fuzzers (HypoFuzz / Atheris). Best on parsers,
  serializers, validators — exactly the surface area in a config-DSL pipeline.
  ([Hypothesis docs][36])

- **Golden file / snapshot testing** — Go: `goldie`, `cupaloy`, `xorcare/golden`.
  Python: `pytest-regressions` (v3.0+, 2025). Scrubbers handle non-determinism
  via regex/path replacement; minimize scrubber count or the test loses signal.
  ([cupaloy][37], [pytest-regressions golden][38])

[36]: https://hypothesis.readthedocs.io/
[37]: https://github.com/bradleyjkemp/cupaloy
[38]: https://johal.in/pytest-regressions-data-golden-file-updates-2025/

---

## Synthesis: transferable patterns for an ML pipeline orchestrator

RyotenkAI maps cleanly onto these references: orchestrator ≈ k8s-controller,
RunPod provider ≈ Terraform provider, trainer subprocess ≈ kubelet's pod
runtime, vLLM inference ≈ Grafana datasource, plugin sandboxes ≈ Grafana
plugin SDK, React UI ≈ Storybook + Cypress.

### Direct maps (no adaptation)

- **Two-tier kubelet/cluster split → runner/control split.** Mirror
  `test/e2e_node` (collocated kubelet tests) for `packages/pod/` (runner +
  trainer) and `test/e2e` (cluster-as-black-box) for `packages/control/`.
  Pod packages get on-pod tests that hit the HTTP runtime directly; control
  packages get black-box tests against a real-or-fake pod.

- **kind-style ephemeral substrate.** A `make start` that boots the local
  control plane (control CLI + fake-RunPod + a stub trainer subprocess +
  vLLM-stub) in <60s is the single most leveraged investment. Argo's
  `make start` and Grafana's plugin-e2e harness are the templates.

- **Plugin SDK contract tests (community/).** Lock the plugin contract, make
  every plugin run the same Cypress-style harness against a bootstrapped
  control instance. Same shape as `@grafana/plugin-e2e`.

- **VCR cassette replay for RunPod provider.** This is the highest-ROI fault
  tolerance investment. Record `runpod` HTTP calls into YAML cassettes,
  replay in CI. Provider quirks (quota errors, eventual-consistency on pod
  status, transient 5xx) become regression tests. See Google's
  `vcr_utils.go` as the template; for Python use `vcrpy`.

- **Failure-vs-Error retry semantics with deliberate fixtures.** Argo's
  pattern is exactly the contract for `provider.launch_pod` retries — every
  retryPolicy branch (transient network, quota, image pull, auth, OOM) needs
  a fixture that triggers exactly that branch.

- **Golden snapshot tests on workflow plans, dataset manifests, run reports.**
  Go: `cupaloy` for orchestrator state. Python: `pytest-regressions` for
  trainer config / report JSON. Path-scrubbers for run IDs and timestamps.

- **Property-based on the config DSL parser.** RyotenkAI has a config builder
  on the frontend producing JSON that the backend validates. That parser is
  the single most valuable Hypothesis target — generate arbitrary configs,
  assert validation never crashes, assert serialize→parse roundtrips.

- **Storybook + Chromatic on `web/`.** Existing component stories should
  auto-convert to visual regression. PR-blocking UI Tests check.

### Need adaptation

- **Chaos-Mesh → "GPU pod chaos".** Real Chaos-Mesh is overkill, but the
  *idea* maps perfectly: parameterized fault injection with namespace-scoped
  blast radius. Build a `--chaos {kill-pod,kill-network,kill-trainer-mid-step,
  fill-disk,oom}` flag on the fake-RunPod and a probe loop that asserts
  steady-state recovery (controller observes terminal state, pod is
  cleaned up, MLflow has a final tombstone). Run in nightly only, not
  presubmit. The Litmus "ChaosWorkflow with probes" pattern is the right
  shape: `pre-check → inject → monitor → post-check → cleanup`.

- **ClusterLoader2 → "RunLoader".** Adapt CL2's YAML-as-test-spec idea:
  one YAML declares "submit 50 runs against fake-RunPod with this throughput,
  assert p99 launch latency < 30s, no orphan pods after teardown". Hook into
  the existing `pipeline/orchestrator.py` so the load test exercises the
  real code path. Replaces ad-hoc stress scripts.

- **Unreal screenshot comparison → trainer log/artifact diffing.** When a
  training run completes, the *artifact* (loss curve, eval metrics JSON,
  HF model card) is the "screenshot". Snapshot a known-good run, rerun
  nightly with a fixed seed + small dataset, diff. Out-of-bounds drift
  fails CI. This is the ML-pipeline analogue of "did it look right?".

- **Gauntlet → multi-process session orchestrator for tests.** A test that
  needs "1 control plane + 1 fake-RunPod + 1 trainer + 1 vLLM stub + 1 web"
  is exactly Gauntlet's "5 clients + 1 server" shape. A small Python
  harness modeled on `pytest-docker-tools` / `testcontainers` that wires
  ports, parses logs, and tears down beats hand-rolled fixtures.

- **Modl.ai-style monkey agent → control-CLI fuzz bot.** A simple agent that
  exercises arbitrary control CLI sequences (start, cancel, restart, retry,
  delete-while-running) against the local stack. Anomaly oracle = "did the
  state machine end in a valid terminal state, or wedge?".

### Don't directly map (but inform mindset)

- **K8s Prow flake-quarantine workflow** — the *culture* matters more than
  the tool. Adopt `[Flaky]` tagging: persistent flakes get tagged, run on
  a separate non-blocking job, surfaced on a dashboard. Never silently
  retry without tagging.

- **Replay-based regression in games** — record a control-CLI session +
  all its HTTP traffic, replay against a new build, diff terminal states.
  Same idea as VCR cassettes but at the *user-flow* layer instead of the
  HTTP layer.

### Grouping by category (proposal-ready)

| Category | Layer in RyotenkAI | Concrete tool/pattern |
|---|---|---|
| **Contract tests** | Plugin SDK (`packages/community/`), `IMLflowManager`, `IPodLifecycleClient` | testify table-tests + protocol fixture; same harness for every plugin (à la Grafana SDK) |
| **Simulation/mocking** | RunPod provider, vLLM, MLflow | `vcrpy` cassettes (record once, replay nightly); in-memory fakes for inner loop |
| **Chaos/fault injection** | Pod lifecycle, network, trainer | `--chaos` flag on fake-RunPod; nightly only; Litmus-style probes for steady-state |
| **Property-based** | Config DSL, plugin manifest parser, retry expr eval | Hypothesis with `@given`; `fuzz_one_input` for HypoFuzz/Atheris in CI nightly |
| **Snapshot/golden** | Workflow plans, run reports, dataset manifests, eval metrics | `cupaloy` (Go), `pytest-regressions` (Python); path-scrubbers for run IDs/timestamps |
| **End-to-end** | Whole stack (orchestrator → fake-RunPod → trainer → vLLM → UI) | `make start` profile + tagged Go/Python e2e; kind-style sub-60s boot |
| **Observability-as-tests** | Long-running training jobs | `Eventually`/`Consistently`-style polling helpers; SIGUSR1-equivalent progress dump on stuck tests |
| **Visual regression** | `web/` design system + dashboards | Chromatic on existing Storybook; Playwright/Cypress page snapshots for full dashboards |
| **Performance regression** | Orchestrator throughput, trainer steps/sec | "RunLoader" YAML-as-test (CL2-shape); CSV-profiler-style trended metrics with explicit budgets |
| **Flake management** | All of CI | `[Flaky]` tag → non-blocking lane; triage dashboard fingerprinting failures by stack-trace |

### Highest-ROI next actions (if writing the proposal)

1. VCR cassettes for the RunPod provider (kills 80% of integration flakes).
2. `make start` ephemeral local stack — every test that's not a unit test
   runs against this. Without it, nothing else scales.
3. Plugin SDK contract harness — every plugin runs the same e2e once.
4. Hypothesis on the config DSL parser — cheapest fuzz with the most yield.
5. Snapshot tests on run-report JSON — catches silent ML drift.

Everything else is adoption-by-extension once those five exist.
