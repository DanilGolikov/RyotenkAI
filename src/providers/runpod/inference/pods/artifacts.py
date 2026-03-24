from __future__ import annotations

from src.constants import INFERENCE_DIRNAME


def render_readme(*, manifest_filename: str, endpoint_url: str) -> str:
    return f"""## Inference endpoint (RunPod Pod + Network Volume)

- **Manifest**: `{manifest_filename}`
- **Endpoint (via SSH tunnel)**: `{endpoint_url}`

### Requirements

Environment variables:

```bash
export RUNPOD_API_KEY="..."
export HF_TOKEN="..."  # (usually already in project secrets.env)
```

You also need the SSH key (listed in the manifest), and preferably a public key file next to it: `<key_path>.pub`.

### Quick commands

Use Python from the project venv:

```bash
python {INFERENCE_DIRNAME}/chat_inference.py
```

Or if the venv is activated:

1) **Interactive chat** (starts the Pod, sets up SSH tunnel, runs merge, starts vLLM, then opens chat; on exit stops the Pod):

```bash
python {INFERENCE_DIRNAME}/chat_inference.py
```

2) **Stop/delete Pod** — inside interactive chat:

- `/exit` → stop Pod (Pod kept for next run)
- `/stop` → delete Pod (volume kept; next run may fail if no free GPUs in the volume DC)
- `/clear` → wipe volume data and delete Pod

### Notes

- `hf_cache` and artifacts live on the Network Volume (see `serve.hf_cache_dir` in manifest) and persist across sessions.
- After provisioning the Pod is parked (stopped) — this saves GPU cost, but Network Volume storage still bills.
"""


CHAT_SCRIPT = r"""#!/usr/bin/env python3
'''
Interactive chat with RunPod Pod vLLM endpoint (SSH tunnel).

Flow:
  1) Ensure Pod exists for the configured Network Volume (create if missing)
  2) Start/resume Pod
  3) Wait for SSH port mapping (publicIp + portMappings[22])
  4) Create SSH tunnel: localhost:{serve_port} → 127.0.0.1:{serve_port} inside Pod
  5) Acquire volume lock (best-effort)
  6) Merge LoRA adapter into base model (cache: /workspace/hf_cache)
  7) Start vLLM serve in background
  8) Wait until /v1/models is ready via tunnel
  9) Start interactive streaming chat
  10) On exit: stop vLLM + release lock + stop Pod (avoid GPU costs)
'''
from __future__ import annotations

import json
import os
import shlex
import ssl
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

RUNTIME: dict[str, Any] = {}


def _load_manifest() -> dict:
    here = Path(__file__).resolve().parent
    p = here / "inference_manifest.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _dotenv_get(path: Path, key: str) -> str | None:
    '''
    Minimal .env parser (KEY=VALUE) for local convenience.
    Supports `export KEY=...` and quoted values.
    '''
    try:
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            k, v = line.split("=", 1)
            if k.strip() != key:
                continue
            value = v.strip()
            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            value = value.strip()
            return value or None
    except Exception:
        return None
    return None


def _find_secrets_file(start_dir: Path) -> Path | None:
    # Repo convention:
    # - <repo_root>/secrets.env
    # - <repo_root>/config/secrets.env (legacy)
    for p in [start_dir, *start_dir.parents]:
        cand = p / "secrets.env"
        if cand.exists():
            return cand
        cand = p / "config" / "secrets.env"
        if cand.exists():
            return cand
    return None


def _require_env(name: str) -> str:
    secrets_path = _find_secrets_file(Path(__file__).resolve().parent)
    if secrets_path is not None:
        v2 = _dotenv_get(secrets_path, name)
        if v2 and v2.strip():
            return v2.strip()
        raise RuntimeError(f"Secrets file exists but {name} is missing: {secrets_path}")

    v = os.environ.get(name)
    if v and v.strip():
        return v.strip()

    raise RuntimeError(
        f"Missing required env var: {name}. "
        "Set it in your shell or put it into `secrets.env` / `config/secrets.env`."
    )


def _ssl_context() -> ssl.SSLContext:
    # macOS-friendly trust store: prefer certifi bundle.
    # Set HELIX_INSECURE_SSL=1 to disable verification (NOT recommended).
    insecure = os.environ.get("HELIX_INSECURE_SSL", "").strip().lower() in {"1", "true", "yes"}
    if insecure:
        return ssl._create_unverified_context()
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _urlopen(req: urllib.request.Request, *, timeout: int):
    return urllib.request.urlopen(req, timeout=timeout, context=_ssl_context())


_HTTP_RETRIES = 3
_HTTP_RETRY_BACKOFF = 2.0  # seconds; doubles each attempt


def _is_transient_network_error(exc: Exception) -> bool:
    # True for SSL/TCP glitches and timeouts that are safe to retry.
    msg = str(exc).lower()
    transient = (
        "handshake operation timed out",
        "timed out",
        "connection reset",
        "connection refused",
        "remote end closed connection",
        "broken pipe",
        "temporary failure in name resolution",
        "name or service not known",
        "nodename nor servname provided",
    )
    return any(t in msg for t in transient)


def _http_json(
    *,
    url: str,
    method: str,
    headers: dict[str, str],
    payload: dict | None = None,
    timeout: int = 30,
) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    last_exc: Exception | None = None
    for attempt in range(1, _HTTP_RETRIES + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with _urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                if not raw:
                    return None
                try:
                    return json.loads(raw.decode("utf-8"))
                except Exception:
                    return raw.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            # HTTP errors (4xx/5xx) are not retried — propagate immediately.
            try:
                raw = e.read() or b""
            except Exception:
                raw = b""
            text = raw.decode("utf-8", errors="replace").strip()
            try:
                body = json.loads(text) if text else None
            except Exception:
                body = text or None
            raise RuntimeError(f"RunPod HTTP {e.code} for {method} {url}: {body!r}") from None
        except Exception as e:
            last_exc = e
            if not _is_transient_network_error(e) or attempt == _HTTP_RETRIES:
                break
            wait = _HTTP_RETRY_BACKOFF * (2 ** (attempt - 1))
            print(
                f"⚠️  Network error (attempt {attempt}/{_HTTP_RETRIES}), retrying in {wait:.0f}s: {e}",
                flush=True,
            )
            time.sleep(wait)

    raise RuntimeError(f"RunPod request failed: {method} {url}: {last_exc!s}") from None


def _runpod_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


_NO_CAPACITY_PHRASES = (
    "could not find any pods with required specifications",
    "there are no instances currently available",
    "no available datacenter with requested resources",
)


def _is_no_capacity_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(phrase in s for phrase in _NO_CAPACITY_PHRASES)



def _runpod_list_pods(*, rest_api_base_url: str, api_key: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    qs = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None}, doseq=True)
    url = rest_api_base_url.rstrip("/") + "/pods"
    if qs:
        url = url + "?" + qs
    out = _http_json(url=url, method="GET", headers=_runpod_headers(api_key), timeout=30)
    return out if isinstance(out, list) else []


def _runpod_get_pod(*, rest_api_base_url: str, api_key: str, pod_id: str) -> dict[str, Any]:
    url = rest_api_base_url.rstrip("/") + f"/pods/{pod_id}"
    out = _http_json(url=url, method="GET", headers=_runpod_headers(api_key), timeout=30)
    return out if isinstance(out, dict) else {}


def _runpod_create_pod(*, rest_api_base_url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = rest_api_base_url.rstrip("/") + "/pods"
    out = _http_json(url=url, method="POST", headers=_runpod_headers(api_key), payload=payload, timeout=120)
    return out if isinstance(out, dict) else {}


def _runpod_start_pod(*, rest_api_base_url: str, api_key: str, pod_id: str) -> None:
    url = rest_api_base_url.rstrip("/") + f"/pods/{pod_id}/start"
    _ = _http_json(url=url, method="POST", headers=_runpod_headers(api_key), payload=None, timeout=60)


def _runpod_stop_pod(*, rest_api_base_url: str, api_key: str, pod_id: str) -> None:
    url = rest_api_base_url.rstrip("/") + f"/pods/{pod_id}/stop"
    _ = _http_json(url=url, method="POST", headers=_runpod_headers(api_key), payload=None, timeout=60)


def _runpod_delete_pod(*, rest_api_base_url: str, api_key: str, pod_id: str) -> None:
    url = rest_api_base_url.rstrip("/") + f"/pods/{pod_id}"
    _ = _http_json(url=url, method="DELETE", headers=_runpod_headers(api_key), payload=None, timeout=60)


def _wait_for_pod_ssh_ready(*, rest_api_base_url: str, api_key: str, pod_id: str, timeout_seconds: int) -> tuple[str, int]:
    deadline = time.time() + timeout_seconds
    public_ip = ""
    status = ""
    last_print = 0.0
    last_preview = ""

    while time.time() < deadline:
        pod = _runpod_get_pod(rest_api_base_url=rest_api_base_url, api_key=api_key, pod_id=pod_id)
        status = str(pod.get("desiredStatus") or "")
        public_ip = str(pod.get("publicIp") or "").strip()
        mappings = pod.get("portMappings") or {}
        ssh_port: int | None = None

        # RunPod shapes (observed / defensive):
        # - dict: {"22": 23828} or {"22/tcp": "23828"} or {"22": {"hostPort": 23828}}
        # - list: [{"containerPort": 22, "hostPort": 23828}, ...]
        if isinstance(mappings, dict):
            cand_keys: list[object] = ["22", 22, "22/tcp", "tcp/22"]
            for k in cand_keys:
                if k not in mappings:
                    continue
                v = mappings.get(k)
                if isinstance(v, dict):
                    v = v.get("hostPort") or v.get("publicPort") or v.get("port")
                if isinstance(v, int):
                    ssh_port = v
                    break
                if isinstance(v, str) and v.strip().isdigit():
                    ssh_port = int(v.strip())
                    break
                try:
                    ssh_port = int(v)  # type: ignore[arg-type]
                    break
                except Exception:
                    continue
        elif isinstance(mappings, list):
            for it in mappings:
                if not isinstance(it, dict):
                    continue
                cport = it.get("containerPort") or it.get("internalPort") or it.get("port")
                hport = it.get("hostPort") or it.get("externalPort") or it.get("publicPort")
                try:
                    if int(cport) != 22:
                        continue
                    ssh_port = int(hport)
                    break
                except Exception:
                    continue

        tcp_ok: bool | None = None
        if status == "RUNNING" and public_ip and ssh_port:
            # `publicIp` + portMappings do not guarantee the SSH service is accepting connections yet.
            try:
                import socket

                with socket.create_connection((public_ip, int(ssh_port)), timeout=3):
                    tcp_ok = True
            except Exception:
                tcp_ok = False

            if tcp_ok:
                return public_ip, int(ssh_port)

        now = time.time()
        if status or public_ip or ssh_port:
            tcp_s = "" if tcp_ok is None else f" tcp={'OK' if tcp_ok else 'NO'}"
            preview = f"status={status or '∅'} ip={public_ip or '∅'} ssh_port={ssh_port if ssh_port is not None else '∅'}{tcp_s}"
            if preview != last_preview or (now - last_print) > 10:
                remaining = int(max(0, deadline - now))
                print(f"⏳ Waiting for SSH... {preview} (remaining {remaining}s)", flush=True)
                last_preview = preview
                last_print = now
        time.sleep(2)

    raise RuntimeError(f"Pod not SSH-ready in time (status={status}, public_ip={public_ip}, ssh_port={ssh_port})")


def _read_public_key(key_path: Path) -> str:
    pub_path = Path(str(key_path) + ".pub")
    try:
        if pub_path.exists():
            return pub_path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""
    return ""


def _ssh_base_args(*, host: str, port: int, key_path: Path) -> list[str]:
    return [
        "ssh",
        "-i",
        str(key_path),
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "BatchMode=yes",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "LogLevel=ERROR",
        f"root@{host}",
    ]


def _ssh_exec(*, host: str, port: int, key_path: Path, command: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    # Always run via bash -lc for consistent PATH and env.
    remote = f"bash -lc {shlex.quote(command)}"
    return subprocess.run(
        [*_ssh_base_args(host=host, port=port, key_path=key_path), remote],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=timeout,
    )


def _ensure_ssh_tunnel(*, host: str, port: int, key_path: Path, local_port: int, remote_port: int) -> None:
    # Best-effort check:
    # we only reuse a tunnel if it targets THIS pod (host + ssh port) and THIS mapping.
    mapping = f"{local_port}:127.0.0.1:{remote_port}"
    ps_check = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    for ln in (ps_check.stdout or "").splitlines():
        if "ssh" not in ln:
            continue
        if mapping not in ln:
            continue
        if f"root@{host}" in ln and f"-p {port}" in ln:
            print(f"✅ SSH tunnel already active: localhost:{local_port} → 127.0.0.1:{remote_port}")
            return
        raise RuntimeError(
            f"Local port {local_port} is already forwarded by another SSH tunnel.\n"
            f"Expected tunnel to root@{host}:{port} with mapping '{mapping}', but found:\n{ln}\n"
            "Close the old tunnel (kill the ssh process) and retry."
        )

    print(f"🔗 Creating SSH tunnel: localhost:{local_port} → 127.0.0.1:{remote_port}")
    tunnel_cmd = [
        "ssh",
        "-f",
        "-N",
        "-i",
        str(key_path),
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "BatchMode=yes",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "LogLevel=ERROR",
        "-L",
        f"{local_port}:127.0.0.1:{remote_port}",
        f"root@{host}",
    ]

    try:
        subprocess.run(
            tunnel_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            stdin=subprocess.DEVNULL,
        )
        time.sleep(1)
        print("✅ SSH tunnel created")
    except subprocess.CalledProcessError as e:
        output = (e.stdout or "") + (e.stderr or "")
        if "address already in use" in (output or "").lower():
            raise RuntimeError(
                f"Tunnel port {local_port} is already in use, but no matching tunnel was found for this pod.\n"
                f"Details: {output.strip()}"
            )
        else:
            raise RuntimeError(f"Tunnel creation failed: {output.strip()}")


def _wait_http_ok(*, url: str, timeout_seconds: int, interval_seconds: float = 2.0) -> None:
    deadline = time.time() + timeout_seconds
    last_err = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception as e:
            last_err = str(e)
        time.sleep(interval_seconds)
    raise RuntimeError(f"Timed out waiting for readiness: {url} (last_err={last_err})")


def _load_chat_template(*, base_model_id: str) -> str | None:
    '''
    vLLM requires chat_template for /v1/chat/completions on some setups.

    We fetch it from the base HF model tokenizer (client-side).
    '''
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None

    try:
        tok = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=False)
        template = getattr(tok, "chat_template", None)
        return template if isinstance(template, str) and template.strip() else None
    except Exception:
        return None


def _print_help() -> None:
    print("\nCommands:")
    print("  - /paste  → paste a multi-line prompt (end with a line /end)")
    print("  - /file <path> → send file contents as one message")
    print("  - /reset  → clear chat history (keep only system prompt)")
    print("  - /status → show pod status")
    print("  - /exit   → normal exit (stop pod, data kept)")
    print("  - /stop   → delete pod (volume kept; next run must recreate pod and may fail without free GPUs)")
    print("  - /clear  → clear volume and delete pod (same as /stop but also deletes volume data)")
    print("  - exit|quit|q → same as /exit\n")


def _input_with_timeout(prompt: str, timeout_sec: float) -> str | None:
    import sys
    import select
    print(prompt, end="", flush=True)
    if sys.platform == "win32":
        try:
            return input()
        except EOFError:
            return None
    rlist, _, _ = select.select([sys.stdin], [], [], timeout_sec)
    if rlist:
        line = sys.stdin.readline()
        if not line:
            return None
        return line.rstrip('\n')
    return None


def _read_multiline(timeout_sec: float) -> str | None:
    import sys
    import select
    print("📋 Paste mode: paste text, end with a line /end", flush=True)
    lines: list[str] = []
    while True:
        if sys.platform == "win32":
            try:
                line = input()
            except EOFError:
                return None
        else:
            rlist, _, _ = select.select([sys.stdin], [], [], timeout_sec)
            if not rlist:
                return None
            line = sys.stdin.readline()
            if not line:
                return None
            line = line.rstrip('\n')
        if line.strip() == "/end":
            break
        lines.append(line)
    return "\n".join(lines).strip("\n")


def _start_chat(
    base_url: str,
    model_name: str,
    *,
    base_model_id: str,
    rest_base: str,
    api_key: str,
    pod_id: str,
    system_prompt: str | None = None,
) -> str:
    import urllib.request
    import urllib.error

    print(f"\n💬 Chat with {model_name.split('/')[-1]}")
    print("Type 'exit' or 'quit' to end session")
    print("For multi-line prompts: type /paste, paste text, then /end")
    print("Hint: /help\n")

    # System prompt is resolved at deploy time and stored as text in inference_manifest.json.
    # This script receives it ready-to-use — no file access or MLflow needed here.
    if system_prompt:
        print("✅ System prompt loaded from manifest.")

    # Override with env var if explicitly set
    if "HELIXQL_SYSTEM_PROMPT" in os.environ:
        system_prompt = os.environ["HELIXQL_SYSTEM_PROMPT"].strip() or None
        if system_prompt:
            print("✅ Overridden system prompt from HELIXQL_SYSTEM_PROMPT env var.")
        else:
            print("⚠️  HELIXQL_SYSTEM_PROMPT is set but empty, continuing without system prompt.")

    temperature = float(os.environ.get("HELIXQL_TEMPERATURE", "0"))
    max_tokens = int(os.environ.get("HELIXQL_MAX_TOKENS", "512"))

    chat_template = _load_chat_template(base_model_id=base_model_id)
    if chat_template:
        print("✅ chat_template loaded (client-side)", flush=True)
    else:
        print("⚠️  chat_template NOT found; /v1/chat/completions may fail on some setups", flush=True)

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}]
    chat_url = base_url.rstrip("/") + "/chat/completions"
    completions_url = base_url.rstrip("/") + "/completions"

    def _strip_code_fences(text: str) -> str:
        s = text.strip()
        if not s.startswith("```"):
            return s
        lines = s.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _messages_to_prompt(msgs: list[dict[str, str]]) -> str:
        # Fallback path if /chat/completions is unstable: use /completions with a rendered prompt.
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=False)
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            parts: list[str] = []
            for m in msgs:
                role = str(m.get("role") or "user")
                content = str(m.get("content") or "")
                parts.append(f"{role}: {content}")
            parts.append("assistant:")
            return "\n".join(parts)

    def _is_err_text(text: str) -> bool:
        s = text.strip()
        return s.startswith("[HTTP ") or s.startswith("[ERROR]") or s.startswith("[REQUEST FAILED]")

    def _looks_like_transport_issue(text: str) -> bool:
        s = text.lower()
        return (
            "connection refused" in s
            or "remote end closed connection" in s
            or "connection reset" in s
            or "timed out" in s
        )

    def _recover_transport() -> bool:
        host = RUNTIME.get("ssh_host")
        port = RUNTIME.get("ssh_port")
        key_path = RUNTIME.get("ssh_key_path")
        serve_port0 = RUNTIME.get("serve_port")
        readiness_url = RUNTIME.get("readiness_url") or (base_url.rstrip("/") + "/models")

        if not (isinstance(host, str) and host.strip()):
            return False
        try:
            port_i = int(port)
            serve_port_i = int(serve_port0)
        except Exception:
            return False

        if isinstance(key_path, Path):
            kp = key_path
        else:
            try:
                kp = Path(str(key_path)).expanduser()
            except Exception:
                return False

        # 1) Re-create tunnel (best-effort)
        try:
            _ensure_ssh_tunnel(host=host, port=port_i, key_path=kp, local_port=serve_port_i, remote_port=serve_port_i)
        except Exception as e:
            msg = str(e)
            ml = msg.lower()
            if "connection refused" in ml or "timed out" in ml or "no route to host" in ml:
                print(f"⚠️  Tunnel restore failed: {msg}", flush=True)
                try:
                    new_host, new_port = _wait_for_pod_ssh_ready(
                        rest_api_base_url=rest_base,
                        api_key=api_key,
                        pod_id=pod_id,
                        timeout_seconds=60,
                    )
                    host = new_host
                    port_i = int(new_port)
                    RUNTIME["ssh_host"] = host
                    RUNTIME["ssh_port"] = port_i
                    _ensure_ssh_tunnel(
                        host=host,
                        port=port_i,
                        key_path=kp,
                        local_port=serve_port_i,
                        remote_port=serve_port_i,
                    )
                except Exception as e2:
                    print(f"⚠️  SSH still not ready: {e2}", flush=True)
                    return False
            else:
                print(f"⚠️  Tunnel restore failed: {e}", flush=True)
                return False

        # 2) Quick readiness probe
        try:
            _wait_http_ok(url=str(readiness_url), timeout_seconds=30, interval_seconds=2.0)
            return True
        except Exception:
            pass

        # 3) Try restart vLLM (if we have the exact start cmd from main)
        start_cmd = RUNTIME.get("start_vllm_cmd")
        if isinstance(start_cmd, str) and start_cmd.strip():
            print("♻️  Endpoint not responding. Restarting vLLM...", flush=True)
            res = _ssh_exec(host=host, port=port_i, key_path=kp, command=start_cmd, timeout=30)
            if res.returncode != 0:
                print(f"❌ Failed to restart vLLM. stderr={res.stderr.strip()[:500]}", flush=True)
                log_file = RUNTIME.get("vllm_log_file")
                if isinstance(log_file, str) and log_file.strip():
                    try:
                        tail_cmd = f"tail -n 80 {shlex.quote(log_file)}"
                        r2 = _ssh_exec(host=host, port=port_i, key_path=kp, command=tail_cmd, timeout=20)
                        out = (r2.stdout or "").strip()
                        if out:
                            print("\n--- vLLM log (tail) ---\n" + out[-4000:] + "\n--- end ---\n", flush=True)
                    except Exception:
                        pass
                return False
            try:
                _wait_http_ok(url=str(readiness_url), timeout_seconds=180, interval_seconds=2.0)
                return True
            except Exception as e:
                print(f"❌ vLLM still not ready: {e}", flush=True)
                return False

        return False

    oneshot = os.environ.get("HELIX_CHAT_ONESHOT", "").strip()
    oneshot_mode = bool(oneshot)

    while True:
        try:
            if oneshot:
                raw = oneshot
                oneshot = ""
            else:
                raw = _input_with_timeout("You: ", 300.0)
                if raw is None:
                    print("\n⏳ No activity for 5 minutes. Auto-exiting to save cost...")
                    return "exit"
            cmd = raw.strip()

            if cmd.lower() in ("exit", "quit", "q", "/exit"):
                print("👋 Goodbye! Stopping pod...", flush=True)
                return "exit"

            if cmd == "/stop":
                print("🛑 Goodbye! Pod will be deleted...", flush=True)
                print("⚠️  Note: the next run must create a pod again and may fail without free GPUs in the volume DC.")
                print("💡 To end the session and resume later, use /exit (pod stops but is kept).")
                return "stop"

            if cmd == "/clear":
                print("🧹 Goodbye! Clearing volume and deleting pod...", flush=True)
                print("⚠️  Note: the next run must create a pod again and may fail without free GPUs in the volume DC.")
                return "clear"

            if cmd == "/status":
                try:
                    pod = _runpod_get_pod(rest_api_base_url=rest_base, api_key=api_key, pod_id=pod_id)
                    st = pod.get("desiredStatus", "UNKNOWN")
                    ip = pod.get("publicIp", "none")
                    gpu = (pod.get("machine", {}) or {}).get("gpuDisplayName", "unknown")
                    print(f"📊 Status: {st} | IP: {ip} | GPU: {gpu}")
                except Exception as e:
                    print(f"❌ Status error: {e}")
                continue

            if cmd == "/help":
                _print_help()
                continue

            if cmd == "/reset":
                messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
                suffix = " (system prompt kept)." if system_prompt else ""
                print(f"🧹 History cleared.{suffix}")
                continue

            if cmd == "/paste":
                user_input = _read_multiline(300.0)
                if user_input is None:
                    print("\n⏳ No activity for 5 minutes. Auto-exiting to save cost...")
                    return "exit"
            elif cmd.startswith("/file "):
                path_str = cmd[len("/file ") :].strip()
                try:
                    user_input = Path(path_str).expanduser().read_text(encoding="utf-8")
                except Exception as e:
                    print(f"❌ Could not read file: {e}")
                    continue
            else:
                user_input = raw

            if not user_input.strip():
                continue

            messages.append({"role": "user", "content": user_input})

            payload_chat = {
                "model": model_name,
                "messages": messages,
                "stream": True,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if chat_template:
                payload_chat["chat_template"] = chat_template

            print("Assistant: ", end="", flush=True)
            full_response = ""

            def _chat_once(*, url: str, base_payload: dict, stream: bool) -> tuple[str, bool]:
                pl = dict(base_payload)
                pl["stream"] = bool(stream)
                req2 = urllib.request.Request(
                    url,
                    data=json.dumps(pl).encode("utf-8"),
                    headers={"Content-Type": "application/json", "Accept": "text/event-stream" if stream else "application/json"},
                )
                try:
                    with urllib.request.urlopen(req2, timeout=180) as response:
                        ct = str(response.headers.get("Content-Type") or "")
                        if stream and "text/event-stream" in ct:
                            chunks: list[str] = []
                            printed_any = False
                            for raw_line in response:
                                line = raw_line.decode("utf-8", errors="replace").strip()
                                if not line:
                                    continue
                                if not line.startswith("data:"):
                                    continue
                                data = line[len("data:") :].strip()
                                if data == "[DONE]":
                                    break
                                try:
                                    obj = json.loads(data)
                                except Exception:
                                    continue

                                # OpenAI-compatible streaming chunk:
                                # - chat: choices[0].delta.content
                                # - completions-like: choices[0].text
                                choice0 = (obj.get("choices") or [{}])[0] if isinstance(obj, dict) else {}
                                finish_reason = choice0.get("finish_reason") if isinstance(choice0, dict) else None
                                delta = choice0.get("delta") if isinstance(choice0, dict) else None
                                text_chunk = ""
                                if isinstance(delta, dict):
                                    c = delta.get("content")
                                    if isinstance(c, str):
                                        text_chunk = c
                                    elif isinstance(c, list):
                                        # Future-proof: content can be list of parts
                                        parts: list[str] = []
                                        for part in c:
                                            if isinstance(part, str):
                                                parts.append(part)
                                            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                                                parts.append(part["text"])
                                        text_chunk = "".join(parts)
                                if not text_chunk and isinstance(choice0, dict) and isinstance(choice0.get("text"), str):
                                    text_chunk = choice0["text"]

                                if text_chunk:
                                    chunks.append(text_chunk)
                                    print(text_chunk, end="", flush=True)
                                    printed_any = True

                                # Some OpenAI-compatible gateways may not send the final [DONE] sentinel.
                                # Break on finish_reason to avoid hanging on an open SSE connection.
                                if finish_reason is not None:
                                    break
                            return "".join(chunks), printed_any

                        # Non-streaming (or unexpected content-type): parse JSON body
                        raw = response.read().decode("utf-8", errors="replace")
                        if not raw.strip():
                            return "", False
                        try:
                            obj = json.loads(raw)
                        except Exception:
                            return raw, False
                        if isinstance(obj, dict) and isinstance(obj.get("error"), dict):
                            err = obj["error"]
                            msg = err.get("message") or err
                            return f"[ERROR] {msg}", False
                        try:
                            choice0 = (obj.get("choices") or [{}])[0]
                            if isinstance(choice0, dict):
                                msg = choice0.get("message") or {}
                                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                                    return msg["content"], False
                                if isinstance(choice0.get("text"), str):
                                    return choice0["text"], False
                        except Exception:
                            pass
                        return "", False
                except urllib.error.HTTPError as e:
                    body = ""
                    try:
                        body = e.read().decode("utf-8", errors="replace")
                    except Exception:
                        body = ""
                    snippet = body.strip().replace("\n", "\\n")
                    if len(snippet) > 2000:
                        snippet = snippet[:2000] + "…"
                    return f"[HTTP {getattr(e, 'code', '?')}] {snippet or e.reason}", False
                except Exception as e:
                    return f"[REQUEST FAILED] {e!s}", False

            def _ask_once() -> tuple[str, bool]:
                # Prefer streaming, but fallback to raw completions / non-streaming if needed.
                url_used = chat_url
                payload_used: dict = payload_chat
                resp, printed = _chat_once(url=url_used, base_payload=payload_used, stream=True)

                # If chat-completions fails/empty (vLLM can be picky about chat templates),
                # fall back to raw completions with a rendered prompt.
                if _is_err_text(resp) or not resp.strip():
                    prompt = _messages_to_prompt(messages)
                    payload_comp = {
                        "model": model_name,
                        "prompt": prompt,
                        "stream": True,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    alt, alt_printed = _chat_once(url=completions_url, base_payload=payload_comp, stream=True)
                    if alt.strip() and not _is_err_text(alt):
                        url_used = completions_url
                        payload_used = payload_comp
                        resp, printed = alt, alt_printed
                    elif not resp.strip() and alt.strip():
                        # Still show something useful for debugging.
                        url_used = completions_url
                        payload_used = payload_comp
                        resp, printed = alt, alt_printed

                # As a last resort, try non-streaming.
                if not resp.strip():
                    resp2, _ = _chat_once(url=url_used, base_payload=payload_used, stream=False)
                    resp = resp2
                return resp, printed

            full_response, printed_any = _ask_once()

            # Recover transport issues (tunnel drop / endpoint restart) once per message.
            if _is_err_text(full_response) and _looks_like_transport_issue(full_response):
                print("\n⚠️  Connection to endpoint seems lost. Trying to recover...", flush=True)
                ok = _recover_transport()
                if ok:
                    print("🔁 Retrying request...", flush=True)
                    print("Assistant: ", end="", flush=True)
                    full_response, printed_any = _ask_once()
                else:
                    print("⚠️  Could not restore connection. Retry later or restart the script.", flush=True)

            if full_response and not printed_any:
                print(full_response, end="", flush=True)
                printed_any = True

            print()
            assistant_text = _strip_code_fences(full_response)

            # If request failed, keep history consistent: rollback last user message and continue.
            if _is_err_text(assistant_text):
                if oneshot_mode:
                    return "exit"
                if messages and messages[-1].get("role") == "user":
                    messages.pop()
                continue

            if assistant_text.strip():
                messages.append({"role": "assistant", "content": assistant_text.strip()})
            else:
                # Keep history consistent: rollback last user message if model returned empty content.
                if messages and messages[-1].get("role") == "user":
                    messages.pop()

            # Non-interactive debug mode: run exactly one request and exit.
            if oneshot_mode:
                return "exit"

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Ending session — Pod will be stopped next...", flush=True)
            return "exit"
        except EOFError:
            print("\n\n👋 Goodbye! Ending session — Pod will be stopped next...", flush=True)
            return "exit"
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return "exit"

    return "exit"


def main() -> int:
    m = _load_manifest()
    api_key = _require_env("RUNPOD_API_KEY")

    rest_base = str(m["runpod"]["rest_api_base_url"]).rstrip("/")
    volume = m["runpod"].get("network_volume")
    pod_cfg = m["runpod"]["pod"]

    volume_id = str((volume or {}).get("id") or "").strip()

    pod_name = str(pod_cfg.get("name") or "").strip()
    if not pod_name:
        raise RuntimeError("manifest.runpod.pod.name is empty")

    serve = m["serve"]
    serve_port = int(serve.get("port") or 8000)
    base_url = str(m["endpoint"]["client_base_url"]).rstrip("/")

    model = m["model"]
    base_model_id = str(model.get("base_model_id") or "")
    adapter_ref = str(model.get("adapter_ref") or "")
    trust_remote_code = bool(model.get("trust_remote_code", False))
    config_hash = str(m.get("config_hash") or "")
    vllm_cfg = m.get("vllm") or {}

    key_path = Path(str(m["ssh"]["key_path"])).expanduser()
    if not key_path.exists():
        raise RuntimeError(f"SSH key not found: {key_path}")

    # Ensure pod exists
    list_params: dict = {"computeType": "GPU", "name": pod_name}
    if volume_id:
        list_params["networkVolumeId"] = volume_id
    pods = _runpod_list_pods(
        rest_api_base_url=rest_base,
        api_key=api_key,
        params=list_params,
    )
    chosen = None
    for p in pods:
        if isinstance(p, dict) and str(p.get("desiredStatus") or "") == "RUNNING":
            chosen = p
            break
    if chosen is None:
        for p in pods:
            if isinstance(p, dict) and str(p.get("desiredStatus") or "") == "EXITED":
                chosen = p
                break
    if chosen is None and pods:
        chosen = pods[0]

    if chosen is None:
        print("ℹ️  Pod not found → creating a new one")
        hf_token = _require_env("HF_TOKEN")
        public_key = _read_public_key(key_path)
        env = {"HF_TOKEN": hf_token}
        if public_key:
            env["PUBLIC_KEY"] = public_key

        payload = {
            "name": pod_name,
            "cloudType": "SECURE",
            "computeType": "GPU",
            "imageName": str(pod_cfg.get("image_name") or ""),
            "gpuCount": int(pod_cfg.get("gpu_count") or 1),
            "gpuTypeIds": list(pod_cfg.get("gpu_type_ids") or []),
            "gpuTypePriority": "availability",
            # NOTE:
            # We intentionally do NOT pass dataCenterIds here.
            # With networkVolumeId — RunPod derives the DC from volume location.
            # Without networkVolumeId — RunPod searches all available DCs for the GPU.
            "allowedCudaVersions": pod_cfg.get("allowed_cuda_versions"),
            "containerDiskInGb": int(pod_cfg.get("container_disk_gb") or 50),
            "volumeMountPath": "/workspace",
            "ports": list(pod_cfg.get("ports") or ["22/tcp"]),
            "supportPublicIp": True,
            "interruptible": False,
            "locked": False,
            "env": env,
        }
        if volume_id:
            payload["networkVolumeId"] = volume_id
        else:
            payload["volumeInGb"] = int(pod_cfg.get("volume_disk_gb") or 50)
        payload = {k: v for k, v in payload.items() if v is not None}
        try:
            created = _runpod_create_pod(rest_api_base_url=rest_base, api_key=api_key, payload=payload)
        except Exception as e:
            if _is_no_capacity_error(e):
                dc = str((volume or {}).get("data_center_id") or "").strip()
                if volume_id:
                    print("❌ RunPod could not create Pod: no available instances (GPU capacity) in the Network Volume datacenter.")
                    if dc:
                        print(f"📍 DC (volume): {dc}")
                    print(f"🧩 Request: gpu_count={payload.get('gpuCount')} gpu_type_ids={payload.get('gpuTypeIds')}")
                    print("💡 Try later; widen GPU type list; or create a volume in another DC and redeploy.")
                else:
                    print("❌ RunPod could not create Pod: no available instances (GPU capacity).")
                    print(f"🧩 Request: gpu_count={payload.get('gpuCount')} gpu_type_ids={payload.get('gpuTypeIds')}")
                    print("💡 Try later or widen the GPU type list in config.")
                return 1
            raise
        pod_id = str(created.get("id") or "").strip()
        if not pod_id:
            raise RuntimeError(f"Pod created but id missing: {created}")
    else:
        pod_id = str(chosen.get("id") or "").strip()
        if not pod_id:
            raise RuntimeError(f"Invalid pod object: {chosen}")

    print(f"✅ Pod selected: {pod_id} ({pod_name})")

    # Start pod (best-effort) and handle "zero GPU on restart" by recreating Pod if needed.
    # Some RunPod calls can return 5xx while still succeeding; also starting an already RUNNING pod may error.
    status0 = ""
    try:
        pod0 = _runpod_get_pod(rest_api_base_url=rest_base, api_key=api_key, pod_id=pod_id)
        status0 = str(pod0.get("desiredStatus") or "")
    except Exception:
        status0 = ""

    start_failed_msg = ""
    if status0 == "RUNNING":
        print("✅ Pod already RUNNING")
    else:
        print("▶️  Starting Pod...")
        try:
            _runpod_start_pod(rest_api_base_url=rest_base, api_key=api_key, pod_id=pod_id)
        except Exception as e:
            start_failed_msg = str(e)
            print(f"⚠️  Start request failed: {e}")

    try:
        if start_failed_msg and "not enough free gpus" in start_failed_msg.lower():
            raise RuntimeError("Fast-failing wait because host machine has no free GPUs.")

        public_ip, ssh_port = _wait_for_pod_ssh_ready(
            rest_api_base_url=rest_base,
            api_key=api_key,
            pod_id=pod_id,
            timeout_seconds=5 * 60,
        )
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user. Stopping Pod...", flush=True)
        try:
            _runpod_stop_pod(rest_api_base_url=rest_base, api_key=api_key, pod_id=pod_id)
            print("✅ Pod stopped", flush=True)
        except Exception as e:
            print(f"⚠️  Stop error: {e}", flush=True)
        return 130
    except Exception as e:
        print(f"⚠️  Pod failed to become ready: {e}")
        print("♻️  Recreating Pod (mitigation for 'zero GPU on restart')...")

        # Best-effort stop + delete old pod (network volume persists)
        try:
            _runpod_stop_pod(rest_api_base_url=rest_base, api_key=api_key, pod_id=pod_id)
        except Exception:
            pass
        try:
            _runpod_delete_pod(rest_api_base_url=rest_base, api_key=api_key, pod_id=pod_id)
        except Exception:
            pass

        hf_token = _require_env("HF_TOKEN")
        public_key = _read_public_key(key_path)
        env = {"HF_TOKEN": hf_token}
        if public_key:
            env["PUBLIC_KEY"] = public_key

        payload = {
            "name": pod_name,
            "cloudType": "SECURE",
            "computeType": "GPU",
            "imageName": str(pod_cfg.get("image_name") or ""),
            "gpuCount": int(pod_cfg.get("gpu_count") or 1),
            "gpuTypeIds": list(pod_cfg.get("gpu_type_ids") or []),
            "gpuTypePriority": "availability",
            # NOTE:
            # We intentionally do NOT pass dataCenterIds here.
            # With networkVolumeId — RunPod derives the DC from volume location.
            # Without networkVolumeId — RunPod searches all available DCs for the GPU.
            "allowedCudaVersions": pod_cfg.get("allowed_cuda_versions"),
            "containerDiskInGb": int(pod_cfg.get("container_disk_gb") or 50),
            "volumeMountPath": "/workspace",
            "ports": list(pod_cfg.get("ports") or ["22/tcp"]),
            "supportPublicIp": True,
            "interruptible": False,
            "locked": False,
            "env": env,
        }
        if volume_id:
            payload["networkVolumeId"] = volume_id
        else:
            payload["volumeInGb"] = int(pod_cfg.get("volume_disk_gb") or 50)
        payload = {k: v for k, v in payload.items() if v is not None}
        try:
            created = _runpod_create_pod(rest_api_base_url=rest_base, api_key=api_key, payload=payload)
        except Exception as e:
            if _is_no_capacity_error(e):
                dc = str((volume or {}).get("data_center_id") or "").strip()
                if volume_id:
                    print("❌ RunPod could not recreate Pod: no available instances (GPU capacity) in the Network Volume datacenter.")
                    if dc:
                        print(f"📍 DC (volume): {dc}")
                else:
                    print("❌ RunPod could not recreate Pod: no available instances (GPU capacity).")
                print(f"🧩 Request: gpu_count={payload.get('gpuCount')} gpu_type_ids={payload.get('gpuTypeIds')}")
                print("💡 Try later or widen the GPU type list.")
                return 1
            raise
        pod_id = str(created.get("id") or "").strip()
        if not pod_id:
            raise RuntimeError(f"Pod re-created but id missing: {created}")

        print(f"✅ New pod created: {pod_id}")
        print("▶️  Starting new Pod...")
        new_start_failed_msg = ""
        try:
            _runpod_start_pod(rest_api_base_url=rest_base, api_key=api_key, pod_id=pod_id)
        except Exception as e:
            new_start_failed_msg = str(e)
            print(f"⚠️  Start request failed: {e}")
        try:
            if new_start_failed_msg and "not enough free gpus" in new_start_failed_msg.lower():
                raise RuntimeError(f"Fast-failing wait because new host machine also has no free GPUs.")

            public_ip, ssh_port = _wait_for_pod_ssh_ready(
                rest_api_base_url=rest_base,
                api_key=api_key,
                pod_id=pod_id,
                timeout_seconds=10 * 60,
            )
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user. Stopping Pod...", flush=True)
            try:
                _runpod_stop_pod(rest_api_base_url=rest_base, api_key=api_key, pod_id=pod_id)
                print("✅ Pod stopped", flush=True)
            except Exception as e:
                print(f"⚠️  Stop error: {e}", flush=True)
            return 130
        except Exception:
            # Avoid leaking GPU costs on repeated failures.
            try:
                _runpod_stop_pod(rest_api_base_url=rest_base, api_key=api_key, pod_id=pod_id)
            except Exception:
                pass
            raise

    print(f"✅ Pod is running: ssh root@{public_ip}:{ssh_port}")

    # Expose runtime connection info for chat recovery (tunnel drops, vLLM restart, etc.).
    RUNTIME["ssh_host"] = public_ip
    RUNTIME["ssh_port"] = int(ssh_port)
    RUNTIME["ssh_key_path"] = key_path
    RUNTIME["serve_port"] = int(serve_port)
    RUNTIME["readiness_url"] = base_url.rstrip("/") + "/models"

    lock_dir = str(serve.get("lock_dir") or "/workspace/.helix_inference_lock")  # == POD_LOCK_DIR
    run_dir = str(serve.get("run_dir") or "/workspace/runs/run")  # == pod_run_dir(run_key)
    hf_cache_dir = str(serve.get("hf_cache_dir") or "/workspace/hf_cache")  # == POD_HF_CACHE_DIR
    merged_dir = str(serve.get("merged_model_dir") or f"{run_dir}/model")  # == pod_merged_dir(run_key)
    pid_file = str(serve.get("vllm_pid_file") or f"{run_dir}/vllm.pid")  # == pod_pid_file(run_key)
    log_file = str(serve.get("vllm_log_file") or f"{run_dir}/vllm.log")  # == pod_log_file(run_key)
    hash_file = str(serve.get("config_hash_file") or f"{run_dir}/config_hash.txt")  # == pod_hash_file(run_key)

    try:
        # Create tunnel first (so local health checks work)
        try:
            _ensure_ssh_tunnel(
                host=public_ip,
                port=ssh_port,
                key_path=key_path,
                local_port=serve_port,
                remote_port=serve_port,
            )
        except Exception as e:
            msg = str(e)
            ml = msg.lower()
            if "connection refused" in ml or "timed out" in ml or "no route to host" in ml:
                print(f"⚠️  SSH not ready yet: {msg}")
                print("⏳ Waiting and retrying...", flush=True)
                public_ip, ssh_port = _wait_for_pod_ssh_ready(
                    rest_api_base_url=rest_base,
                    api_key=api_key,
                    pod_id=pod_id,
                    timeout_seconds=60,
                )
                _ensure_ssh_tunnel(
                    host=public_ip,
                    port=ssh_port,
                    key_path=key_path,
                    local_port=serve_port,
                    remote_port=serve_port,
                )
            else:
                print(f"❌ Could not create SSH tunnel: {msg}", flush=True)
                return 1

        # Acquire lock
        print("🔒 Acquiring volume lock...")
        lock_cmd = (
            f"set +e; "
            f"mkdir {shlex.quote(lock_dir)} 2>/dev/null; rc=$?; "
            f"if [ $rc -eq 0 ]; then echo ACQUIRED; exit 0; fi; "
            f"if test -f {shlex.quote(pid_file)}; then "
            f"  pid=$(cat {shlex.quote(pid_file)} 2>/dev/null || true); "
            f"  if test -n \"$pid\" && kill -0 \"$pid\" 2>/dev/null; then echo BUSY; exit 2; fi; "
            f"fi; "
            f"rm -rf {shlex.quote(lock_dir)}; "
            f"mkdir {shlex.quote(lock_dir)} 2>/dev/null; exit $?"
        )
        res_lock = _ssh_exec(host=public_ip, port=ssh_port, key_path=key_path, command=lock_cmd, timeout=30)
        if res_lock.returncode != 0:
            hint = (
                "Another session may be running (lock is held and vLLM is alive)."
                if res_lock.returncode == 2 or res_lock.stdout.strip() == "BUSY"
                else "Failed to acquire lock."
            )
            raise RuntimeError(f"{hint} stderr={res_lock.stderr.strip()[:200]} stdout={res_lock.stdout.strip()[:200]}")

        # Ensure dirs
        _ = _ssh_exec(
            host=public_ip,
            port=ssh_port,
            key_path=key_path,
            command=f"mkdir -p {shlex.quote(hf_cache_dir)} {shlex.quote(run_dir)}",
            timeout=30,
        )

        # Merge if needed
        merge_needed = True
        check_cmd = (
            f"test -f {shlex.quote(hash_file)} && "
            f"test -f {shlex.quote(merged_dir)}/config.json && "
            f"test \"$(cat {shlex.quote(hash_file)} 2>/dev/null || true)\" = {shlex.quote(config_hash)} "
            f"&& echo OK || echo NO"
        )
        res_chk = _ssh_exec(host=public_ip, port=ssh_port, key_path=key_path, command=check_cmd, timeout=30)
        if res_chk.returncode == 0 and res_chk.stdout.strip() == "OK":
            merge_needed = False

        if merge_needed:
            print("🧬 Running merge (LoRA → unified model)...")
            trust_arg = "--trust-remote-code" if trust_remote_code else ""
            hf_token = _require_env("HF_TOKEN")
            merge_cmd = (
                f"HF_TOKEN={shlex.quote(hf_token)} "
                f"python3 /opt/helix/merge_lora.py "
                f"--base-model {shlex.quote(base_model_id)} "
                f"--adapter {shlex.quote(adapter_ref)} "
                f"--output {shlex.quote(merged_dir)} "
                f"--cache-dir {shlex.quote(hf_cache_dir)} "
                f"{trust_arg}"
            ).strip()
            res_merge = _ssh_exec(host=public_ip, port=ssh_port, key_path=key_path, command=merge_cmd, timeout=3600)
            if res_merge.returncode != 0:
                raise RuntimeError(f"Merge failed. stderr={res_merge.stderr.strip()[:500]}")
            _ = _ssh_exec(
                host=public_ip,
                port=ssh_port,
                key_path=key_path,
                command=f"echo {shlex.quote(config_hash)} > {shlex.quote(hash_file)}",
                timeout=10,
            )
            print("✅ Merge completed")
        else:
            print("✅ Merge skipped (config_hash matches)")

        # Start (or restart) vLLM
        print("🚀 Starting vLLM...")
        tp = int(vllm_cfg.get("tensor_parallel_size") or 1)
        max_len = int(vllm_cfg.get("max_model_len") or 4096)
        gpu_mem = float(vllm_cfg.get("gpu_memory_utilization") or 0.90)
        quant = vllm_cfg.get("quantization")
        eager = bool(vllm_cfg.get("enforce_eager") or False)

        args = [
            "vllm serve",
            shlex.quote(merged_dir),
            "--host", "127.0.0.1",
            "--port", str(serve_port),
            "--tensor-parallel-size", str(tp),
            "--max-model-len", str(max_len),
            "--gpu-memory-utilization", str(gpu_mem),
        ]
        if quant:
            args += ["--quantization", shlex.quote(str(quant))]
        if eager:
            args += ["--enforce-eager"]
        if trust_remote_code:
            args += ["--trust-remote-code"]

        start_cmd = (
            f"set +e; "
            f"python3 -m pip install -q \"setuptools<70.0.0\"; "
            f"if test -f {shlex.quote(pid_file)}; then "
            f"pid=$(cat {shlex.quote(pid_file)} 2>/dev/null || true); "
            f"if test -n \"$pid\" && kill -0 \"$pid\" 2>/dev/null; then "
            f"kill \"$pid\" || true; sleep 2; kill -9 \"$pid\" 2>/dev/null || true; "
            f"fi; "
            f"fi; "
            f"rm -f {shlex.quote(pid_file)}; "
            f"nohup {' '.join(args)} > {shlex.quote(log_file)} 2>&1 & echo $! > {shlex.quote(pid_file)}; "
            f"set -e"
        )
        RUNTIME["start_vllm_cmd"] = start_cmd
        RUNTIME["vllm_log_file"] = log_file
        RUNTIME["vllm_pid_file"] = pid_file
        res_start = _ssh_exec(host=public_ip, port=ssh_port, key_path=key_path, command=start_cmd, timeout=30)
        if res_start.returncode != 0:
            raise RuntimeError(f"Failed to start vLLM. stderr={res_start.stderr.strip()[:500]}")

        # Wait for readiness via tunnel
        print("⏳ Waiting for /v1/models ...")
        try:
            _wait_http_ok(url=base_url.rstrip('/') + "/models", timeout_seconds=180, interval_seconds=2.0)
        except RuntimeError as e:
            print(f"\n❌ vLLM did not become ready within 180s: {e}", flush=True)
            print("📋 Last lines of vLLM log:", flush=True)
            try:
                tail_res = _ssh_exec(
                    host=public_ip,
                    port=ssh_port,
                    key_path=key_path,
                    command=f"tail -n 60 {shlex.quote(log_file)}",
                    timeout=20,
                )
                out = (tail_res.stdout or "").strip()
                if out:
                    print(out[-4000:], flush=True)
                else:
                    print("(log empty or unavailable)", flush=True)
            except Exception as log_err:
                print(f"(could not read log: {log_err})", flush=True)
            print(
                "\n💡 Possible causes: OOM (insufficient VRAM), CUDA error, "
                "wrong merged_model_dir, or vLLM crashed while loading the model. "
                "Check the log above.",
                flush=True,
            )
            return 1
        print("✅ vLLM is ready")

        # Resolve actual model name from /v1/models
        try:
            models_url = base_url.rstrip("/") + "/models"
            with urllib.request.urlopen(models_url, timeout=5) as response:
                models_data = json.loads(response.read().decode())
                if models_data.get("data"):
                    actual_model_name = models_data["data"][0]["id"]
                else:
                    actual_model_name = base_model_id
        except Exception:
            actual_model_name = base_model_id

        action = "exit"
        try:
            action = _start_chat(
                base_url,
                actual_model_name,
                base_model_id=base_model_id,
                rest_base=rest_base,
                api_key=api_key,
                pod_id=pod_id,
                system_prompt=m.get("llm", {}).get("system_prompt"),
            )
        except Exception:
            pass
        return 0

    finally:
        print("🧹 Cleanup: stopping vLLM and stopping Pod...", flush=True)
        # Best-effort vLLM stop + lock release
        try:
            stop_cmd = (
                f"set +e; "
                f"if test -f {shlex.quote(pid_file)}; then "
                f"pid=$(cat {shlex.quote(pid_file)} 2>/dev/null || true); "
                f"if test -n \"$pid\" && kill -0 \"$pid\" 2>/dev/null; then "
                f"kill \"$pid\" || true; sleep 2; kill -9 \"$pid\" 2>/dev/null || true; "
                f"fi; "
                f"fi; "
                f"rm -f {shlex.quote(pid_file)}; "
                f"rm -rf {shlex.quote(lock_dir)} 2>/dev/null || true; "
                f"set -e"
            )
            _ = _ssh_exec(host=public_ip, port=ssh_port, key_path=key_path, command=stop_cmd, timeout=30)
        except Exception:
            pass

        try:
            if "action" in locals() and action in ("stop", "clear"):
                if action == "clear":
                    print("🧹 Clearing volume contents (/workspace/runs, /workspace/hf_cache)...", flush=True)
                    clear_cmd = f"rm -rf {shlex.quote(hf_cache_dir)} {shlex.quote(run_dir)}"
                    _ = _ssh_exec(host=public_ip, port=ssh_port, key_path=key_path, command=clear_cmd, timeout=120)

                print("🛑 Deleting pod...", flush=True)
                _runpod_delete_pod(rest_api_base_url=rest_base, api_key=api_key, pod_id=pod_id)
                print("✅ Pod deleted", flush=True)
            else:
                print("⏹️  Stopping Pod (to save GPU costs)...", flush=True)
                _runpod_stop_pod(rest_api_base_url=rest_base, api_key=api_key, pod_id=pod_id)
                print("✅ Pod stopped", flush=True)
        except Exception as e:
            print(f"⚠️  Cleanup error: {e}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
"""


__all__ = [
    "CHAT_SCRIPT",
    "render_readme",
]
