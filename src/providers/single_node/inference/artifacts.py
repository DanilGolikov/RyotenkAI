from __future__ import annotations

from src.constants import INFERENCE_DIRNAME


def render_readme(*, manifest_filename: str, endpoint_url: str) -> str:
    return f"""## Inference endpoint (MVP)

- **Manifest**: `{manifest_filename}`
- **Endpoint (via SSH tunnel)**: `{endpoint_url}`

### Quick commands

Use Python from the project venv:

```bash
python {INFERENCE_DIRNAME}/chat_inference.py
```

Or if the venv is activated:

1) **Interactive chat** (checks status automatically):

```bash
python {INFERENCE_DIRNAME}/chat_inference.py
```

2) **Stop inference** (removes the container):

```bash
python {INFERENCE_DIRNAME}/stop_inference.py
```

### SSH tunnel

See `tunnel_hint` in `inference_manifest.json`.

### Troubleshooting

If inference is not running, the chat script shows status and instructions.
To redeploy, run the full pipeline with `inference.enabled=true`.
"""


CHAT_SCRIPT = r"""#!/usr/bin/env python3
'''
Interactive chat with inference endpoint.
Checks status first, then starts OpenAI-compatible chat session.
'''
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def _load_manifest() -> dict:
    here = Path(__file__).resolve().parent
    p = here / "inference_manifest.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _build_ssh_args(ssh_cfg: dict) -> list[str]:
    args = ["ssh", "-o", "StrictHostKeyChecking=no"]
    alias = ssh_cfg.get("alias")
    if isinstance(alias, str) and alias:
        args.append(alias)
        return args

    host = ssh_cfg.get("host")
    user = ssh_cfg.get("user")
    port = int(ssh_cfg.get("port") or 22)
    key_path = ssh_cfg.get("key_path")
    key_env = ssh_cfg.get("key_env")
    if not key_path and key_env:
        key_path = os.environ.get(key_env)

    if key_path:
        args += ["-i", str(key_path)]

    args += ["-p", str(port)]
    args.append(f"{user}@{host}")
    return args


def _check_status_remote(ssh_args: list[str], container_name: str, host: str, port: int) -> tuple[bool, bool]:
    '''Check if container is running and healthy. Returns (is_running, is_healthy).'''
    # Check container running
    check_container = f"docker ps -q -f name={container_name} -f status=running"
    res_container = subprocess.run([*ssh_args, check_container], capture_output=True, text=True)
    is_running = bool(res_container.stdout.strip())

    if not is_running:
        return False, False

    # Check health endpoint
    check_health = f"curl -s -f -m 5 http://{host}:{port}/v1/models >/dev/null 2>&1 && echo 1 || echo 0"
    res_health = subprocess.run([*ssh_args, check_health], capture_output=True, text=True)
    is_healthy = res_health.stdout.strip() == "1"

    return is_running, is_healthy


def _ensure_ssh_tunnel(ssh_args: list[str], local_port: int, remote_host: str, remote_port: int) -> None:
    '''Ensure SSH tunnel is active for remote access.'''
    import time

    # Check if tunnel already exists
    tunnel_pattern = f"{local_port}:{remote_host}:{remote_port}"
    ps_check = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    if tunnel_pattern in ps_check.stdout:
        print(f"✅ SSH tunnel already active: localhost:{local_port} → {remote_host}:{remote_port}")
        return

    # Create tunnel
    print(f"🔗 Creating SSH tunnel: localhost:{local_port} → {remote_host}:{remote_port}")
    tunnel_cmd = [
        ssh_args[0],  # 'ssh'
        "-f", "-N",  # background, no command
        "-o", "StrictHostKeyChecking=no",
        "-L", f"{local_port}:{remote_host}:{remote_port}",
        *ssh_args[1:],  # rest of args
    ]

    try:
        subprocess.run(tunnel_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        time.sleep(1)  # Wait for tunnel to establish
        print(f"✅ SSH tunnel created")
    except subprocess.CalledProcessError as e:
        output = e.output.decode() if e.output else ""
        if "address already in use" in output.lower():
            print(f"✅ Tunnel port already in use (likely active)")
        else:
            print(f"⚠️  Tunnel creation failed: {output}")

def _load_chat_template(*, base_model_id: str) -> str | None:
    '''
    vLLM needs chat_template for /v1/chat/completions if the served model tokenizer
    (local merged-dir) does not include it.

    We take chat_template from the base HF model (same as used during merge).
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
    print("  - /status → show status")
    print("  - /exit   → leave chat (container keeps running)")
    print("  - /stop   → stop and remove container")
    print("  - /clear  → remove container and clear HF cache / model dirs")
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
    ssh_args: list[str],
    container_name: str,
    host: str,
    port: int,
    system_prompt: str | None = None,
) -> str:
    '''Start interactive chat session using OpenAI-compatible Chat Completions.'''
    import urllib.request
    import json

    print(f"\n💬 Chat with {model_name.split('/')[-1]}")
    print("Type 'exit' or 'quit' to end session")
    print("For multi-line prompts: type /paste, paste text, then /end")
    print("Hint: /help\n")

    # IMPORTANT:
    # - Use /v1/chat/completions (NOT /v1/completions) to leverage the model's chat template
    # - System prompt is resolved at deploy time and stored as text in inference_manifest.json.
    #   This script receives it ready-to-use — no file access or MLflow needed here.
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

    while True:
        try:
            raw = _input_with_timeout("You: ", 300.0)
            if raw is None:
                print("\n⏳ No activity for 5 minutes. Auto-exiting to save cost...")
                return "exit"
            cmd = raw.strip()

            if cmd.lower() in ("exit", "quit", "q", "/exit"):
                print("👋 Goodbye!")
                return "exit"

            if cmd == "/stop":
                print("🛑 Goodbye! Stopping container...")
                return "stop"

            if cmd == "/clear":
                print("🧹 Goodbye! Clearing volume and removing container...")
                return "clear"

            if cmd == "/status":
                is_running, is_healthy = _check_status_remote(ssh_args, container_name, host, port)
                st = "RUNNING" if is_running else "STOPPED"
                hlt = "HEALTHY" if is_healthy else "UNHEALTHY"
                print(f"📊 Container: {container_name} | Status: {st} | Health: {hlt}")
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
                # Single-line input (multi-line paste will be split by terminal → use /paste).
                user_input = raw

            if not user_input.strip():
                continue

            # Append to chat history
            messages.append({"role": "user", "content": user_input})

            # Build request
            payload = {
                "model": model_name,
                "messages": messages,
                "stream": True,
                "temperature": temperature,
                "max_tokens": max_tokens,
                # Safe stop sequences to avoid markdown / long explanations.
                "stop": ["```", "\n###", "\n\n###"],
            }
            if chat_template:
                payload["chat_template"] = chat_template

            req = urllib.request.Request(
                chat_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )

            print("Assistant: ", end="", flush=True)
            full_response = ""

            # Stream response (SSE format)
            with urllib.request.urlopen(req, timeout=60) as response:
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
                        chunk_data = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk_data.get("choices", []) if isinstance(chunk_data, dict) else []
                    if not choices:
                        continue
                    choice0 = choices[0] if isinstance(choices[0], dict) else {}
                    delta = choice0.get("delta", {}) if isinstance(choice0, dict) else {}
                    text = delta.get("content") if isinstance(delta, dict) else None
                    if isinstance(text, str) and text:
                        print(text, end="", flush=True)
                        full_response += text

                    # Some gateways may omit the final [DONE] sentinel.
                    finish_reason = choice0.get("finish_reason") if isinstance(choice0, dict) else None
                    if finish_reason is not None:
                        break

            print()  # Newline after response
            assistant_text = full_response.strip()
            messages.append({"role": "assistant", "content": assistant_text})

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            return "exit"
        except EOFError:
            print("\n\n👋 Goodbye!")
            return "exit"
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return "exit"

    return "exit"


def main() -> int:
    m = _load_manifest()
    ssh_args = _build_ssh_args(m["ssh"])
    container_name = m["docker"]["container_name"]
    host = m["docker"]["host_bind"]
    port = int(m["docker"]["port"])
    base_url = m["endpoint"]["client_base_url"]
    display_name = m["model"]["base_model_id"]

    print(f"🔍 Checking inference status...")
    is_running, is_healthy = _check_status_remote(ssh_args, container_name, host, port)

    if not is_running:
        print(f"❌ Inference is NOT running (container: {container_name})")
        print(f"\n💡 To deploy inference, run the full pipeline with inference.enabled=true")
        return 1

    if not is_healthy:
        print(f"⚠️  Container is running but service is NOT ready")
        print(f"   Check logs: docker logs {container_name}")
        return 1

    print(f"✅ Inference is running and healthy")
    print(f"📡 Endpoint: {base_url}")
    print(f"🤖 Model: {display_name}")

    # Ensure SSH tunnel (if using SSH alias)
    alias = m["ssh"].get("alias")
    if alias:
        _ensure_ssh_tunnel(ssh_args, port, host, port)

    # Get actual model name from vLLM API
    try:
        import urllib.request
        models_url = base_url.replace("/v1", "/v1/models")
        with urllib.request.urlopen(models_url, timeout=5) as response:
            import json
            models_data = json.loads(response.read().decode())
            if models_data.get("data"):
                actual_model_name = models_data["data"][0]["id"]
            else:
                actual_model_name = display_name
    except Exception:
        # Fallback to display name if API call fails
        actual_model_name = display_name

        action = "exit"
        try:
            action = _start_chat(
                base_url,
                actual_model_name,
                base_model_id=display_name,
                ssh_args=ssh_args,
                container_name=container_name,
                host=host,
                port=port,
                system_prompt=m.get("llm", {}).get("system_prompt"),
            )
        except Exception:
            pass

    if action in ("stop", "clear"):
        print(f"🛑 Stopping and removing container: {container_name}")
        remote_cmd = f"docker rm -f {container_name}"
        res = subprocess.run([*ssh_args, remote_cmd], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"✅ Container removed: {container_name}")
        else:
            if "No such container" not in res.stderr and "Error: No such container" not in res.stderr:
                print(f"❌ Failed to remove container: {res.stderr}")

        if action == "clear":
            workspace = m["docker"]["workspace"]
            print(f"🧹 Clearing volume contents (/workspace/runs, /workspace/hf_cache)...")
            # We don't want to blindly rm -rf / if workspace is misconfigured, so we carefully remove specific dirs
            clear_cmd = f"rm -rf {workspace}/hf_cache {workspace}/runs"
            subprocess.run([*ssh_args, clear_cmd], capture_output=True, text=True)
            print("✅ Workspace cleared")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


__all__ = [
    "CHAT_SCRIPT",
    "render_readme",
]
