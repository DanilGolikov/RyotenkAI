# MLflow Tracking Stack

Local MLflow server for RyotenkAI with PostgreSQL (metadata), MinIO
(S3-compatible artifact storage), and a Caddy reverse proxy enforcing HTTP
basic-auth in front of the MLflow UI/API.

## Architecture

```
                    ┌──────────────────┐
   public/local ───▶│  Caddy (:5002)   │── basic-auth ──┐
                    └──────────────────┘                 │
                                                         ▼
                    ┌──────────────────┐     ┌────────────────┐     ┌─────────────┐
                    │  MLflow Server   │────▶│  PostgreSQL    │     │   MinIO     │
                    │  (internal :5102)│────▶│  (metadata)    │     │ (artifacts) │
                    └──────────────────┘     └────────────────┘     └─────────────┘
                                                                            ▲
                                                                            │
                              (--serve-artifacts proxies all artifact I/O)──┘
```

- **Caddy** — reverse proxy, the ONLY external entry point. Listens on
  `:5002`, enforces HTTP basic-auth, exposes `/health` unauthenticated for
  Tailscale Funnel probes.
- **MLflow Server** — tracking UI + artifact proxy on internal `:5102`. Not
  published to the host; reachable only through Caddy via the docker
  network.
- **PostgreSQL 16** — stores runs, params, metrics.
- **MinIO** — S3-compatible storage for model artifacts, plots, logs. The
  bucket is **private**: anonymous read was removed in Phase M6, and all
  artifact I/O goes through MLflow's `--serve-artifacts` (which is
  itself behind basic-auth via Caddy).

## Quick Start

Before the first start you MUST configure Caddy basic-auth credentials —
the stack will refuse to come up otherwise.

```bash
cd docker/mlflow

# 1. Generate a bcrypt hash of the password you want for MLflow access.
#    Paste the output (starts with $2a$ or $2b$) somewhere safe.
docker run --rm caddy:2-alpine caddy hash-password --plaintext 'YOUR_PASSWORD'

# 2. Add to .env.mlflow (alongside the existing POSTGRES_* / MINIO_* vars):
#
#      CADDY_BASIC_AUTH_USER=mlflow_user
#      CADDY_BASIC_AUTH_HASH=$2a$14$........<paste-hash-here>........
#
#    Bcrypt hashes contain `$` characters — do NOT wrap the value in
#    double quotes (compose would interpolate it). Either leave it
#    unquoted or use single quotes.

# 3. Start all services
./start.sh

# Check status
./start.sh status

# Follow logs
./start.sh logs

# Stop
./start.sh stop
```

After startup:
- **MLflow UI:** http://localhost:5002 — prompts for the basic-auth
  credentials configured above.
- **MinIO Console:** http://localhost:9001 (login: whatever
  `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` you set).

### Why basic-auth?

The MLflow server itself ships with no authentication. Putting Caddy in
front of it is the cheapest way to keep the UI and tracking API behind a
shared secret when exposed via Tailscale Funnel. The `/health` endpoint
stays open so the Funnel probe can reach it.

### Why is MinIO private now?

Phase M6 removed `mc anonymous set download minio/${MINIO_BUCKET}` from
the bucket-init step. Previously anyone with the public URL of the bucket
could fetch any artifact. All artifact reads/writes now go through the
MLflow server's `--serve-artifacts` proxy, which is itself behind
basic-auth.

## Pipeline Configuration

### Local training (single machine)

```yaml
tracking:
  mlflow:
    tracking_uri: "http://localhost:5002"
    experiment_name: "my-experiment"
```

### Remote training (e.g. RunPod)

When the training node cannot reach `localhost`, expose MLflow publicly
(see next section) and use two URIs:

```yaml
tracking:
  mlflow:
    tracking_uri: "https://your-machine.your-tailnet.ts.net"
    local_tracking_uri: "http://localhost:5002"
    experiment_name: "my-experiment"
```

| Field | Used by | Purpose |
|---|---|---|
| `tracking_uri` | Remote training node | Public URL reachable from the internet |
| `local_tracking_uri` | Local orchestrator | Direct localhost access (faster, no TLS) |

If `local_tracking_uri` is omitted, the orchestrator also uses `tracking_uri`.

## Public Access via Tailscale

Publish MLflow to the internet so a remote GPU machine can log to it.

**Prerequisites:** Docker, [Tailscale](https://tailscale.com/download) installed and authenticated (`tailscale up`).

```bash
cd docker/mlflow

# Start stack + expose via Tailscale Funnel (one command)
./expose-tailscale.sh up

# Check public status
./expose-tailscale.sh status

# Print only the public URL
./expose-tailscale.sh url

# Disable public access
./expose-tailscale.sh down
```

The script automatically:
- starts or reuses the local MLflow stack
- detects your Tailscale hostname
- injects `allowed-hosts` into the container at runtime (`.env.mlflow` is never modified)
- enables `tailscale funnel` for MLflow only (MinIO stays private)
- waits until the public HTTPS endpoint is reachable

Re-running `./expose-tailscale.sh up` is safe — it reuses the existing stack.

Default HTTPS port is `443`. Override with `TAILSCALE_FUNNEL_HTTPS_PORT=8443` if needed.

## Configuration

Edit `.env.mlflow` to customize credentials and ports:

```env
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=mlflow_dev_pass
POSTGRES_DB=mlflow_db
POSTGRES_PORT=5432

MINIO_ROOT_USER=minio_admin
MINIO_ROOT_PASSWORD=minio_dev_pass
MINIO_BUCKET=mlflow
MINIO_API_PORT=9000
MINIO_CONSOLE_PORT=9001

# External port the Caddy reverse proxy listens on. MLflow itself is
# internal-only on :5102 inside the docker network.
MLFLOW_PORT=5002

# Required: Caddy basic-auth credentials. Generate the hash with:
#   docker run --rm caddy:2-alpine caddy hash-password --plaintext '...'
CADDY_BASIC_AUTH_USER=mlflow_user
CADDY_BASIC_AUTH_HASH=$2a$14$replace_with_real_bcrypt_hash

# Required when exposing publicly (allowed-hosts non-empty). Generate with:
#   python -c "import secrets; print(secrets.token_urlsafe(48))"
MLFLOW_FLASK_SERVER_SECRET_KEY=replace_with_random_secret
```

> **Note:** Change passwords before any non-local usage. The bcrypt hash
> contains `$` characters — do NOT double-quote the value in `.env.mlflow`.

### Generating `MLFLOW_FLASK_SERVER_SECRET_KEY`

The Flask session secret signs MLflow's session cookies and CSRF tokens.
Phase M6 made the entrypoint refuse to start when
`MLFLOW_SERVER_ALLOWED_HOSTS` is set but this secret is empty — an empty
secret would let an attacker forge sessions over the public endpoint.

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

## File Structure

```
docker/mlflow/
├── docker-compose.mlflow.yml  # Service definitions (postgres/minio/mlflow/caddy)
├── Caddyfile                   # Reverse-proxy config with basic-auth
├── Dockerfile.mlflow           # MLflow image (+ psycopg2, boto3)
├── .dockerignore
├── .env.mlflow                 # Environment config (NOT committed; secrets live here)
├── entrypoint.mlflow.sh        # MLflow startup with optional security flags
├── expose-tailscale.sh         # Public HTTPS access via Tailscale Funnel
├── start.sh                    # Startup/management script
└── README.md
```

## Data Persistence

Data is stored in Docker named volumes:
- `ryotenkai_postgres_data` — MLflow metadata
- `ryotenkai_minio_data` — Artifact files

To reset everything:

```bash
./start.sh stop
docker volume rm ryotenkai_postgres_data ryotenkai_minio_data
```

## Troubleshooting

### Port conflict

Change ports in `.env.mlflow`:

```env
MLFLOW_PORT=5003
POSTGRES_PORT=5433
MINIO_API_PORT=9002
MINIO_CONSOLE_PORT=9003
```

### MLflow fails to start

```bash
./start.sh status
docker logs mlflow_postgres
docker logs mlflow_minio
docker logs mlflow_server
```

### "Invalid Host header" on public URL

This means the container was started without the Tailscale hostname in `allowed-hosts`.
Run `./expose-tailscale.sh up` — it injects the hostname at runtime automatically.
If you restarted the stack with `start.sh restart`, run `expose-tailscale.sh up` again.

### Password mismatch after volume recreation

If you changed passwords in `.env.mlflow` but the Docker volume was created with old passwords, reset the volumes:

```bash
./start.sh stop
docker volume rm ryotenkai_postgres_data ryotenkai_minio_data
./start.sh
```
