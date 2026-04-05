# MLflow Tracking Stack

Local MLflow server for RyotenkAI with PostgreSQL (metadata) and MinIO (S3-compatible artifact storage).

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   MLflow     │────▶│  PostgreSQL  │     │    MinIO     │
│   Server     │────▶│  (metadata)  │     │  (artifacts) │
│  :5002       │     │  :5432       │     │  :9000/:9001 │
└─────────────┘     └─────────────┘     └─────────────┘
```

- **MLflow Server** — tracking UI + artifact proxy (`http://localhost:5002`)
- **PostgreSQL 16** — stores runs, params, metrics
- **MinIO** — S3-compatible storage for model artifacts, plots, logs

## Quick Start

```bash
cd docker/mlflow

# Start all services (creates .env.mlflow from template on first run)
./start.sh

# Check status
./start.sh status

# Follow logs
./start.sh logs

# Stop
./start.sh stop
```

The stack works immediately with dev defaults from `.env.mlflow` — no manual configuration needed.

After startup:
- **MLflow UI:** http://localhost:5002
- **MinIO Console:** http://localhost:9001 (login: `minio_admin` / `minio_dev_pass`)

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

MLFLOW_PORT=5002
```

> **Note:** Change passwords before any non-local usage.

## File Structure

```
docker/mlflow/
├── docker-compose.mlflow.yml  # Service definitions
├── Dockerfile.mlflow           # MLflow image (+ psycopg2, boto3)
├── .dockerignore
├── .env.mlflow                 # Environment config (dev defaults, works out of the box)
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
