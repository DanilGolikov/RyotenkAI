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

# Start all services
./start.sh

# Check status
./start.sh status

# Follow logs
./start.sh logs

# Stop
./start.sh stop
```

After startup:
- **MLflow UI:** http://localhost:5002
- **MinIO Console:** http://localhost:9001 (login: `minio_admin` / `minio_secure_pass_2024`)

## Public Access via Tailscale

If you want a remote machine (for example, RunPod) to log to your local MLflow stack,
you can publish only the MLflow endpoint through Tailscale Funnel.

Prerequisites:
- Docker
- Tailscale installed locally
- `tailscale up` completed on your machine

```bash
cd docker/mlflow

# Start/rebuild local stack and publish MLflow over HTTPS
./expose-tailscale.sh up

# Inspect current public status
./expose-tailscale.sh status

# Print only the public URL
./expose-tailscale.sh url

# Disable public access
./expose-tailscale.sh down
```

The script:
- starts or rebuilds the local MLflow stack
- enables `tailscale funnel` only for MLflow
- configures `MLflow` `allowed-hosts` for the generated `*.ts.net` hostname
- leaves MinIO private on your machine
- asks for confirmation before changing local state
- falls back to a rootless local `tailscaled` when the system daemon is unavailable
- waits until the public HTTPS endpoint is actually reachable before printing the final URL

Re-running `./expose-tailscale.sh up` is safe:
- it reuses the existing stack when possible
- it refreshes the Funnel configuration
- it does not force a Docker rebuild on every run

Use the printed URL as your remote tracking URI:

```bash
export MLFLOW_TRACKING_URI=https://your-machine.your-tailnet.ts.net:8443
```

Security notes:
- Prefer enabling MLflow auth before using this outside your private machine.
- Only `MLflow` is exposed publicly; `MinIO` is not.
- Default Funnel port is `443`. Override with `TAILSCALE_FUNNEL_HTTPS_PORT=8443` if needed.
- Rootless Tailscale runtime files are stored in `~/.local/state/ryotenkai-mlflow-tailscale` by default.

## Configuration

Edit `.env.mlflow` to customize credentials and ports:

```env
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=mlflow_secure_pass_2024
POSTGRES_DB=mlflow_db
POSTGRES_PORT=5432

MINIO_ROOT_USER=minio_admin
MINIO_ROOT_PASSWORD=minio_secure_pass_2024
MINIO_BUCKET=mlflow
MINIO_API_PORT=9000
MINIO_CONSOLE_PORT=9001

MLFLOW_PORT=5002

# Optional public exposure / security settings
# MLFLOW_SERVER_ALLOWED_HOSTS=my-node.example.ts.net,localhost:
# MLFLOW_SERVER_CORS_ALLOWED_ORIGINS=https://my-ui.example.com
# MLFLOW_APP_NAME=basic-auth
# MLFLOW_FLASK_SERVER_SECRET_KEY=CHANGE_ME
```

## Pipeline Integration

The training pipeline connects to MLflow via the `tracking` section in your pipeline config:

```yaml
tracking:
  mlflow:
    tracking_uri: "http://localhost:5002"
    experiment_name: "ryotenkai"
```

The pipeline logs metrics, parameters, and artifacts automatically during training.

## File Structure

```
docker/mlflow/
├── docker-compose.mlflow.yml  # Service definitions
├── Dockerfile.mlflow           # MLflow server image (+ psycopg2, boto3)
├── .dockerignore
├── .env.mlflow                 # Environment configuration
├── entrypoint.mlflow.sh        # MLflow startup with optional security flags
├── expose-tailscale.sh         # Public HTTPS access via Tailscale Funnel
├── start.sh                    # Startup/management script
└── README.md

# Note: an init container (minio/mc) auto-creates the MLflow
# bucket on first startup. See docker-compose.mlflow.yml.
```

## Data Persistence

Data is stored in Docker named volumes:
- `postgres_data` — MLflow metadata
- `minio_data` — Artifact files

To reset everything:

```bash
./start.sh stop
docker volume rm ryotenkai_postgres_data ryotenkai_minio_data
```

## Troubleshooting

### Port conflict

Change ports in `.env.mlflow` if defaults are in use:

```env
MLFLOW_PORT=5003
POSTGRES_PORT=5433
MINIO_API_PORT=9002
MINIO_CONSOLE_PORT=9003
```

### MLflow fails to start

Check that PostgreSQL and MinIO are healthy first:

```bash
./start.sh status
docker logs mlflow_postgres
docker logs mlflow_minio
```
