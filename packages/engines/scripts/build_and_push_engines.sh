#!/usr/bin/env bash
# ==============================================================================
# Build & Push RyotenkAI Engine Images
# ==============================================================================
#
# Walks every engine plugin under
#   packages/engines/src/ryotenkai_engines/*/
# and, for each folder that ships both an ``engine.toml`` and a
# ``Dockerfile``, builds the image with the convention name:
#
#   {prefix}/inference-{engine.id}:{engine.version}
#
# where ``prefix`` defaults to ``ryotenkai`` (override via ``--username``
# or ``DOCKER_USERNAME``), and ``engine.id`` / ``engine.version`` come
# straight from the manifest. This mirrors the runtime resolution chain
# in ``ryotenkai_engines.images.resolve_image`` so what you build is
# what the providers pull.
#
# Adding a new engine ⇒ drop a folder with ``engine.toml`` + ``Dockerfile``
# and re-run this script. No edits here.
#
# Usage:
#   ./packages/engines/scripts/build_and_push_engines.sh [OPTIONS]
#
# Options:
#   --username <name>    Image namespace prefix (default: ``ryotenkai`` /
#                        ``DOCKER_USERNAME`` env)
#   --engine <id>        Restrict to a single engine id
#   --no-push            Build only, don't push to the registry
#   --skip-login         Skip ``docker login`` (assume already logged in)
#   --help               Show this help message
#
# Environment Variables:
#   DOCKER_USERNAME      Default namespace prefix when ``--username`` not given
#   DOCKER_PASSWORD      Optional password / token for non-interactive login
#
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

DOCKER_USERNAME="${DOCKER_USERNAME:-ryotenkai}"
ENGINE_FILTER=""
PUSH_IMAGE=true
SKIP_LOGIN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --username)   DOCKER_USERNAME="$2"; shift 2 ;;
        --engine)     ENGINE_FILTER="$2"; shift 2 ;;
        --no-push)    PUSH_IMAGE=false; shift ;;
        --skip-login) SKIP_LOGIN=true; shift ;;
        --help)       head -n 38 "$0" | grep "^#" | sed 's/^# //'; exit 0 ;;
        *)            echo -e "${RED}Error: Unknown option $1${NC}"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENGINES_DIR="$PROJECT_ROOT/packages/engines/src/ryotenkai_engines"

# Tiny TOML extractor — pulls a single ``key = "value"`` from the
# ``[engine]`` block. Keeps the script dependency-free (no ``yq`` /
# ``tomlq``). Manifest uses double-quoted string scalars so this is
# safe for ``id`` / ``version``; cross-checked by the python drift
# detector at packages/engines/scripts/check_engine_manifests.py.
_toml_get_engine_field() {
    local file="$1" key="$2"
    # Pull a string scalar out of the [engine] block. Tolerates trailing
    # ``# inline comments``. Quotes are mandatory in the manifest, so
    # we strip exactly the opening ``"`` and slice up to the closing ``"``.
    awk -v key="$key" '
        /^\[engine\]/ { in_block=1; next }
        /^\[/        { in_block=0 }
        in_block && $0 ~ "^"key"[[:space:]]*=" {
            sub("^"key"[[:space:]]*=[[:space:]]*\"", "")
            n = index($0, "\"")
            if (n > 0) print substr($0, 1, n - 1)
            exit
        }
    ' "$file"
}

if [[ "$PUSH_IMAGE" == "true" ]] && [[ "$SKIP_LOGIN" == "false" ]]; then
    echo "Logging in to Docker Hub as ${DOCKER_USERNAME}..."
    if [[ -n "${DOCKER_PASSWORD:-}" ]]; then
        echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin
    else
        docker login -u "$DOCKER_USERNAME"
    fi
    echo ""
fi

found_any=false
build_failures=()

for manifest in "$ENGINES_DIR"/*/engine.toml; do
    [[ -f "$manifest" ]] || continue
    engine_dir="$(dirname "$manifest")"
    engine_id=$(_toml_get_engine_field "$manifest" "id")
    engine_version=$(_toml_get_engine_field "$manifest" "version")
    dockerfile="$engine_dir/Dockerfile"

    if [[ -z "$engine_id" || -z "$engine_version" ]]; then
        echo -e "${YELLOW}skip:${NC} $manifest — missing [engine].id or [engine].version"
        continue
    fi
    if [[ ! -f "$dockerfile" ]]; then
        echo -e "${YELLOW}skip:${NC} ${engine_id} — no Dockerfile sibling, image build delegated"
        continue
    fi
    if [[ -n "$ENGINE_FILTER" && "$engine_id" != "$ENGINE_FILTER" ]]; then
        continue
    fi

    found_any=true
    image="${DOCKER_USERNAME}/inference-${engine_id}:${engine_version}"

    echo "============================================================"
    echo -e "${GREEN}engine: ${engine_id} v${engine_version}${NC}"
    echo "  context: ${engine_dir#$PROJECT_ROOT/}"
    echo "  image:   ${image}"
    echo "============================================================"

    set +e
    docker build \
        --platform linux/amd64 \
        --build-arg "IMAGE_VERSION=${engine_version}" \
        -f "$dockerfile" \
        -t "$image" \
        -t "${DOCKER_USERNAME}/inference-${engine_id}:latest" \
        "$engine_dir"
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
        echo -e "${RED}build failed: ${engine_id}${NC}"
        build_failures+=("$engine_id")
        continue
    fi
    echo -e "${GREEN}built: ${image}${NC}"

    if [[ "$PUSH_IMAGE" == "true" ]]; then
        docker push "$image"
        docker push "${DOCKER_USERNAME}/inference-${engine_id}:latest"
        echo -e "${GREEN}pushed: ${image}${NC}"
    fi
    echo ""
done

if [[ "$found_any" != "true" ]]; then
    if [[ -n "$ENGINE_FILTER" ]]; then
        echo -e "${RED}No engine matched --engine ${ENGINE_FILTER}${NC}"
    else
        echo -e "${RED}No engines found under ${ENGINES_DIR}${NC}"
    fi
    exit 1
fi

if [[ ${#build_failures[@]} -gt 0 ]]; then
    echo -e "${RED}Some engines failed to build:${NC} ${build_failures[*]}"
    exit 1
fi

echo -e "${GREEN}Done${NC}"
