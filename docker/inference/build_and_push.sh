#!/usr/bin/env bash
# ==============================================================================
# Build and Push Inference Docker Image
# ==============================================================================
#
# Purpose:
#   Build ryotenkai/inference-vllm image and push to Docker Hub.
#   Automatically increments version tag (patch/minor/major).
#
# Usage:
#   ./build_and_push.sh [OPTIONS]
#
# Options:
#   --username <name>    Docker Hub username/org (default: from DOCKER_USERNAME env)
#   --version <ver>      Image version tag (overrides auto-increment)
#   --bump <type>        Auto-increment: patch, minor, or major (default: patch)
#   --no-push            Build only, don't push to Docker Hub
#   --skip-login         Skip Docker Hub login (if already logged in)
#   --help               Show this help message
#
# Environment Variables:
#   DOCKER_USERNAME      Docker Hub username/org (required if --username not given)
#   DOCKER_PASSWORD      Docker Hub password/token (for login)
#
# Examples:
#   ./build_and_push.sh --username ryotenkai --bump patch
#   ./build_and_push.sh --username ryotenkai --version v1.2.0
#   ./build_and_push.sh --username ryotenkai --no-push
#
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

DOCKER_USERNAME="${DOCKER_USERNAME:-}"
VERSION=""
BUMP_TYPE="patch"
PUSH_IMAGE=true
SKIP_LOGIN=false
IMAGE_NAME="inference-vllm"

while [[ $# -gt 0 ]]; do
    case $1 in
        --username)   DOCKER_USERNAME="$2"; shift 2 ;;
        --version)    VERSION="$2"; shift 2 ;;
        --bump)
            BUMP_TYPE="$2"
            if [[ ! "$BUMP_TYPE" =~ ^(patch|minor|major)$ ]]; then
                echo -e "${RED}Error: --bump must be one of: patch, minor, major${NC}"
                exit 1
            fi
            shift 2 ;;
        --no-push)    PUSH_IMAGE=false; shift ;;
        --skip-login) SKIP_LOGIN=true; shift ;;
        --help)       head -n 35 "$0" | grep "^#" | sed 's/^# //'; exit 0 ;;
        *)            echo -e "${RED}Error: Unknown option $1${NC}"; exit 1 ;;
    esac
done

if [[ -z "$DOCKER_USERNAME" ]]; then
    echo -e "${RED}Error: Docker Hub username not provided${NC}"
    echo "Set DOCKER_USERNAME env or use --username"
    exit 1
fi

cd "$(dirname "$0")" || exit 1

FULL_IMAGE="${DOCKER_USERNAME}/${IMAGE_NAME}"

# --- Version resolution ---

_get_latest_remote_version() {
    local api_url="https://hub.docker.com/v2/repositories/${DOCKER_USERNAME}/${IMAGE_NAME}/tags?page_size=100&ordering=-last_updated"
    if command -v curl >/dev/null 2>&1; then
        curl -s "$api_url" 2>/dev/null | \
            grep -oE '"name":"v[0-9]+\.[0-9]+\.[0-9]+"' | \
            sed 's/"name":"//g; s/"//g' | sort -V | tail -1 || echo ""
    else
        echo ""
    fi
}

_get_latest_local_version() {
    docker images "${FULL_IMAGE}" --format "{{.Tag}}" 2>/dev/null | \
        grep -E "^v[0-9]+\.[0-9]+\.[0-9]+$" | sort -V | tail -1 || echo ""
}

_increment_version() {
    local ver="$1" bump="$2"
    if [[ "$ver" =~ ^v([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
        local major="${BASH_REMATCH[1]}" minor="${BASH_REMATCH[2]}" patch="${BASH_REMATCH[3]}"
        case "$bump" in
            patch) patch=$((patch + 1)) ;;
            minor) minor=$((minor + 1)); patch=0 ;;
            major) major=$((major + 1)); minor=0; patch=0 ;;
        esac
        echo "v${major}.${minor}.${patch}"
    else
        echo "v1.0.0"
    fi
}

if [[ -z "$VERSION" ]]; then
    echo "Resolving latest version..."
    local_ver=$(_get_latest_local_version)
    remote_ver=$(_get_latest_remote_version)
    latest=$(printf "%s\n%s" "$local_ver" "$remote_ver" | grep -v '^$' | sort -V | tail -1 || echo "")

    if [[ -z "$latest" ]]; then
        VERSION="v1.0.0"
        echo -e "${YELLOW}No existing version found, starting at ${VERSION}${NC}"
    else
        VERSION=$(_increment_version "$latest" "$BUMP_TYPE")
        echo -e "${GREEN}${latest} -> ${VERSION} (${BUMP_TYPE})${NC}"
    fi
fi

# --- Build ---

echo ""
echo "============================================================"
echo -e "${GREEN}RyotenkAI Inference (vLLM) — Build & Push${NC}"
echo "============================================================"
echo "Image:    ${FULL_IMAGE}:${VERSION}"
echo "Push:     ${PUSH_IMAGE}"
echo ""

if [[ "$PUSH_IMAGE" == "true" ]] && [[ "$SKIP_LOGIN" == "false" ]]; then
    echo "Logging in to Docker Hub..."
    if [[ -n "${DOCKER_PASSWORD:-}" ]]; then
        echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin
    else
        docker login -u "$DOCKER_USERNAME"
    fi
    echo ""
fi

echo "Building ${FULL_IMAGE}:${VERSION}..."
docker build \
    --platform linux/amd64 \
    -f Dockerfile \
    -t "${FULL_IMAGE}:${VERSION}" \
    -t "${FULL_IMAGE}:latest" \
    .

echo -e "${GREEN}Build complete${NC}"
echo ""

# --- Push ---

if [[ "$PUSH_IMAGE" == "true" ]]; then
    echo "Pushing ${FULL_IMAGE}:${VERSION}..."
    docker push "${FULL_IMAGE}:${VERSION}"
    docker push "${FULL_IMAGE}:latest"
    echo -e "${GREEN}Push complete${NC}"
    echo ""

    echo "Cleaning up local images..."
    docker rmi "${FULL_IMAGE}:${VERSION}" "${FULL_IMAGE}:latest" 2>/dev/null || true
fi

# --- Summary ---

echo ""
echo "============================================================"
echo -e "${GREEN}Done${NC}"
echo "============================================================"
echo ""
echo "Update your pipeline config:"
echo "  inference:"
echo "    engines:"
echo "      vllm:"
echo "        merge_image: \"${FULL_IMAGE}:${VERSION}\""
echo "        serve_image: \"${FULL_IMAGE}:${VERSION}\""
echo ""
echo "  # RunPod pods (if used):"
echo "  # providers.runpod.inference.pod.image_name: \"${FULL_IMAGE}:${VERSION}\""
echo ""
