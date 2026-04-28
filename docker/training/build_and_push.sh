#!/usr/bin/env bash
# ==============================================================================
# Build and Push Training Runtime Docker Image
# ==============================================================================
#
# Purpose:
#   Build ryotenkai-training-runtime image and (optionally) push to Docker Hub.
#   Tags are clean semver (vX.Y.Z); the CUDA / Python / Torch profile is
#   recorded in OCI labels and ``/opt/ryotenkai/version.txt`` instead of
#   the tag — so the pinned constant in ``src/runner/__about__.py`` stays
#   short and stable across dependency bumps.
#
# Inspect a published image without pulling its layers:
#
#   docker inspect ryotenkai/ryotenkai-training-runtime:v1.0.2 \
#     | jq '.[0].Config.Labels'
#
# Or, on a running container:
#
#   docker run --rm ryotenkai/...:v1.0.2 cat /opt/ryotenkai/version.txt
#
# Usage:
#   ./build_and_push.sh [OPTIONS]
#
# Options:
#   --username <name>    Docker Hub username (default: from DOCKER_USERNAME env)
#   --version <ver>      Image version tag (overrides auto-increment)
#   --bump <type>        Auto-increment version: patch, minor, or major (default: patch)
#   --no-push            Build only, don't push to Docker Hub
#   --skip-login         Skip Docker Hub login (if already logged in)
#   --show-latest        Show latest version info and exit (for copy-paste)
#   --help               Show this help message
#
# Environment Variables:
#   DOCKER_USERNAME      Docker Hub username (required if --username not provided)
#   DOCKER_PASSWORD      Docker Hub password/token (for login)
#
# Examples:
#   ./build_and_push.sh --username ryotenkai --bump patch  # v1.0.1 -> v1.0.2
#   ./build_and_push.sh --username ryotenkai --bump minor  # v1.0.1 -> v1.1.0
#   ./build_and_push.sh --username ryotenkai --bump major  # v1.0.1 -> v2.0.0
#   ./build_and_push.sh --username ryotenkai --version v1.2.0  # Use specific version
#
# ==============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default values
DOCKER_USERNAME="${DOCKER_USERNAME:-}"
VERSION=""  # Will be auto-generated if not provided
BUMP_TYPE="patch"  # patch, minor, or major
PUSH_IMAGES=true
SKIP_LOGIN=false
SHOW_LATEST=false

# ---------------------------------------------------------------------------
# Dependency profile — baked into the image as labels + version.txt. The
# image tag itself stays clean semver so the constant in
# ``src/runner/__about__.py`` doesn't dance every time we bump CUDA/Python.
# Bump these in lock-step with the FROM line in Dockerfile.runtime.
# ---------------------------------------------------------------------------
DEFAULT_CUDA_VERSION="12.4"
DEFAULT_PYTHON_VERSION="3.12"
DEFAULT_TORCH_VERSION="2.5.1"
CUDA_VERSION="${CUDA_VERSION:-$DEFAULT_CUDA_VERSION}"
PYTHON_VERSION="${PYTHON_VERSION:-$DEFAULT_PYTHON_VERSION}"
TORCH_VERSION="${TORCH_VERSION:-$DEFAULT_TORCH_VERSION}"

# Default fallback when no prior version exists locally OR on Hub.
DEFAULT_BOOTSTRAP_VERSION="v1.0.0"

# Regex matching the new clean-semver tags.
SEMVER_TAG_RE='^v[0-9]+\.[0-9]+\.[0-9]+$'

while [[ $# -gt 0 ]]; do
  case $1 in
    --username)
      DOCKER_USERNAME="$2"
      shift 2
      ;;
    --version)
      VERSION="$2"
      shift 2
      ;;
    --bump)
      BUMP_TYPE="$2"
      if [[ ! "$BUMP_TYPE" =~ ^(patch|minor|major)$ ]]; then
        echo -e "${RED}Error: --bump must be one of: patch, minor, major${NC}"
        exit 1
      fi
      shift 2
      ;;
    --no-push)
      PUSH_IMAGES=false
      shift
      ;;
    --skip-login)
      SKIP_LOGIN=true
      shift
      ;;
    --show-latest)
      SHOW_LATEST=true
      shift
      ;;
    --help)
      head -n 50 "$0" | grep "^#" | sed 's/^# //'
      exit 0
      ;;
    *)
      echo -e "${RED}Error: Unknown option $1${NC}"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

if [[ -z "$DOCKER_USERNAME" ]]; then
  echo -e "${RED}Error: Docker Hub username not provided${NC}"
  echo "Set DOCKER_USERNAME environment variable or use --username option"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

IMAGE="${DOCKER_USERNAME}/ryotenkai-training-runtime"

# ---------------------------------------------------------------------------
# Version discovery — pulls latest matching SEMVER_TAG_RE from local
# images and Docker Hub. Pre-refactor tags carried the dep profile inside
# (e.g. ``v1.0.1-cu124-py312``); those are now ignored when computing the
# next version so the new clean tag (``v1.0.2``) lands cleanly.
# ---------------------------------------------------------------------------

_get_latest_local_version() {
  docker images "${IMAGE}" --format "{{.Tag}}" \
    | grep -E "${SEMVER_TAG_RE}" \
    | sort -V | tail -1 || echo ""
}

_get_latest_remote_version() {
  local repo_name="${IMAGE#*/}"
  local username="${IMAGE%%/*}"
  local api_url="https://hub.docker.com/v2/repositories/${username}/${repo_name}/tags?page_size=100&ordering=-last_updated"

  if ! command -v curl >/dev/null 2>&1; then
    echo ""
    return
  fi

  local tags_json
  tags_json=$(curl -s "$api_url" 2>/dev/null || echo "")
  if [[ -z "$tags_json" ]]; then
    echo ""
    return
  fi

  echo "$tags_json" \
    | grep -oE '"name":"v[0-9]+\.[0-9]+\.[0-9]+"' \
    | sed -E 's/"name":"|"//g' \
    | sort -V | tail -1 || echo ""
}

_get_latest_version() {
  local local_version
  local remote_version
  local_version=$(_get_latest_local_version)
  remote_version=$(_get_latest_remote_version)

  if [[ -z "$local_version" ]] && [[ -z "$remote_version" ]]; then
    echo ""
  elif [[ -z "$local_version" ]]; then
    echo "$remote_version"
  elif [[ -z "$remote_version" ]]; then
    echo "$local_version"
  else
    printf "%s\n%s" "$local_version" "$remote_version" | sort -V | tail -1
  fi
}

_increment_version() {
  local current_version="$1"
  local bump_type="$2"

  if [[ ! "$current_version" =~ ^v([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
    echo "Error: Invalid version format: ${current_version}" >&2
    echo "Expected format: vX.Y.Z" >&2
    return 1
  fi

  local major="${BASH_REMATCH[1]}"
  local minor="${BASH_REMATCH[2]}"
  local patch="${BASH_REMATCH[3]}"

  case "$bump_type" in
    patch)
      patch=$((patch + 1))
      ;;
    minor)
      minor=$((minor + 1))
      patch=0
      ;;
    major)
      major=$((major + 1))
      minor=0
      patch=0
      ;;
  esac

  echo "v${major}.${minor}.${patch}"
}

# ---------------------------------------------------------------------------
# Show-latest short-circuit (no build, no push).
# ---------------------------------------------------------------------------
if [[ "$SHOW_LATEST" == "true" ]]; then
  LATEST_VERSION=$(_get_latest_version)
  if [[ -z "$LATEST_VERSION" ]]; then
    echo -e "${YELLOW}No semver-tagged version found (would bootstrap at ${DEFAULT_BOOTSTRAP_VERSION})${NC}"
    LATEST_VERSION="${DEFAULT_BOOTSTRAP_VERSION}"
  fi

  echo ""
  echo "Latest training image:"
  echo "  Image: ${IMAGE}"
  echo "  Tag:   ${LATEST_VERSION}"
  echo "  Full:  ${IMAGE}:${LATEST_VERSION}"
  echo ""
  echo "Inspect dependency profile of a built image:"
  echo "  docker inspect ${IMAGE}:${LATEST_VERSION} | jq '.[0].Config.Labels'"
  echo ""
  echo "Copy-paste:"
  echo "${IMAGE}:${LATEST_VERSION}"
  echo ""
  exit 0
fi

# Log file (overwrite on each run)
LOG_FILE="${SCRIPT_DIR}/build_and_push.log"
exec > >(tee "$LOG_FILE") 2>&1

# ---------------------------------------------------------------------------
# Resolve next version
# ---------------------------------------------------------------------------
if [[ -z "$VERSION" ]]; then
  echo "Checking for latest version (local + Docker Hub)..."
  LATEST_VERSION=$(_get_latest_version)

  if [[ -z "$LATEST_VERSION" ]]; then
    echo -e "${YELLOW}⚠️  No existing semver tag found (local or remote), bootstrapping at: ${DEFAULT_BOOTSTRAP_VERSION}${NC}"
    VERSION="${DEFAULT_BOOTSTRAP_VERSION}"
  else
    LOCAL_VER=$(_get_latest_local_version)
    REMOTE_VER=$(_get_latest_remote_version)

    if [[ -n "$LOCAL_VER" ]] && [[ -n "$REMOTE_VER" ]]; then
      echo -e "${GREEN}📌 Latest local: ${LOCAL_VER}, latest remote: ${REMOTE_VER}${NC}"
    elif [[ -n "$LOCAL_VER" ]]; then
      echo -e "${GREEN}📌 Latest local: ${LOCAL_VER}${NC}"
    else
      echo -e "${GREEN}📌 Latest remote: ${REMOTE_VER}${NC}"
    fi

    echo -e "${GREEN}📌 Using latest version: ${LATEST_VERSION}${NC}"
    if ! VERSION=$(_increment_version "$LATEST_VERSION" "$BUMP_TYPE"); then
      echo -e "${RED}❌ Failed to increment version${NC}"
      exit 1
    fi
    echo -e "${GREEN}🚀 Incrementing ${BUMP_TYPE}: ${LATEST_VERSION} → ${VERSION}${NC}"
  fi
fi

# Sanity: require new tags to be clean semver. Old-format
# ``vX.Y.Z-cuXXX-pyXXX`` tags are rejected here so the bake-out happens
# only after the whole pipeline migrates to short tags.
if [[ ! "$VERSION" =~ ${SEMVER_TAG_RE} ]]; then
  echo -e "${RED}Error: --version must be clean semver (e.g. v1.0.2). Got: ${VERSION}${NC}"
  echo -e "${YELLOW}Hint: dependency profile (CUDA/Python) is now baked into image labels, not the tag.${NC}"
  exit 1
fi

echo "============================================================"
echo -e "${GREEN}RyotenkAI Training - Build and Push Runtime Image${NC}"
echo "============================================================"
echo "Image:              ${IMAGE}"
echo "Version:            ${VERSION}"
echo "Bump type:          ${BUMP_TYPE}"
echo "Push to Hub:        ${PUSH_IMAGES}"
echo "Profile:            CUDA ${CUDA_VERSION} / Python ${PYTHON_VERSION} / Torch ${TORCH_VERSION}"
echo ""

if [[ "$PUSH_IMAGES" == "true" ]] && [[ "$SKIP_LOGIN" == "false" ]]; then
  echo "Logging in to Docker Hub..."
  if [[ -n "${DOCKER_PASSWORD:-}" ]]; then
    echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin
  else
    docker login -u "$DOCKER_USERNAME"
  fi
  echo -e "${GREEN}✓ Logged in to Docker Hub${NC}"
  echo ""
fi

echo "============================================================"
echo "Building ${IMAGE}:${VERSION}..."
echo "============================================================"

# Build args plumb the dependency profile into the image (labels +
# version.txt). The Dockerfile re-declares each ARG after FROM so it's
# usable in both stages.
docker build \
  --platform linux/amd64 \
  --build-arg "IMAGE_VERSION=${VERSION}" \
  --build-arg "CUDA_VERSION=${CUDA_VERSION}" \
  --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
  --build-arg "TORCH_VERSION=${TORCH_VERSION}" \
  -f docker/training/Dockerfile.runtime \
  -t "${IMAGE}:${VERSION}" \
  .

echo -e "${GREEN}✓ Image built successfully${NC}"
echo ""

if [[ "$PUSH_IMAGES" == "true" ]]; then
  echo "Pushing ${IMAGE}:${VERSION}..."
  docker push "${IMAGE}:${VERSION}" 2>&1 | tee "/tmp/docker_push_${VERSION}.log"
  PUSH_EXIT_CODE="${PIPESTATUS[0]}"

  if [[ "$PUSH_EXIT_CODE" -eq 0 ]] && grep -q "digest:" "/tmp/docker_push_${VERSION}.log"; then
    DIGEST=$(grep "digest:" "/tmp/docker_push_${VERSION}.log" | tail -1 | awk '{print $NF}')
    echo -e "${GREEN}✓ Image pushed successfully${NC}"
    echo -e "${GREEN}✓ Digest: ${DIGEST}${NC}"

    # Remove local image after successful push
    echo ""
    echo "Removing local image after successful push..."
    if docker rmi "${IMAGE}:${VERSION}" 2>/dev/null; then
      echo -e "${GREEN}✓ Local image removed${NC}"
    else
      echo -e "${YELLOW}⚠️  Could not remove local image (may be in use)${NC}"
    fi
  else
    echo -e "${RED}❌ Push failed (exit code: ${PUSH_EXIT_CODE}) - keeping local image${NC}"
    if [[ -f "/tmp/docker_push_${VERSION}.log" ]]; then
      echo -e "${RED}Last log lines:${NC}"
      tail -10 "/tmp/docker_push_${VERSION}.log"
    fi
    exit 1
  fi
  echo ""
else
  echo -e "${YELLOW}Image built locally (not pushed)${NC}"
  echo ""
fi

echo "============================================================"
echo -e "${GREEN}Build Complete${NC}"
echo "============================================================"
echo ""
echo "Tag:"
echo "  - ${IMAGE}:${VERSION}"
echo ""
echo "Pin in code (single source of truth):"
echo "  src/runner/__about__.py:"
echo "    _DEFAULT_RUNTIME_IMAGE: Final[str] = ("
echo "        \"${IMAGE}:${VERSION}\""
echo "    )"
echo ""
echo "Inspect dependency profile of the published image:"
echo "  docker run --rm ${IMAGE}:${VERSION} cat /opt/ryotenkai/version.txt"
echo "  docker inspect ${IMAGE}:${VERSION} | jq '.[0].Config.Labels'"
echo ""
