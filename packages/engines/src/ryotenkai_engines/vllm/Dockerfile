# vLLM inference image — built from this Dockerfile by the (future)
# CI pipeline at ``.github/workflows/build-engine-images.yml``.
#
# Convention image name:
#   ryotenkai/inference-vllm:{engine.toml [engine].version}
# i.e. ``ryotenkai/inference-vllm:1.0.0`` when version=1.0.0.
#
# Source: this folder is the canonical home of the vLLM image. The CI
# pipeline walks ``packages/engines/src/ryotenkai_engines/*/Dockerfile``,
# reads the sibling ``engine.toml [engine].version``, and builds + pushes.
#
# This Dockerfile is a placeholder during the engines refactor —
# the actual production vLLM image lives in ``docker/inference/Dockerfile``
# until PR-10 completes the migration. PR-10 either:
#   (a) moves docker/inference/Dockerfile here, or
#   (b) keeps the build context external and references it from
#       ``engine.toml [image].default`` (escape hatch).

FROM vllm/vllm-openai:v0.7.0

# RyotenkAI customizations (HF cache layout, healthcheck binary, etc.)
# go here. The actual config is in docker/inference/Dockerfile today.

WORKDIR /workspace
