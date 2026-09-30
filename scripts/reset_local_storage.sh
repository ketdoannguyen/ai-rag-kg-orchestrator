#!/usr/bin/env bash
#
# Delete the local ArangoDB and MinIO data so both stores start clean.
#
# Only these two services are touched. Redis is never stopped, recreated or
# removed, and the script verifies that afterwards.
#
# This is destructive and there is no backup step. Run it without --confirm
# first to see exactly what would be deleted:
#
#   bash scripts/reset_local_storage.sh
#   bash scripts/reset_local_storage.sh --confirm RESET-ARANGO-MINIO
#
# Exit codes: 0 success, 1 runtime error, 2 confirmation missing or wrong.

set -euo pipefail

CONFIRM_PHRASE="RESET-ARANGO-MINIO"
SERVICES=(arangodb minio)
VOLUMES=(ai-rag-kg-arangodb-data ai-rag-kg-minio-data)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/docker/docker-compose.yml"
ENV_FILE="${REPO_ROOT}/.env"
DATA_DIR="${REPO_ROOT}/docker/data"
LEGACY_PATHS=("${DATA_DIR}/arangodb_clean" "${DATA_DIR}/minio")

confirm=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --confirm)
      if [[ $# -lt 2 ]]; then
        echo "--confirm needs the exact phrase: ${CONFIRM_PHRASE}" >&2
        exit 2
      fi
      confirm="$2"
      shift 2
      ;;
    --confirm=*)
      confirm="${1#*=}"
      shift
      ;;
    -h|--help)
      sed -n '3,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

compose() {
  docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" "$@"
}

redis_fingerprint() {
  local container
  container="$(compose ps -q redis 2>/dev/null || true)"
  if [[ -z "${container}" ]]; then
    echo "redis=not-running"
    return
  fi
  local started size
  started="$(docker inspect -f '{{.State.StartedAt}}' "${container}" 2>/dev/null || echo unknown)"
  size="$(docker exec "${container}" redis-cli DBSIZE 2>/dev/null || echo unknown)"
  echo "redis id=${container} started=${started} dbsize=${size}"
}

echo "Repository        : ${REPO_ROOT}"
echo "Compose file      : ${COMPOSE_FILE}"
echo "Env file          : ${ENV_FILE}"
echo

echo "== Containers that will be stopped and removed =="
for service in "${SERVICES[@]}"; do
  container="$(compose ps -aq "${service}" 2>/dev/null || true)"
  if [[ -n "${container}" ]]; then
    docker inspect -f '   {{.Name}} image={{.Config.Image}} status={{.State.Status}}' "${container}"
  else
    echo "   ${service}: no container"
  fi
done
echo

echo "== Named volumes that will be deleted =="
for volume in "${VOLUMES[@]}"; do
  if docker volume inspect "${volume}" >/dev/null 2>&1; then
    docker volume inspect -f '   {{.Name}} mountpoint={{.Mountpoint}}' "${volume}"
  else
    echo "   ${volume}: does not exist"
  fi
done
echo

echo "== Legacy bind mounts that will be deleted =="
for path in "${LEGACY_PATHS[@]}"; do
  # Refuse to delete anything resolving outside this repository's docker/data.
  resolved="$(realpath -m "${path}")"
  if [[ "${resolved}" != "${DATA_DIR}"/* ]]; then
    echo "Refusing to delete ${resolved}: outside ${DATA_DIR}" >&2
    exit 1
  fi
  if [[ -d "${path}" ]]; then
    echo "   ${path} size=$(du -sh "${path}" 2>/dev/null | cut -f1)"
  else
    echo "   ${path}: does not exist"
  fi
done
echo

echo "== Redis (never touched) =="
echo "   $(redis_fingerprint)"
echo

if [[ "${confirm}" != "${CONFIRM_PHRASE}" ]]; then
  echo "Confirmation required. Nothing was changed." >&2
  echo "Re-run with: bash scripts/reset_local_storage.sh --confirm ${CONFIRM_PHRASE}" >&2
  exit 2
fi

if ! docker info >/dev/null 2>&1; then
  echo "Cannot reach the Docker daemon. Add your user to the 'docker' group or run via sudo." >&2
  exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}; Compose needs it to resolve container credentials." >&2
  exit 1
fi

evidence_dir="/tmp/ai-rag-kg-storage-reset-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${evidence_dir}"
redis_fingerprint >"${evidence_dir}/redis-before.txt"
compose ps -a >"${evidence_dir}/compose-ps-before.txt" 2>&1 || true
docker volume ls >"${evidence_dir}/volumes-before.txt"

echo "Evidence directory: ${evidence_dir}"
echo

echo "== Stopping ${SERVICES[*]} =="
compose stop "${SERVICES[@]}"
compose rm -f "${SERVICES[@]}"

echo "== Removing named volumes =="
volume_failures=0
for volume in "${VOLUMES[@]}"; do
  if ! docker volume inspect "${volume}" >/dev/null 2>&1; then
    echo "   ${volume}: already absent"
  elif docker volume rm "${volume}" >/dev/null 2>&1; then
    echo "   removed ${volume}"
  else
    # Removing an in-use volume fails; do not report that as "already absent".
    echo "   ${volume}: REMOVAL FAILED (still in use?)" >&2
    volume_failures=$((volume_failures + 1))
  fi
done

echo "== Removing legacy bind mounts =="
for path in "${LEGACY_PATHS[@]}"; do
  if [[ -d "${path}" ]]; then
    rm -rf "${path}"
    echo "   removed ${path}"
  else
    echo "   ${path}: already absent"
  fi
done

redis_fingerprint >"${evidence_dir}/redis-after.txt"
compose ps -a >"${evidence_dir}/compose-ps-after.txt" 2>&1 || true
docker volume ls >"${evidence_dir}/volumes-after.txt"

echo
if ! diff -q "${evidence_dir}/redis-before.txt" "${evidence_dir}/redis-after.txt" >/dev/null; then
  echo "Redis changed during the reset — this is a bug." >&2
  diff -u "${evidence_dir}/redis-before.txt" "${evidence_dir}/redis-after.txt" >&2 || true
  exit 1
fi
echo "Redis unchanged: $(cat "${evidence_dir}/redis-after.txt")"

if (( volume_failures > 0 )); then
  echo "${volume_failures} volume(s) could not be removed — some data may remain." >&2
  exit 1
fi
echo "ArangoDB and MinIO data removed. Run 'make storage-up storage-init' next."
