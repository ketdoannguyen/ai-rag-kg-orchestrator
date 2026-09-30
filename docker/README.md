# Local data services

`docker-compose.yml` in this folder runs the two stores the application needs on
a developer machine:

| Service | Image | Host endpoint | Data |
| --- | --- | --- | --- |
| ArangoDB | `arangodb:3.12.12` (Enterprise build, digest-pinned) | `127.0.0.1:8529` | volume `ai-rag-kg-arangodb-data` |
| MinIO | `quay.io/minio/minio:RELEASE.2025-09-07T16-13-09Z` (digest-pinned) | `127.0.0.1:9002` (console `:9003`) | volume `ai-rag-kg-minio-data` |
| Redis | `redis:7` (unchanged) | `0.0.0.0:6379` | bind mount `docker/data/redis` |

ArangoDB and MinIO are bound to loopback only. Redis keeps its pre-existing
definition and is not part of the storage lifecycle targets below.

MinIO maps to host ports **9002/9003** rather than its usual 9000/9001, because
another Compose project on this machine (`vku-science-infra`) already publishes
`127.0.0.1:9000` and `:9001`. `MINIO_ENDPOINT` in `.env` must stay in step with
this mapping. If that other project is removed, the ports can go back to
9000/9001 by editing both files.

## Prerequisites

- Docker Engine with the Compose plugin.
- Your user can reach the Docker socket. If `docker info` reports
  `permission denied ... /var/run/docker.sock`, add yourself to the `docker`
  group and start a new session (or run the commands with `sudo`):

  ```bash
  sudo usermod -aG docker "$USER"
  ```

- `uv` for the Python scripts, with the workspace members installed once:

  ```bash
  uv sync --all-packages
  ```

  `--all-packages` matters: `arango_client` and `minio_client` are workspace
  members, not dependencies of the root project, so a plain `uv sync` leaves
  them out and the scripts fail to import them.

## Credentials

A single untracked file at the repository root, `.env`, configures both the host
application and these containers. Create it once:

```bash
cp .env.example .env
# then replace every <placeholder>
```

`ARANGO_PASSWORD`, `MINIO_ACCESS_KEY` and `MINIO_SECRET_KEY` are passed to the
containers as their root credentials. Use values without spaces, quotes or `$`,
because the ArangoDB healthcheck reads the password through a shell.

Every Compose command must pass `--env-file .env` explicitly, otherwise Compose
looks for `.env` inside this `docker/` folder instead of the repository root.
The `make` targets and the scripts already do this. That includes the
pre-existing `make up` / `down` / `restart` / `logs` / `ps` / `build` targets,
which now need a filled-in `.env` because this compose file takes its
credentials from it rather than from a literal in the file.

`docker/.env.dev` is only used by `docker-compose.services.yml` (the API and
worker images); it plays no part in the storage stack.

## First-time setup

```bash
make storage-up      # start ArangoDB + MinIO only
make storage-init    # create the schema and the two buckets
make storage-verify  # assert everything is in place
```

`storage-init` is idempotent: it reuses `ArangoClient(auto_init=True)` and
`StorageClient.ensure_bucket()`, so a second run creates nothing new. (The
client still reconciles the graph if the edge definitions in the code changed
since the last run.) It never inserts documents, ontology rows or objects.

Verification covers:

- ArangoDB reports `3.12.12` on database `knowledge_db`.
- The ArangoDB license is usable: status `good`, disk limit not reached, and not
  approaching read-only (see the edition note under Known constraints).
- Exactly the 12 document collections and 1 edge collection declared in
  `libs/arango_client/src/arango_client/collections/schema.py`, every one empty.
- Every index declared in `INDEXES` exists with matching name, fields, `unique`
  and `sparse`.
- Graph `knowledge_graph` has the expected edge definition over the four
  knowledge collections.
- View `knowledge_vector_view` exists and links those collections.
- `COSINE_SIMILARITY()` evaluates; the reported index types are printed.
- MinIO answers `/minio/health/live` and `/minio/health/ready`.
- Buckets `ingest` and `knowledge-files` exist and are empty.
- A temporary object survives an upload/download round-trip and is deleted again.

## Routine operations

```bash
make storage-up        # start both stores
make storage-down      # stop both stores, keep their data
make storage-verify    # re-check the current state
make logs              # follow logs for all data services
```

Do not use `make down` for routine work: it removes every container in this
compose file, including Redis.

## Resetting ArangoDB and MinIO

There is no backup step. Preview first, then confirm:

```bash
make storage-reset                                  # prints what would be deleted, deletes nothing
make storage-reset CONFIRM=RESET-ARANGO-MINIO       # actually deletes
```

Equivalent direct invocation:

```bash
bash scripts/reset_local_storage.sh --confirm RESET-ARANGO-MINIO
```

The script validates that every path it deletes resolves inside `docker/data/`,
stops and removes only `arangodb` and `minio`, deletes only the named volumes
`ai-rag-kg-arangodb-data` and `ai-rag-kg-minio-data`, and refuses to use
`compose down`, `down -v`, `system prune` or `volume prune`. It captures evidence
under `/tmp/ai-rag-kg-storage-reset-<timestamp>/`, compares Redis before and
after, and fails if Redis changed in any way.

## Known constraints

- **MinIO community images no longer come from Docker Hub.** Docker Hub serves
  `401` for every `minio/minio` tag, the upstream repository is archived, and the
  project now distributes the community edition as source only. This stack pins
  the Quay build `RELEASE.2025-09-07T16-13-09Z`, which is the last community
  container verified locally. It receives no further upstream updates: treat it
  as a frozen dependency, and plan a replacement if MinIO becomes unsupported
  for your use case.
- **ArangoDB's official image is the Enterprise build, not Community.** ArangoDB
  no longer publishes a Community container — the Docker Hub repository exposes no
  `community` tag, and the pinned digest reports
  `ArangoDB 3.12.12 enterprise`. Unlicensed it still runs fully, below a disk
  limit (100 GiB and ~10 years until read-only on this instance), which
  `storage-verify` asserts. A genuine Community build would have to be compiled
  from source; do not describe this stack as Community.
- **No ANN vector index.** The running server reports only `primary` and
  `persistent` indexes, so similarity search is a brute-force
  `COSINE_SIMILARITY()` scan. Do not expect ANN-level latency at large corpus
  sizes, and do not describe the system as using an approximate index.
- **Reset is irreversible.** ArangoDB and MinIO data are deleted permanently;
  there is no dump, snapshot or replica.
- **Both images are pinned by manifest digest**, read from the verified local
  images rather than from a floating tag.
- **Both healthchecks are confirmed working.** `arangosh` and `mc` were verified
  present in their pinned images, and `docker compose ps` reports `healthy` for
  both services. If ArangoDB stays `starting`, check the password for spaces or
  quotes, which break the shell command in the check.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `permission denied ... docker.sock` | user not in the `docker` group | `sudo usermod -aG docker "$USER"`, then start a new session |
| Compose: `ARANGO_PASSWORD must be set` | `.env` missing or the variable is blank | `cp .env.example .env` and fill it in |
| Compose: `open .env: no such file or directory` | command run without `--env-file .env` | use the `make` targets |
| `storage-init` reports `ARANGO_DATABASE is not set` | `.env` still holds the template placeholder | replace every `<placeholder>` |
| `storage-verify` fails on version | a different ArangoDB tag is running | pass `--expect-arango-version`, or repin the image |
| ArangoDB container never becomes `healthy` | password contains spaces/quotes/`$` | use a simple password, then `make storage-down storage-up` |
| MinIO container never becomes `healthy` | the server is not up, or the check runs before `start_period` elapses | check `docker compose logs minio`; `curl 127.0.0.1:9002/minio/health/live` must answer `200` |
| `docker compose pull` returns `401 Unauthorized` for quay.io | this host's egress blocks quay.io | harmless when the image is already local — `make storage-up` runs without pulling. Pull on a host that can reach quay.io, or `docker load` a saved image. |
| `storage-up` fails with `port is already allocated` | another project publishes the same host port | `docker ps --format '{{.Names}} {{.Ports}}'`; this stack expects 8529, 9002 and 9003 to be free |
| `storage-verify` fails on edition | a different ArangoDB build is running | pass `--expect-license`, or repin the image |
| `make storage-reset` ends with `Error 2` | preview mode exits `2` on purpose so a missing `CONFIRM` cannot be mistaken for success | re-run with `CONFIRM=RESET-ARANGO-MINIO` |
