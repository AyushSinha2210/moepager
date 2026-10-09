# Docker & Container Deployment Architecture

Deploying `moepager` in microservices and containerized environments.

## Capability Grants
When containerized, grant required ambient capabilities:
- `--cap-add=IPC_LOCK`: Required if allocating an `mlock` pin budget.
- `--cap-add=SYS_NICE`: Required for `process_madvise(MADV_COLD)` demotion.

## Storage Mounts
Mount model storage with direct host filesystem paths to preserve POSIX fadvise and mincore semantics. Avoid overlayfs layers for model storage.
