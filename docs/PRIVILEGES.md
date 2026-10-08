# Privileges and safety

`moepager` only **reads** the model file and gives the kernel hints about it.
It never writes model data, and it never touches the engine's memory except
through `process_madvise` advice, which cannot change contents.

## Feature Privilege Matrix

| feature | needs | without it |
|---|---|---|
| sentinel / full-scan `mincore` observation | read access to the model file | — |
| `cachestat` probe | owner of, or write access to, the file (EPERM otherwise, verified on 6.19) | falls back to `mincore` |
| `posix_fadvise(WILLNEED)` prefetch / completion readahead | read access | — |
| pinning (`mlock` on the daemon's own mapping) | `CAP_IPC_LOCK` or `RLIMIT_MEMLOCK` ≥ pin budget (default: 8 MB) | pin budget clamped to the rlimit, warning logged |
| demotion (`process_madvise(MADV_COLD)` on the engine) | `CAP_SYS_NICE` + ptrace-read access to the engine (same user, `ptrace_scope` ≤ 1) | disabled; reclaim left to the kernel |
| eBPF filemap tracepoints | root or `CAP_BPF`+`CAP_PERFMON` | sentinel polling |
| MGLRU on/off baseline | root (`/sys/kernel/mm/lru_gen/enabled`) | baseline skipped |
| cgroup budget for experiments | systemd user delegation of `memory` (default on Fedora) | — |

---

## Least-Privilege Setup

Recommended least-privilege setup for a dedicated binary:
```bash
sudo setcap cap_ipc_lock,cap_sys_nice+ep target/release/moepagerd
```
Do **not** run the daemon as root just to get these two capabilities.

### Systemd User Service Setup

If running as a persistent user service, use ambient capabilities without elevating to root:

```ini
[Unit]
Description=moepager MoE page-cache daemon
After=network.target

[Service]
Type=simple
ExecStart=%h/.local/bin/moepagerd %h/models/model.gguf --policy v
AmbientCapabilities=CAP_IPC_LOCK CAP_SYS_NICE
CapabilityBoundingSet=CAP_IPC_LOCK CAP_SYS_NICE
LimitMEMLOCK=infinity
NoNewPrivileges=false
Restart=on-failure

[Install]
WantedBy=default.target
```

---

## Troubleshooting Permissions

### 1. `RLIMIT_MEMLOCK` Clamping
- **Symptom**: `moepagerd` logs `clamping pin budget from <requested> to <limit>`.
- **Cause**: Unprivileged user accounts are typically restricted to 8 MB of mlock memory (`ulimit -l`).
- **Fix**: Grant `CAP_IPC_LOCK` via `setcap`, or raise limits in `/etc/security/limits.d/99-moepager.conf`:
  ```
  <username> soft memlock unlimited
  <username> hard memlock unlimited
  ```

### 2. `process_madvise` EPERM
- **Symptom**: Demotion actions fail with `Operation not permitted`.
- **Cause**: Missing `CAP_SYS_NICE` or Yama ptrace protection restriction (`kernel.yama.ptrace_scope > 1`).
- **Fix**: Verify ownership matches the target llama.cpp process and ensure `CAP_SYS_NICE` is assigned. Check `/proc/sys/kernel/yama/ptrace_scope`.

---

## Risk Analysis

- **Memory Starvation**: Pinning too much memory starves co-tenant applications. The pin budget (`--pin-budget-bytes`) must always be explicitly bounded below free memory.
- **Data Integrity**: `moepager` opens files in read-only mode (`O_RDONLY`). Neither `posix_fadvise` nor `process_madvise(MADV_COLD)` can corrupt memory or disk data.


---

## AppArmor & SELinux Profiles
On distributions enforcing LSMs (AppArmor on Ubuntu/Debian, SELinux on Fedora/RHEL):
- SELinux: Ensure the unconfined_service_t or container policy allows ptrace checks for process_madvise cross-process signaling.
- AppArmor: Add capability ipc_lock, capability sys_nice, to /etc/apparmor.d/usr.local.bin.moepagerd.

