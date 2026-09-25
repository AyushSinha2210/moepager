# Privileges and safety

moepager only **reads** the model file and gives the kernel hints about it.
It never writes model data, and it never touches the engine's memory except
through `process_madvise` advice, which cannot change contents.

| feature | needs | without it |
|---|---|---|
| sentinel / full-scan `mincore` observation | read access to the model file | — |
| `cachestat` probe | owner of, or write access to, the file (EPERM otherwise, verified on 6.19) | falls back to `mincore` |
| `posix_fadvise(WILLNEED)` prefetch / completion readahead | read access | — |
| pinning (`mlock` on the daemon's own mapping) | `CAP_IPC_LOCK` or `RLIMIT_MEMLOCK` ≥ pin budget (default here: 8 MB) | pin budget clamped to the rlimit, warning logged |
| demotion (`process_madvise(MADV_COLD)` on the engine) | `CAP_SYS_NICE` + ptrace-read access to the engine (same user, `ptrace_scope` ≤ 1) | disabled; reclaim left to the kernel |
| eBPF filemap tracepoints | root or `CAP_BPF`+`CAP_PERFMON` | sentinel polling |
| MGLRU on/off baseline | root (`/sys/kernel/mm/lru_gen/enabled`) | baseline skipped |
| cgroup budget for experiments | systemd user delegation of `memory` (default on Fedora) | — |

Recommended least-privilege setup for a dedicated binary:
`sudo setcap cap_ipc_lock,cap_sys_nice+ep target/release/moepagerd`.
Do **not** run the daemon as root just to get these two capabilities.

Risks:
- Pinning too much starves other apps. The pin budget is always bounded,
  and phase 9 adds PSI-driven shrinking.
- Mis-targeted `process_madvise` only affects performance, never
  correctness.
