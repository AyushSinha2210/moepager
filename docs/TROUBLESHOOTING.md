# Troubleshooting & Diagnostics

Common warnings and operational issues encountered when running `moepager` and `moepagerd`.

## Common Issues & Fixes

### 1. `EPERM` on cachestat
- **Cause**: Linux kernel 6.5+ `cachestat` syscall requires file ownership or write permission.
- **Resolution**: `moepager` automatically degrades to `mincore` probing without errors.

### 2. High Churn Warning
- **Cause**: Pinned experts oscillating due to tight hysteresis margin.
- **Resolution**: Increase `--hysteresis` from `1.25` to `1.40` or `--replan-every` to `8`.

### 3. Missing Slices in WILLNEED
- **Cause**: Kernel truncating readahead length.
- **Resolution**: Ensure `--willneed-chunk-kb 128` is active (default).
