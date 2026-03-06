# Changelog

All notable changes to this project will be documented in this file.

## 2026-03-05

### Fixed
- Fixed GitHub Actions workflow YAML (removed accidental config block) and made CI run the offline smoke test.
- Made artifact upload non-fatal when no files exist.
- Fixed a syntax error in `research/smoke_test.py` subprocess paths (removed stray escaped quotes) so `python research/smoke_test.py` runs.
- Fixed `NameError: env is not defined` in `research/smoke_test.py` by removing an unintended `env` reference inside `write_partitioned()`.

### Added
- `setup.cfg` (flake8 config placeholder).
- Completed README quick starts (offline + live) with outputs and config knobs.
- Offline smoke test now generates synthetic TOB + trades and runs the minimal end-to-end pipeline.

### Improved
- Added `edge_per_bar = avg_net_bps * trade_rate` to both taker and maker grid outputs, plus matching heatmaps.
- `run_pipeline.py` is now cross-platform (Windows + macOS/Linux) and supports `DURATION_MIN` and `PRINT_EVERY`.
- `.gitignore` expanded for common local artifacts.

### Verified
- Offline smoke test runs end-to-end on Windows PowerShell and writes metrics CSVs and heatmaps under `results/`.
