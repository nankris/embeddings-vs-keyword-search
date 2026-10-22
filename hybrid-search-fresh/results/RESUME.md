# Run state and how to reproduce

The detached worker completed all three datasets and the `hybrid-search` Colab session was stopped. Results are local, so reproducing the metrics and figures needs no GPU: run the scripts below on the saved rankings.

## What happened during the run

The first allocation wait was interrupted and the run was resumed. The Colab session `hybrid-search` was allocated on 2026-10-18 at 22:46:41 UTC (see `allocation.json`). The detached worker ran at `/content/hybrid-search/run.py` on the VM and checkpointed every 512 encoded inputs. Only one worker ran at a time.

The pilot is frozen in `pilot_selection.json` and committed in 5690d18. Confirmation is FiQA, NFCorpus and ArguAna, with no retuning. All lexical ranks were computed locally beforehand. The exact executed worker is `executed_worker.py`; `scripts/run.py` additionally makes writes atomic for future reruns. The hypothesis and decision rule were committed initially in af27633.

## Reproducing from saved results

Run, in order: `scripts/stamp_timing.py eval_start`, `scripts/check_metrics.py`, `scripts/audit_results.py`, `scripts/analyze.py`, `scripts/supplement.py`, `scripts/query_diagnostics.py`, `scripts/stamp_timing.py eval_end`, `scripts/stamp_timing.py finish`, then `scripts/report.py` for the measured report and README appendix. `scripts/check_deliverables.py` checks that the figures and documents are consistent.

To inspect a live session instead, run `colab exec -s hybrid-search -f scripts/resume_probe.py` (or `scripts/health_remote.py`). If it is still running, resume `scripts/monitor.py`, which downloads snapshots and stops only that session on done/error. If `worker_done.json` and `stopped.json` already exist locally, no GPU is needed.
