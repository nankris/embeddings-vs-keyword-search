# Completed state

The detached worker completed all three datasets and b9-120b was stopped. Results are local. Do not allocate another session. Reproduce metrics and figures from saved rankings using the scripts below.

# Resume state

The user resumed this experiment after the original allocation wait was interrupted.
The only permitted Colab session is b9-120b. It was allocated on 2026-10-18 at 22:46:41 UTC (see allocation.json). The detached worker runs at /content/exp120b/run.py and checkpoints every 512 encoded inputs on the VM. Never start a second worker while this one runs.

Source /workspace/tools/setup.sh first. Run colab usage and colab sessions. If b9-120b exists, run `colab exec -s b9-120b -f scripts/resume_probe.py` or health_remote.py to inspect. If still running, resume scripts/monitor.py, which downloads snapshots and stops only b9-120b on done/error. If worker_done.json and stopped.json already exist locally, do not allocate a GPU again.

The pilot is frozen in pilot_selection.json and committed in 5690d18. Confirmation is FiQA, NFCorpus and ArguAna, with no retuning. All lexical ranks are already local. The exact executed worker is executed_worker.py; scripts/run.py additionally makes writes atomic for future reruns. Hypothesis and decision rule were committed initially in af27633.

After successful completion and named-session stop: run scripts/stamp_timing.py eval_start, scripts/check_metrics.py, scripts/audit_results.py, scripts/analyze.py, scripts/supplement.py, scripts/query_diagnostics.py, then scripts/stamp_timing.py eval_end. Finish the publication draft based on the measured analysis. Run scripts/stamp_timing.py finish and scripts/report.py for the measured report and README appendix. Inspect figures and run scripts/check_deliverables.py. Commit all final artifacts locally without pushing. Finish with the user's five-line summary.

blog.md is still a methodology draft, not a completed result. Do not publish it before filling measured results, a finding title, surprises, production guidance, limitations and Code link placeholder. Target 1,200-1,300 words to satisfy both requested length ranges. The required diagram has been rendered and visually inspected.
