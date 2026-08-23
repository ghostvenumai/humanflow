.PHONY: status test torture-test benchmark realtime-benchmark recovery-benchmark replay scorecard runtime-quality router-report tournament-report demo demo-benchmark demo-package dashboard-capture challenge-demo dashboard turn-tournament checkpoint

status:
	python3 scripts/status.py

test:
	python3 -m pytest -q

torture-test:
	python3 -m pytest -q tests/golden tests/unit
	PYTHONPATH=src python3 scripts/run_torture.py

benchmark:
	PYTHONPATH=src python3 scripts/benchmark_turns.py

realtime-benchmark:
	PYTHONPATH=src python3 scripts/benchmark_realtime_core.py

recovery-benchmark:
	PYTHONPATH=src python3 scripts/benchmark_recovery.py

replay:
	PYTHONPATH=src python3 scripts/capture_timeline_replay.py

scorecard:
	PYTHONPATH=src python3 scripts/build_scorecard.py

runtime-quality:
	PYTHONPATH=src python3 scripts/evaluate_runtime_quality.py

router-report:
	PYTHONPATH=src python3 scripts/report_development_router.py

tournament-report:
	PYTHONPATH=src python3 scripts/report_tournament_readiness.py

demo:
	PYTHONPATH=src uvicorn humanflow.web.app:app --host 127.0.0.1 --port 8765

demo-benchmark:
	PYTHONPATH=src python3 scripts/benchmark_browser_demo.py

demo-package:
	PYTHONPATH=src python3 scripts/build_everlast_demo.py

dashboard-capture:
	PYTHONPATH=src python3 scripts/capture_dashboard.py

challenge-demo: demo-package demo

dashboard: demo

turn-tournament: benchmark

checkpoint:
	python3 scripts/checkpoint.py
