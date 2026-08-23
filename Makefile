.PHONY: status test torture-test benchmark realtime-benchmark recovery-benchmark replay scorecard runtime-quality demo demo-benchmark turn-tournament checkpoint

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

demo:
	PYTHONPATH=src uvicorn humanflow.web.app:app --host 127.0.0.1 --port 8765

demo-benchmark:
	PYTHONPATH=src python3 scripts/benchmark_browser_demo.py

turn-tournament: benchmark

checkpoint:
	python3 scripts/checkpoint.py
