.PHONY: status test torture-test benchmark realtime-benchmark turn-tournament checkpoint

status:
	python3 scripts/status.py

test:
	python3 -m pytest -q

torture-test:
	python3 -m pytest -q tests/golden tests/unit

benchmark:
	PYTHONPATH=src python3 scripts/benchmark_turns.py

realtime-benchmark:
	PYTHONPATH=src python3 scripts/benchmark_realtime_core.py

turn-tournament: benchmark

checkpoint:
	python3 scripts/checkpoint.py
