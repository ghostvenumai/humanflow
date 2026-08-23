.PHONY: status test torture-test benchmark turn-tournament

status:
	python3 scripts/status.py

test:
	python3 -m pytest -q

torture-test:
	python3 -m pytest -q tests/golden tests/unit

benchmark:
	PYTHONPATH=src python3 scripts/benchmark_turns.py

turn-tournament: benchmark
