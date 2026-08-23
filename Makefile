.PHONY: status test

status:
	python3 scripts/status.py

test:
	python3 -m pytest -q

