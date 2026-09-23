.PHONY: setup test verify

setup:
	bash scripts/setup_fast.sh

test:
	pytest -q

verify:
	python scripts/verify_consistent_final.py
