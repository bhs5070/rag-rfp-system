.PHONY: setup fmt lint test ingest index serve ask

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt
	pre-commit install

fmt:
	ruff check --fix .
	ruff format .

lint:
	ruff check .
	mypy src || true

test:
	pytest -q

ingest:
	python -m src.cli.ingest --config configs/config.local.yaml

index:
	PYTHONPATH=. .venv311/bin/python src/cli/build_index.py

serve:
	PYTHONPATH=. .venv311/bin/python -m uvicorn src.cli.serve_api:app --host 0.0.0.0 --port 8000 --reload

ask:
	PYTHONPATH=. .venv311/bin/python src/cli/ask.py
