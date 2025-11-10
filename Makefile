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
	python -m src.cli.build_index --config configs/config.local.yaml

serve:
	uvicorn src.cli.serve_api:app --host 0.0.0.0 --port 8000 --reload

ask:
	python -m src.cli.ask --q "RFP 제출 마감 요약해줘" --top_k 5 --config configs/config.local.yaml
