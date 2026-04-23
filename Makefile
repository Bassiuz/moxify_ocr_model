.PHONY: test lint typecheck fmt check

test:
	uv run pytest -v

lint:
	uv run ruff check src tests scripts

typecheck:
	uv run mypy src scripts

fmt:
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

check: lint typecheck test
