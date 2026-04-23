.PHONY: test lint typecheck fmt check

test:
	uv run pytest -v

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy src

fmt:
	uv run ruff format src tests
	uv run ruff check --fix src tests

check: lint typecheck test
