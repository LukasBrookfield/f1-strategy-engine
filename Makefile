.DEFAULT_GOAL := help
.PHONY: help setup ingest test lint format

help: ## Show this help message
	@echo "Available commands:"
	@awk -F ':.*## ' '/^[a-z_]+:.*##/ {printf "  %-10s %s\n", $$1, $$2}' Makefile

setup:  ## Install project dependencies
	uv sync

test:  ## Run test suite
	uv run pytest

lint:  ## Run code linter and formatting checks
	uv run ruff check
	uv run ruff format --check

format:  ## Ruff will make changes to the files to format the code
	uv run ruff check --fix
	uv run ruff format

