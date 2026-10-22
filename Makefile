.DEFAULT_GOAL := help
.PHONY: help setup ingest fixture test lint format train leakage-report recommend

# Overridable from the command line, e.g. `make ingest SEASON=2025 ROUND=3`
SEASON ?=
ROUND ?=

# Demo race state for `make recommend`, e.g. `make recommend DRIVER=NOR LAP=30`
REC_SEASON ?= 2025
REC_ROUND ?= 14
DRIVER ?= NOR
LAP ?= 44

FIXTURE_PARTITION := season=2024/round=15/session=R

help: ## Show this help message
	@echo "Available commands:"
	@awk -F ':.*## ' '/^[a-z_-]+:.*##/ {printf "  %-16s %s\n", $$1, $$2}' Makefile

setup:  ## Install project dependencies
	uv sync

ingest:  ## Backfill FastF1 sessions into data/raw (SEASON=, ROUND= to narrow)
	uv run f1se ingest $(if $(SEASON),--season $(SEASON)) $(if $(ROUND),--round $(ROUND))

fixture:  ## Copy the fixture race (2024 R15, Zandvoort) from data/raw into tests/fixtures
	mkdir -p tests/fixtures
	cp data/raw/laps/$(FIXTURE_PARTITION)/data.parquet tests/fixtures/laps.parquet
	cp data/raw/weather/$(FIXTURE_PARTITION)/data.parquet tests/fixtures/weather.parquet

test:  ## Run test suite
	uv run pytest

lint:  ## Run code linter and formatting checks
	uv run ruff check
	uv run ruff format --check

format:  ## Ruff will make changes to the files to format the code
	uv run ruff check --fix
	uv run ruff format

train:  ## Fit degradation models and print held-out scores
	uv run f1se train

leakage-report:  ## Run both point-in-time arms and print the leakage gap
	uv run f1se leakage-report

recommend:  ## Pit recommendation (REC_SEASON=, REC_ROUND=, DRIVER=, LAP=)
	uv run f1se recommend --season $(REC_SEASON) --round $(REC_ROUND) --driver $(DRIVER) --lap $(LAP)
