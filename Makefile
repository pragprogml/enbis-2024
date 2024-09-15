# Makefile for project management

# Variables
SHELL := /bin/bash
API_PORT ?= 8000
PYTHON := poetry run python
DOCKER_IMAGE_NAME := api

# Colors for pretty printing
CYAN := \033[36m
RESET := \033[0m

.DEFAULT_GOAL := help

.PHONY: help docs check-deps run-demo run-api docker-build docker-run clean lint test

help: ## Display this help message
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

docs: ## Update documentation
	pdoc --force --output-dir docs src/stages src/libs

check-deps: ## Check dependencies
	@./scripts/check-deps.sh

run-demo: ## Run the Demo UI
	$(PYTHON) src/app.py --config=params-dev.yaml --model=models/best_model.pt

run-api: ## Run the predict API
	$(PYTHON) -m uvicorn src.api.predict:app --reload --host=0.0.0.0 --port=$(API_PORT)

docker-build: ## Build the Docker image
	docker buildx build -t $(DOCKER_IMAGE_NAME) -f src/api/Dockerfile .

docker-run: ## Run the Docker image
	docker run --gpus all -e API_PORT=$(API_PORT) -p $(API_PORT):$(API_PORT) -ti $(DOCKER_IMAGE_NAME)

clean: ## Clean all generated files and directories
	@echo "Cleaning up..."
	@mkdir -p runs/detect
	@rm -rf runs/detect/*
	@rm -rf wandb/*
	@find data -iname '*.cache' -delete
	@echo "Cleanup complete."

lint: ## Run linter
	ruff check src/ notebooks/* --fix
	ruff format src/ notebooks/*

.SILENT: help
