SHELL := /bin/zsh

.DEFAULT_GOAL := help

help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: venv
venv: ## Create a python virtual environment under .venv
	python -m virtualenv -p python3 .venv

.PHONY: venv-rm
venv-rm: ## Delete the python virtual environment under .venv
	rm -rf .venv

.PHONY: docs
docs: ## Update documentation
	pdoc --force --output-dir docs  src/stages src/libs

.PHONY: check-deps
check-deps: ## Check dependencies
	@./scripts/check-deps.sh

.PHONY: demo
demo: ## Run the Demo UI
	poetry run python src/app.py --config=params-dev.yaml --model=models/best_model.pt

.PHONY: api
api: ## Run the predict API
	poetry run uvicorn src.api.predict:app --reload --host=0.0.0.0 --port=9090

.PHONY: clean
clean: ## Clean all
	mkdir -p runs/detect
	rm -rf runs/detect/*
	rm -rf wandb/*
	find data -iname '*.cache' -exec rm {} \;

