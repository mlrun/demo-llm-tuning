
PYTHON_INTERPRETER = python3
SHARED_DIR ?= ~/mlrun-data
MLRUN_TAG ?= 1.3.0
HOST_IP ?=$$(ip route get 1.2.3.4 | awk '{print $$7}')
CONDA_ENV ?= mlrun
SHELL=/bin/bash
CONDA_PY_VER ?= 3.9
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: help
help: ## Display available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: all
all:
	$(error please pick a target)

.PHONY: install-requirements
install-requirements: ## Install all requirements needed for development
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt -r dev-requirements.txt


.PHONY: package-wheel
package-wheel: clean ## Build python package wheel
	python setup.py bdist_wheel

.PHONY: clean
clean: ## Clean python package build artifacts
	rm -rf build
	rm -rf dist
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

.PHONY: fmt
fmt: ## Format the code (using black and isort)
	@echo "Running black fmt..."
	$(PYTHON_INTERPRETER) -m black src tests
	$(PYTHON_INTERPRETER) -m isort src tests

.PHONY: lint
lint: fmt-check flake8 ## Run lint on the code

.PHONY: fmt-check
fmt-check: ## Format and check the code (using black and isort)
	@echo "Running black+isort fmt check..."
	$(PYTHON_INTERPRETER) -m black --check --diff src tests
	$(PYTHON_INTERPRETER) -m isort --check --diff src tests

.PHONY: flake8
flake8: ## Run flake8 lint
	@echo "Running flake8 lint..."
	$(PYTHON_INTERPRETER) -m flake8 src tests

.PHONY: test
test: clean ## Run tests
	$(PYTHON_INTERPRETER) -m pytest -v --capture=no --disable-warnings -rf tests

.PHONY: mlrun-docker
mlrun-docker: ## Start MLRun & Nuclio containers (using Docker compose)
	mkdir $(SHARED_DIR) -p
	@echo "HOST_IP=$(HOST_IP)" > .env
	SHARED_DIR=$(SHARED_DIR) TAG=$(MLRUN_TAG) docker-compose -f compose.yaml up -d
	@echo "use docker-compose stop / logs commands to stop or view logs"

.PHONY: mlrun-api
mlrun-api: ## Run MLRun DB locally (as process)
	@echo "Installing MLRun API dependencies ..."
	$(PYTHON_INTERPRETER) -m pip install uvicorn~=0.17.0 dask-kubernetes~=0.11.0 apscheduler~=3.6 sqlite3-to-mysql~=1.4
	@echo "Starting local mlrun..."
	MLRUN_ARTIFACT_PATH=$$(realpath ./artifacts) MLRUN_ENV_FILE= mlrun db -b

.PHONY: conda-env
conda-env: ## Create a conda environment
	@echo "Creating new conda environment $(CONDA_ENV)..."
	conda create -n $(CONDA_ENV) -y python=$(CONDA_PY_VER) ipykernel graphviz pip
	test -s ./mlrun.env && conda env config vars set -n $(CONDA_ENV) MLRUN_ENV_FILE=$$(realpath ./mlrun.env)
	@echo "Installing requirements.txt..."
	$(CONDA_ACTIVATE) $(CONDA_ENV); pip install -r requirements.txt
	@echo -e "\nTo run mlrun API as a local process type:\n  conda activate $(CONDA_ENV) && make mlrun-api"