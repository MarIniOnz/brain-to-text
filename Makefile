VENV_NAME ?= .venv

USER_PYTHON ?= python3.13
VENV_PYTHON = ${VENV_NAME}/bin/python

export PYTHONPATH := $(PYTHONPATH):./src

.PHONY: venv install clean format lint

prepare-venv: $(VENV_NAME)/bin/python

$(VENV_NAME)/bin/python:
	make clean && ${USER_PYTHON} -m venv $(VENV_NAME)

install: prepare-venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -e .

lint: install
	${VENV_PYTHON} -m ruff check
	${VENV_PYTHON} -m ruff check --select I

format: install
	${VENV_PYTHON} -m ruff check --select I --fix
	${VENV_PYTHON} -m ruff format

clean:
	rm -rf .venv
