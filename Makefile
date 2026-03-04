PYTHON ?= python

.PHONY: install train app test lint format

install:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e .[dev]

train:
	$(PYTHON) scripts/train_model.py

app:
	streamlit run app/app.py

test:
	pytest

lint:
	ruff check .

format:
	ruff format .
