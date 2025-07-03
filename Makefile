.PHONY: run-backend run-ui lint format mypy precommit all

run-backend:
	uvicorn app.main:app --reload --port 8000

run-ui:
	streamlit run app/ui/streamlit_app.py

lint:
	ruff check .

format:
	ruff format .

mypy:
	mypy app/

precommit:
	pre-commit run --all-files

all: lint format mypy precommit
