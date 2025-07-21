.PHONY: run-backend run-ui lint format mypy precommit all test docker-build clean

run-backend:
	@chmod +x ./start_local.sh
	@./start_local.sh

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

test:
	@pytest

.PHONY: docker-build
docker-build:
	@docker build -t doc-chat-api .

.PHONY: clean
clean:
