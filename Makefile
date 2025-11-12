.PHONY: run-backend run-ui sweep lint format mypy precommit all test docker-build-local docker-build-cloud deploy clean

# Variables
PROJECT_ID = le-wagon-data-science-376310
REGION = europe-west3
REPO_NAME = doc-chat-api
IMAGE_NAME = $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(REPO_NAME)/doc-chat-api:latest
BACKEND_URL = https://doc-chat-api-581282400880.europe-west3.run.app

# Run backend locally
run-backend:
	@chmod +x ./scripts/start_local.sh
	@./scripts/start_local.sh

# Run UI locally
run-ui:
	streamlit run app/ui/streamlit_app.py

# sweep alpha and lambda values for tuning
sweep:
	@chmod +x ./scripts/sweep.sh
	@FILE_ID="$(FILE_ID)" \
	QUESTION="$(QUESTION)" \
	ALPHAS="$(ALPHAS)" \
	LAMBDAS="$(LAMBDAS)" \
	DEBUG_HYBRID="$(DEBUG_HYBRID)" \
	./scripts/sweep.sh

# Code quality checks
lint:
	ruff check .

format:
	ruff format .

mypy:
	mypy app/

precommit:
	pre-commit run --all-files

all: lint format mypy precommit

# Run tests
test:
	@pytest

# Build backend Docker image for local testing
docker-build-local:
	@docker build -t doc-chat-api .

docker-run-local:
	@docker run -p 8000:80 doc-chat-api

# Build Docker image for Google Artifact Registry
docker-build-cloud:
	@docker build -t $(IMAGE_NAME) .

docker-run-cloud:
	@docker run -e PORT=8080 -p 8080:8080 $(IMAGE_NAME)

# Deploy to Cloud Run
deploy:
	gcloud auth configure-docker $(REGION)-docker.pkg.dev
	docker push $(IMAGE_NAME)
	gcloud run deploy doc-chat-api \
		--image $(IMAGE_NAME) \
		--platform managed \
		--region $(REGION) \
		--allow-unauthenticated \
		--memory 512Mi \
		--cpu 1 \
		--min-instances 0 \
		--max-instances 3

# Cleanup
clean:
	docker system prune -f

# Build Docker image for Streamlit UI
docker-build-ui:
	@docker build -t doc-chat-ui -f Dockerfile.ui .

# Run Streamlit UI locally via Docker
docker-run-ui:
	@docker run -p 8501:8080 --env DOC_CHAT_API_URL=http://host.docker.internal:8000 doc-chat-ui

# Build and push Streamlit UI to Artifact Registry
docker-build-ui-cloud:
	@docker build -t $(REGION)-docker.pkg.dev/$(PROJECT_ID)/doc-chat-api/doc-chat-ui:latest -f Dockerfile.ui .
	@docker push $(REGION)-docker.pkg.dev/$(PROJECT_ID)/doc-chat-api/doc-chat-ui:latest

# Deploy Streamlit UI to Cloud Run
deploy-ui:
	gcloud run deploy doc-chat-ui \
		--image $(REGION)-docker.pkg.dev/$(PROJECT_ID)/doc-chat-api/doc-chat-ui:latest \
		--platform managed \
		--region $(REGION) \
		--allow-unauthenticated \
		--memory 512Mi \
		--cpu 1 \
		--set-env-vars DOC_CHAT_API_URL=$(BACKEND_URL)
