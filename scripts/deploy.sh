#!/usr/bin/env bash
set -e

PROJECT_ID="le-wagon-data-science-376310"
IMAGE="gcr.io/${PROJECT_ID}/doc-chat-api"
SERVICE_NAME="doc-chat-api"

echo "🔧 Building Docker image..."
docker build -t ${IMAGE} .

echo "🚀 Pushing Docker image to GCR..."
docker push ${IMAGE}

echo "🌐 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE} \
  --platform managed \
  --allow-unauthenticated \
  --region europe-west1

echo "✅ Deployment complete!"
