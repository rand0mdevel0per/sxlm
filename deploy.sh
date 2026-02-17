#!/bin/bash
# Deploy SXLM training to Vertex AI

set -e

PROJECT_ID=$1
REGION=${2:-us-central1}

if [ -z "$PROJECT_ID" ]; then
    echo "Usage: ./deploy.sh PROJECT_ID [REGION]"
    exit 1
fi

echo "Building Docker image..."
docker build -t gcr.io/$PROJECT_ID/sxlm-trainer:latest .

echo "Pushing to GCR..."
docker push gcr.io/$PROJECT_ID/sxlm-trainer:latest

echo "Deploying with Terraform..."
cd terraform/vertex
terraform init
terraform apply -var="project_id=$PROJECT_ID" -var="region=$REGION" -auto-approve

echo "Deployment complete!"
