#!/bin/bash
set -e

PROJECT_ID=${1:-$PROJECT_ID}
REGION=${2:-us-central1}

if [ -z "$PROJECT_ID" ]; then
    echo "Usage: $0 PROJECT_ID [REGION]"
    exit 1
fi

echo "Deploying SXLM to Vertex AI (Single A100, 80 hours)"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"

# Build Docker image
docker build -t gcr.io/$PROJECT_ID/sxlm:latest .

# Push to GCR
docker push gcr.io/$PROJECT_ID/sxlm:latest

# Submit training job
gcloud ai custom-jobs create \
    --region=$REGION \
    --display-name=sxlm-auto-training \
    --worker-pool-spec=machine-type=a2-highgpu-1g,replica-count=1,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,container-image-uri=gcr.io/$PROJECT_ID/sxlm:latest \
    --args="python,scripts/train_auto.py"

echo "Training job submitted. Monitor with:"
echo "gcloud ai custom-jobs list --region=$REGION"
