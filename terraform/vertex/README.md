# SXLM Vertex AI Deployment

## Prerequisites

1. GCP account with Vertex AI API enabled
2. gcloud CLI installed and authenticated
3. Docker installed
4. Terraform installed

## Setup

```bash
# Authenticate with GCP
gcloud auth login
gcloud auth configure-docker

# Set your project
export PROJECT_ID="your-project-id"
```

## Deploy

```bash
# Make deploy script executable
chmod +x deploy.sh

# Deploy to Vertex AI
./deploy.sh $PROJECT_ID us-central1
```

## Configuration

Edit `terraform/vertex/variables.tf` to change:
- Machine type (default: a2-highgpu-8g)
- GPU type (default: A100)
- GPU count (default: 8)

## Monitor

```bash
# View training job
gcloud ai custom-jobs list --region=us-central1

# View logs
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
```
