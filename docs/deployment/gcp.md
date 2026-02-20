# GCP Deployment

## Prerequisites

- GCP account with GPU quota
- Terraform >= 1.0
- gcloud CLI configured

## Quick Deploy

```bash
cd terraform/gcp

# Copy and edit variables
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your project ID

# Deploy
terraform init
terraform plan
terraform apply
```

## Resources Created

- **GPU Instance**: A100 GPU compute instance
- **VPC Network**: Isolated network for Quila
- **Firewall Rules**: Ports 8000, 8001 open
- **Storage Bucket**: Model checkpoints and data

## Configuration

Edit `terraform.tfvars`:

```hcl
project_id   = "your-gcp-project-id"
region       = "us-central1"
zone         = "us-central1-a"
machine_type = "a2-highgpu-1g"
gpu_type     = "nvidia-tesla-a100"
gpu_count    = 1
```

## Access

After deployment:

```bash
# Get instance IP
terraform output instance_ip

# Access API
curl http://<instance_ip>:8000/status
```

## Cleanup

```bash
terraform destroy
```
