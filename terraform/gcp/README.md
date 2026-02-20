# Quila GCP Deployment

Terraform configuration for deploying Quila to Google Cloud Platform.

## Prerequisites

- Terraform >= 1.0
- GCP account with GPU quota
- gcloud CLI configured

## Usage

1. Copy example variables:
```bash
cp terraform.tfvars.example terraform.tfvars
```

2. Edit `terraform.tfvars` with your GCP project ID

3. Initialize and apply:
```bash
terraform init
terraform plan
terraform apply
```

## Resources Created

- GPU compute instance (A100)
- VPC network and subnet
- Firewall rules (ports 8000, 8001)
- Storage bucket for models/data

## Outputs

- `instance_ip`: External IP address
- `api_endpoint`: API endpoint URL
