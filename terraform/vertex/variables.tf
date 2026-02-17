# Vertex AI Training Deployment for SXLM

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "machine_type" {
  description = "Machine type for training"
  type        = string
  default     = "a2-highgpu-8g"
}

variable "accelerator_type" {
  description = "GPU accelerator type"
  type        = string
  default     = "NVIDIA_TESLA_A100"
}

variable "accelerator_count" {
  description = "Number of GPUs"
  type        = number
  default     = 8
}
