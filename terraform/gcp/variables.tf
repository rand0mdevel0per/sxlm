variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "prod"
}

variable "machine_type" {
  description = "Machine type for GPU instance"
  type        = string
  default     = "a2-highgpu-1g"
}

variable "gpu_type" {
  description = "GPU type"
  type        = string
  default     = "nvidia-tesla-a100"
}

variable "gpu_count" {
  description = "Number of GPUs"
  type        = number
  default     = 1
}
