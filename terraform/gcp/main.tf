terraform {
  required_version = ">= 1.0"
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
  zone    = var.zone
}

# GPU compute instance for Quila inference/training
resource "google_compute_instance" "quila_gpu" {
  name         = "quila-gpu-${var.environment}"
  machine_type = var.machine_type
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 200
      type  = "pd-ssd"
    }
  }

  guest_accelerator {
    type  = var.gpu_type
    count = var.gpu_count
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = false
  }

  network_interface {
    network = google_compute_network.quila_vpc.id
    access_config {}
  }

  metadata_startup_script = file("${path.module}/startup.sh")

  tags = ["quila-gpu", "http-server", "https-server"]
}

# VPC network
resource "google_compute_network" "quila_vpc" {
  name                    = "quila-vpc-${var.environment}"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "quila_subnet" {
  name          = "quila-subnet-${var.environment}"
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.quila_vpc.id
}

# Firewall rules
resource "google_compute_firewall" "quila_api" {
  name    = "quila-api-${var.environment}"
  network = google_compute_network.quila_vpc.id

  allow {
    protocol = "tcp"
    ports    = ["8000", "8001"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["quila-gpu"]
}

# Storage bucket for model checkpoints and data
resource "google_storage_bucket" "quila_storage" {
  name          = "quila-storage-${var.project_id}-${var.environment}"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
}
