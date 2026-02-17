# Vertex AI Custom Training Job

resource "google_storage_bucket" "training_bucket" {
  name     = "${var.project_id}-sxlm-training"
  location = var.region
}

resource "google_vertex_ai_custom_job" "sxlm_training" {
  display_name = "sxlm-training"
  location     = var.region

  job_spec {
    worker_pool_specs {
      machine_spec {
        machine_type     = var.machine_type
        accelerator_type = var.accelerator_type
        accelerator_count = var.accelerator_count
      }

      replica_count = 1

      container_spec {
        image_uri = "gcr.io/${var.project_id}/sxlm-trainer:latest"
        command   = ["python", "scripts/train_simple.py"]
      }
    }
  }
}
