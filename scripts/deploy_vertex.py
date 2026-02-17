"""Deploy training to Vertex AI"""

from google.cloud import aiplatform

def deploy_vertex_training(
    project_id: str,
    location: str = "us-central1",
    machine_type: str = "a2-highgpu-8g",
    accelerator_type: str = "NVIDIA_TESLA_A100",
    accelerator_count: int = 8
):
    """Deploy SXLM training to Vertex AI"""

    aiplatform.init(project=project_id, location=location)

    job = aiplatform.CustomContainerTrainingJob(
        display_name="sxlm-training",
        container_uri="gcr.io/{}/sxlm-trainer:latest".format(project_id),
    )

    job.run(
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        replica_count=1,
    )

    return job

if __name__ == "__main__":
    import sys
    project_id = sys.argv[1] if len(sys.argv) > 1 else "your-project-id"
    deploy_vertex_training(project_id)
