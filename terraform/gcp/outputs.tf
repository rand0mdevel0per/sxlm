output "instance_ip" {
  description = "External IP of Quila GPU instance"
  value       = google_compute_instance.quila_gpu.network_interface[0].access_config[0].nat_ip
}

output "instance_name" {
  description = "Name of Quila GPU instance"
  value       = google_compute_instance.quila_gpu.name
}

output "storage_bucket" {
  description = "Storage bucket name"
  value       = google_storage_bucket.quila_storage.name
}

output "api_endpoint" {
  description = "API endpoint URL"
  value       = "http://${google_compute_instance.quila_gpu.network_interface[0].access_config[0].nat_ip}:8000"
}
