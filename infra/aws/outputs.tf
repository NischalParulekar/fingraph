output "api_url" {
  description = "The API over HTTPS. Try <api_url>/healthz."
  value       = "https://${aws_cloudfront_distribution.api.domain_name}"
}

output "ecr_repository_url" {
  description = "Push the API image here (deploy.sh does)."
  value       = aws_ecr_repository.api.repository_url
}

output "log_group" {
  value = aws_cloudwatch_log_group.api.name
}
