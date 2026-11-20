variable "region" {
  description = "AWS region. us-east-1 is where the cost estimate in README.md was made."
  type        = string
  default     = "us-east-1"
}

variable "alert_email" {
  description = "Where the budget alerts go. Kept in the git-ignored terraform.tfvars."
  type        = string
}

variable "monthly_budget_usd" {
  description = "AWS Budgets alerts at 50% and 100% of this (actual), and 100% (forecast)."
  type        = number
  default     = 10
}

variable "anthropic_api_key" {
  description = "Pass as TF_VAR_anthropic_api_key, never in a file. Stored in Secrets Manager."
  type        = string
  sensitive   = true
}

variable "daily_spend_cap_usd" {
  description = "The API's own daily cap on model spend."
  type        = string
  default     = "0.50"
}

variable "cors_origins" {
  description = "Comma-separated origins allowed to call the API from a browser."
  type        = string
  default     = "https://nischalparulekar.github.io"
}

variable "image_tag" {
  description = "Tag of the API image in ECR."
  type        = string
  default     = "latest"
}
