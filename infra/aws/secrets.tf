# Both secrets reach the container through the task definition's `secrets`, never as
# plain environment variables. recovery_window_in_days = 0 lets a destroy-then-apply
# reuse the names at once. Their values are also in the Terraform state, which is why
# the state stays local and git-ignored.
resource "aws_secretsmanager_secret" "anthropic" {
  name                    = "fingraph/anthropic-api-key"
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "anthropic" {
  secret_id     = aws_secretsmanager_secret.anthropic.id
  secret_string = var.anthropic_api_key
}

resource "aws_secretsmanager_secret" "database_url" {
  name                    = "fingraph/database-url"
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "database_url" {
  secret_id = aws_secretsmanager_secret.database_url.id
  secret_string = format(
    "postgresql://%s:%s@%s:5432/%s?sslmode=require",
    aws_db_instance.db.username,
    random_password.db.result,
    aws_db_instance.db.address,
    aws_db_instance.db.db_name,
  )
}
