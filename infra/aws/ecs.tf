# ECR for the API image, and one Fargate task running two containers: the API and a
# Redis sidecar (answer cache, rate limits, daily spend), which share localhost. Redis in
# the task instead of ElastiCache saves about $12 a month; its data does not outlive the
# task, as with Render's free Key Value.

resource "aws_ecr_repository" "api" {
  name                 = "fingraph-api"
  image_tag_mutability = "MUTABLE"
  force_delete         = true # destroy removes the images too
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the last three images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 3
      }
      action = { type = "expire" }
    }]
  })
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/fingraph"
  retention_in_days = 7
}

data "aws_iam_policy_document" "ecs_tasks_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# Used by ECS itself to pull the image, write logs and read the two secrets.
resource "aws_iam_role" "execution" {
  name               = "fingraph-task-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

data "aws_iam_policy_document" "read_secrets" {
  statement {
    actions = ["secretsmanager:GetSecretValue"]
    resources = [
      aws_secretsmanager_secret.anthropic.arn,
      aws_secretsmanager_secret.database_url.arn,
    ]
  }
}

resource "aws_iam_role_policy" "read_secrets" {
  name   = "read-fingraph-secrets"
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.read_secrets.json
}

resource "aws_ecs_cluster" "main" {
  name = "fingraph"
}

resource "aws_ecs_task_definition" "api" {
  family                   = "fingraph-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn # bedrock.tf

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64" # what a laptop's docker build produces
  }

  container_definitions = jsonencode([
    {
      name              = "api"
      image             = "${aws_ecr_repository.api.repository_url}:${var.image_tag}"
      essential         = true
      memoryReservation = 768
      portMappings      = [{ containerPort = 8000, protocol = "tcp" }]
      environment = [
        { name = "REDIS_URL", value = "redis://localhost:6379/0" },
        { name = "DAILY_SPEND_CAP_USD", value = var.daily_spend_cap_usd },
        { name = "CORS_ORIGINS", value = var.cors_origins },
        { name = "FINGRAPH_CONCURRENCY", value = "4" },
        # Set by CloudFront (see cloudfront.tf); the API keys rate limits on its address.
        { name = "CLIENT_IP_HEADER", value = "CloudFront-Viewer-Address" },
        { name = "LLM_PROVIDER", value = var.llm_provider },
        { name = "AWS_REGION", value = var.region },
      ]
      secrets = [
        { name = "ANTHROPIC_API_KEY", valueFrom = aws_secretsmanager_secret.anthropic.arn },
        { name = "DATABASE_URL", valueFrom = aws_secretsmanager_secret.database_url.arn },
      ]
      dependsOn = [{ containerName = "redis", condition = "START" }]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.api.name
          awslogs-region        = var.region
          awslogs-stream-prefix = "api"
        }
      }
    },
    {
      name              = "redis"
      image             = "redis:7.4.1-alpine"
      essential         = true
      memoryReservation = 96
      # noeviction: evicting keys could reset the daily spend counter or rate limits.
      command = ["redis-server", "--maxmemory", "64mb", "--maxmemory-policy", "noeviction"]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.api.name
          awslogs-region        = var.region
          awslogs-stream-prefix = "redis"
        }
      }
    },
  ])
}

resource "aws_ecs_service" "api" {
  name                              = "fingraph-api"
  cluster                           = aws_ecs_cluster.main.id
  task_definition                   = aws_ecs_task_definition.api.arn
  desired_count                     = 1
  launch_type                       = "FARGATE"
  health_check_grace_period_seconds = 120

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.task.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }

  depends_on = [aws_lb_listener.http]
}
