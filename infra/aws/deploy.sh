#!/usr/bin/env bash
# Deploy the FinGraph API to AWS, in the order that keeps spending visible:
#   1. the budget alarm, on its own;
#   2. the ECR repository, then the image pushed to it;
#   3. everything else (RDS, ECS, load balancer, CloudFront).
#
#   cd infra/aws
#   cp terraform.tfvars.example terraform.tfvars   # set alert_email
#   export TF_VAR_anthropic_api_key=...            # never written to a file
#   ./deploy.sh
#
# Needs: terraform >= 1.10, the AWS CLI signed in to your account, and Docker running.
# Tear down with: terraform destroy
set -euo pipefail
cd "$(dirname "$0")"

: "${TF_VAR_anthropic_api_key:?export TF_VAR_anthropic_api_key first}"
[ -f terraform.tfvars ] || { echo "copy terraform.tfvars.example to terraform.tfvars first"; exit 1; }

terraform init -input=false

echo "== 1/3 budget alarm"
terraform apply -input=false -target=aws_budgets_budget.monthly

echo "== 2/3 container registry and image"
terraform apply -input=false -target=aws_ecr_repository.api -target=aws_ecr_lifecycle_policy.api
repo=$(terraform output -raw ecr_repository_url)
region=$(terraform console <<< 'var.region' | tr -d '"')
aws ecr get-login-password --region "$region" | docker login --username AWS --password-stdin "${repo%%/*}"
docker build --platform linux/amd64 -t "$repo:latest" ../..
docker push "$repo:latest"

echo "== 3/3 database, service, load balancer, CloudFront (RDS takes 5-10 minutes)"
terraform apply -input=false

url=$(terraform output -raw api_url)
echo "API: $url  (CloudFront can take a few minutes more to answer)"
echo "Check: curl $url/healthz"
