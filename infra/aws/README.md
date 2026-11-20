# FinGraph on AWS (temporary)

Minimal Terraform for the API on AWS: ECS Fargate behind an Application Load Balancer and
CloudFront (HTTPS), RDS PostgreSQL 16 for LangGraph's checkpoints, Secrets Manager for the
Anthropic key and the database URL, and an AWS Budgets alarm. It is meant to run for a few
days to record the demo, then be destroyed. The always-on demo is the Render one
(`render.yaml`).

```
        HTTPS                HTTP (CloudFront addresses only)
viewer ──────► CloudFront ──────────────────► ALB ──► Fargate task ──► RDS Postgres 16
                                                      ├ api  (port 8000)
                                                      └ redis (sidecar, localhost)
```

## Cost estimate (us-east-1, on-demand, left running)

| Item | Basis | Per month |
|---|---|---|
| Fargate task, 0.5 vCPU + 1 GB, x86 | $0.04048 per vCPU-hour, $0.004445 per GB-hour | $18.0 |
| RDS db.t4g.micro + 20 GB gp3 | $0.016 per hour; storage about $0.115 per GB-month | $14.0 |
| Application Load Balancer | about $0.0225 per hour plus capacity units | $16.5 |
| Public IPv4 addresses: task + 2 for the load balancer | $0.005 per address-hour | $11.0 |
| Secrets Manager, 2 secrets | $0.40 each | $0.8 |
| CloudFront, ECR, CloudWatch Logs | demo traffic, 1.4 GB image, 7-day logs | under $1 |
| **Total** | | **about $60 a month, $2 a day** |

Recording the demo over two or three days should cost about $5, plus the model's own
spend (capped by `daily_spend_cap_usd` and by the Anthropic Console's workspace limit).
The Fargate and RDS rates come from AWS's pricing page and third-party price lists; the
load balancer, storage and IPv4 figures are from memory. Check them on the AWS pricing
pages before applying. No NAT gateway (about $33 a month) and no ElastiCache (about $12):
the task sits in the default VPC's public subnets, reachable only from the load balancer,
and Redis runs beside the API in the same task.

## Deploy

1. Have the AWS CLI signed in to your account, Terraform 1.10 or newer, and Docker running.
2. `cp terraform.tfvars.example terraform.tfvars` and set `alert_email`.
3. `export TF_VAR_anthropic_api_key=...` (never write it to a file).
4. `./deploy.sh` (or `make aws-up` from the repository root). It applies the budget alarm
   first and on its own, then the container registry, pushes the image, then the rest.
   Confirm the AWS Budgets email when it arrives.
5. `curl $(terraform output -raw api_url)/healthz`.

**With Bedrock instead of the Anthropic API:** set `llm_provider = "bedrock"` in
`terraform.tfvars`. The task role (`bedrock.tf`) then gets `bedrock:InvokeModel` on the
global Claude Haiku 5.5 inference profile and on its model when reached through it, and
nothing else. The API calls Bedrock with that role; there is no key.

- `region` can stay `us-east-1` or be `ap-south-1`. Both reach Haiku 5.5 through the
  global profile ([docs/bedrock.md](../../docs/bedrock.md)).
- The first Bedrock call in an account subscribes it to the model through AWS
  Marketplace. Make that call once with your own administrator profile before deploying;
  the task role cannot subscribe.
- `TF_VAR_anthropic_api_key` is still required by the variables, but is not used in
  this mode.

## Tear down

`terraform destroy` (or `make aws-down`). Nothing here keeps data worth saving: the
database has no backups and no final snapshot, the secrets have no recovery window, and
the registry is deleted with its images.

## Limits worth knowing

- **Between CloudFront and the load balancer the traffic is HTTP**, since there is no
  custom domain for an ACM certificate. The load balancer only accepts CloudFront's
  origin addresses (an AWS-managed prefix list).
- **CloudFront waits at most 60 seconds for an answer**, without a quota increase. Arm C
  with retries can come close; a timeout shows as a 504.
- **The Terraform state holds the database password and the Anthropic key.** It stays
  local and git-ignored; do not commit it or move it to a shared backend without
  encryption.
- **AWS Budgets lags by up to a day.** It is a backstop; tearing down is the real limit.
