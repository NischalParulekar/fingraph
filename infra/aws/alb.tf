# The load balancer CloudFront talks to. Plain HTTP, reachable from CloudFront's origin
# addresses only; viewers get HTTPS from CloudFront (no custom domain, so no ACM
# certificate for the load balancer itself).
resource "aws_lb" "api" {
  name               = "fingraph"
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  # Two subnets (the minimum; default subnets are one per zone): each one gives the
  # load balancer a public IPv4 address, and AWS bills every one, about $3.65 a month.
  subnets      = slice(sort(data.aws_subnets.default.ids), 0, 2)
  idle_timeout = 150 # an answer can take a while; CloudFront waits at most 60 s
}

resource "aws_lb_target_group" "api" {
  name                 = "fingraph-api"
  port                 = 8000
  protocol             = "HTTP"
  target_type          = "ip"
  vpc_id               = data.aws_vpc.default.id
  deregistration_delay = 30

  health_check {
    path                = "/healthz"
    matcher             = "200"
    interval            = 15
    healthy_threshold   = 2
    unhealthy_threshold = 4
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.api.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}
