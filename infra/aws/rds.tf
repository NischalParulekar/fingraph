# Postgres 16 for LangGraph's checkpoints (pgvector is available on RDS if corpus-mode
# retrieval is wanted later). Smallest Graviton instance, no backups, no final snapshot:
# it holds paused review runs only, and is destroyed after the demo.
resource "random_password" "db" {
  length  = 32
  special = false # RDS refuses / @ " and spaces; letters and digits are enough
}

resource "aws_db_subnet_group" "default" {
  name       = "fingraph"
  subnet_ids = data.aws_subnets.default.ids
}

resource "aws_db_instance" "db" {
  identifier              = "fingraph"
  engine                  = "postgres"
  engine_version          = "16"
  instance_class          = "db.t4g.micro"
  allocated_storage       = 20
  storage_type            = "gp3"
  db_name                 = "fingraph"
  username                = "fingraph"
  password                = random_password.db.result
  db_subnet_group_name    = aws_db_subnet_group.default.name
  vpc_security_group_ids  = [aws_security_group.db.id]
  publicly_accessible     = false
  backup_retention_period = 0
  skip_final_snapshot     = true
  deletion_protection     = false
  apply_immediately       = true
}
