# The task role: what the API's own code may do in AWS. With llm_provider = "bedrock" it
# may call Claude Haiku 5.5 through the global cross-region inference profile, and
# nothing else (no bedrock:*). With "anthropic" (the default) it has no permissions.
#
# The Converse API is authorised by bedrock:InvokeModel; there is no separate
# bedrock:Converse action. FinGraph does not stream, so InvokeModelWithResponseStream
# (ConverseStream) is not granted. A cross-region profile needs the profile itself and
# the foundation model it routes to, the latter only when reached through that profile
# (the bedrock:InferenceProfileArn condition). The empty-region model ARN is how AWS's
# example writes the global profile's destinations. See docs/bedrock.md.

data "aws_caller_identity" "current" {}

locals {
  bedrock_profile_arn = "arn:aws:bedrock:${var.region}:${data.aws_caller_identity.current.account_id}:inference-profile/${var.bedrock_model_id}"
  bedrock_model_name  = trimprefix(var.bedrock_model_id, "global.")
}

resource "aws_iam_role" "task" {
  name               = "fingraph-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

data "aws_iam_policy_document" "invoke_bedrock" {
  statement {
    sid       = "InferenceProfile"
    actions   = ["bedrock:InvokeModel"]
    resources = [local.bedrock_profile_arn]
  }
  statement {
    sid     = "ModelThroughTheProfile"
    actions = ["bedrock:InvokeModel"]
    resources = [
      "arn:aws:bedrock:${var.region}::foundation-model/${local.bedrock_model_name}",
      "arn:aws:bedrock:::foundation-model/${local.bedrock_model_name}",
    ]
    condition {
      test     = "StringLike"
      variable = "bedrock:InferenceProfileArn"
      values   = [local.bedrock_profile_arn]
    }
  }
}

resource "aws_iam_role_policy" "invoke_bedrock" {
  count  = var.llm_provider == "bedrock" ? 1 : 0
  name   = "invoke-claude-haiku-5-5"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.invoke_bedrock.json
}
