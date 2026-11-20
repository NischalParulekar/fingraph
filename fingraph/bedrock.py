"""Amazon Bedrock behind the same interface as the Anthropic client (`LLM_PROVIDER=bedrock`).

`BedrockClient().messages.create(...)` takes the keyword arguments `fingraph.llm.LLM`
sends to the Anthropic SDK, sends them through Bedrock's Converse API, and returns an
object shaped like the SDK's Message: `.content` blocks (text, tool_use, thinking),
`.stop_reason` and `.usage` with the same four token counts. So the arms and the cost
accounting work unchanged. What maps to what, and why, is in docs/bedrock.md.

Credentials and region come from boto3's usual chain (an SSO profile via AWS_PROFILE, or
the ECS task role, plus AWS_REGION); nothing here reads a key.
"""

import logging
import random
import time
from types import SimpleNamespace

log = logging.getLogger(__name__)

# Haiku 5.5 is reachable from ap-south-1 (Mumbai) only through the global cross-region
# inference profile; see docs/bedrock.md (checked 21 November 2026).
MODEL_IDS = {"claude-haiku-5-5": "global.anthropic.claude-haiku-5-5"}

# Retried with exponential backoff and full jitter; anything else is raised at once.
RETRYABLE = {
    "ThrottlingException",
    "ServiceUnavailableException",
    "ModelNotReadyException",
    "InternalServerException",
}
MAX_ATTEMPTS = 6
BASE_DELAY_S, MAX_DELAY_S = 1.0, 20.0

# Converse stop reasons that have no Anthropic equivalent. A refusal shows up on Bedrock
# as content_filtered; the arms treat "refusal" as "stop, no answer".
STOP_REASONS = {"content_filtered": "refusal", "guardrail_intervened": "refusal"}


class BedrockError(RuntimeError):
    """A Bedrock error that retrying will not fix, with what to check."""


class BedrockClient:
    def __init__(self, model_id: str | None = None, client=None, sleep=time.sleep):
        if client is None:
            import boto3
            from botocore.config import Config

            # boto3's own retries are off so that every retry is ours: logged and counted.
            config = Config(read_timeout=300, retries={"max_attempts": 1, "mode": "standard"})
            client = boto3.client("bedrock-runtime", config=config)
        self.model_id = model_id
        self._client = client
        self._sleep = sleep
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, *, model, max_tokens, system, tools, messages, output_config, cache_control):
        request = {
            "modelId": self.model_id or MODEL_IDS[model],
            "system": [{"text": system}],
            "messages": [_message(m) for m in messages],
            "inferenceConfig": {"maxTokens": max_tokens},
            "additionalModelRequestFields": {
                "output_config": output_config,
                "cache_control": cache_control,
            },
        }
        if tools:
            request["toolConfig"] = {"tools": [_tool(t) for t in tools]}
        raw, retries = self._call(request)
        return _response(raw, retries)

    def _call(self, request: dict) -> tuple[dict, int]:
        from botocore.exceptions import ClientError

        for attempt in range(MAX_ATTEMPTS):
            try:
                return self._client.converse(**request), attempt
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                message = e.response.get("Error", {}).get("Message", "")
                if code in RETRYABLE and attempt + 1 < MAX_ATTEMPTS:
                    delay = random.uniform(0, min(MAX_DELAY_S, BASE_DELAY_S * 2**attempt))
                    log.warning("bedrock %s, retry %d in %.1fs", code, attempt + 1, delay)
                    self._sleep(delay)
                    continue
                raise _explain(code, message, request["modelId"]) from e
        raise AssertionError("unreachable")


def _explain(code: str, message: str, model_id: str) -> Exception:
    if code == "AccessDeniedException":
        hint = (
            "the caller's IAM policy must allow bedrock:InvokeModel on the inference profile "
            "and its foundation model (infra/aws/bedrock.tf), and the model must be enabled "
            "for the account: the first call subscribes to it through AWS Marketplace, which "
            "needs aws-marketplace:Subscribe once (docs/bedrock.md)"
        )
    elif code in ("ResourceNotFoundException", "ValidationException") and "model" in message:
        hint = f"{model_id} may not be offered in this region; check AWS_REGION"
    elif code in RETRYABLE:
        hint = f"still failing after {MAX_ATTEMPTS} attempts"
    else:
        return BedrockError(f"{code}: {message}")
    return BedrockError(f"{code}: {message} ({hint})")


# --- request mapping -----------------------------------------------------------------


def _tool(tool: dict) -> dict:
    spec = {
        "name": tool["name"],
        "description": tool["description"],
        "inputSchema": {"json": tool["input_schema"]},
    }
    if "strict" in tool:
        spec["strict"] = tool["strict"]
    return {"toolSpec": spec}


def _message(message: dict) -> dict:
    content = message["content"]
    if isinstance(content, str):
        return {"role": message["role"], "content": [{"text": content}]}
    return {"role": message["role"], "content": [_block(b) for b in content]}


def _block(block) -> dict:
    # A block from an earlier Bedrock response goes back exactly as it came: thinking
    # blocks carry a signature that must be returned unchanged.
    raw = getattr(block, "raw", None)
    if raw is not None:
        return raw
    if block["type"] == "text":
        return {"text": block["text"]}
    if block["type"] == "tool_result":
        return {
            "toolResult": {
                "toolUseId": block["tool_use_id"],
                "content": [{"text": block["content"]}],
                "status": "error" if block.get("is_error") else "success",
            }
        }
    raise ValueError(f"no Converse mapping for a {block['type']} block")


# --- response mapping ----------------------------------------------------------------


def _response(raw: dict, retries: int) -> SimpleNamespace:
    usage = raw["usage"]
    return SimpleNamespace(
        content=[_content(b) for b in raw["output"]["message"]["content"]],
        stop_reason=STOP_REASONS.get(raw["stopReason"], raw["stopReason"]),
        # Converse's inputTokens, like the Anthropic API's input_tokens, counts only the
        # tokens that were neither read from nor written to the cache.
        usage=SimpleNamespace(
            input_tokens=usage["inputTokens"],
            output_tokens=usage["outputTokens"],
            cache_read_input_tokens=usage.get("cacheReadInputTokens", 0),
            cache_creation_input_tokens=usage.get("cacheWriteInputTokens", 0),
        ),
        latency_ms=raw.get("metrics", {}).get("latencyMs"),
        retries=retries,
    )


def _content(block: dict) -> SimpleNamespace:
    if "text" in block:
        return SimpleNamespace(type="text", text=block["text"], raw=block)
    if "toolUse" in block:
        use = block["toolUse"]
        return SimpleNamespace(
            type="tool_use", id=use["toolUseId"], name=use["name"], input=use["input"], raw=block
        )
    if "reasoningContent" in block:
        reasoning = block["reasoningContent"].get("reasoningText", {})
        return SimpleNamespace(type="thinking", thinking=reasoning.get("text", ""), raw=block)
    return SimpleNamespace(type=next(iter(block)), raw=block)
