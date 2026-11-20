# FinGraph on Amazon Bedrock

What was checked before any code was written, with sources and the date checked. Every
fact below was read on **21 November 2026** from the AWS page named next to it. Model
IDs, regions and prices change; re-check them before relying on this page.

## Is Claude Haiku 5.5 available on Bedrock?

**Yes.** Source: the [Claude Haiku 5.5 model card][card].

- Base model ID: `anthropic.claude-haiku-5-5`, generally available (launched
  7 October 2026, end of life no sooner than 7 October 2027).
- It is not offered in-region anywhere on the `bedrock-runtime` endpoint. It is reached
  only through cross-region inference profiles:
  - **global:** `global.anthropic.claude-haiku-5-5`;
  - **geo:** `us.`, `ca.`, `eu.`, `au.` and `jp.anthropic.claude-haiku-5-5`.
- There is no `apac.` profile.

## Is ap-south-1 (Mumbai) one of the regions?

**Only through the global profile.** The model card's availability table marks Mumbai
"Global: supported" and "In-Region: not supported, Geo: not supported", and Mumbai is not
a source region in any geo profile.

**FinGraph from Mumbai uses `global.anthropic.claude-haiku-5-5`** (the default in
`fingraph/bedrock.py`, overridable with `BEDROCK_MODEL_ID`). A global profile routes each
request to whichever commercial region has capacity, so prompts and outputs may be
processed outside India. That is fine for public TAT-QA data. It is not fine for data
that must stay in-country; AWS documents an IAM deny on `aws:RequestedRegion =
"unspecified"` that blocks global routing ([prerequisites][prereq]). The model card gives
no destination list for the global profile; `aws bedrock get-inference-profile` called
from the source region shows the current one.

## Model access: what the account has to do

Source: [Access Amazon Bedrock foundation models][access].

- **No access request.** In commercial regions there is nothing to request on the Model
  access page: "Amazon Bedrock starts the subscription for you on the first invocation".
  The first call subscribes the account to the model through AWS Marketplace.
- **Once per account, the first caller needs** `aws-marketplace:Subscribe` and
  `aws-marketplace:ViewSubscriptions`, plus a payment method valid for Marketplace
  purchases. After that, callers need only `bedrock:InvokeModel` ("These permissions are
  only required the first time a model is being used in an account"). The subscription
  can take up to about 15 minutes to settle.
- **No Anthropic use-case form for this model.** The First Time Use form is required
  only for "Anthropic models earlier than Claude Opus 4.7"; Haiku 5.5 is later.
- **Billing.** Charges appear on the AWS bill "under the model provider (not under Amazon
  Bedrock)", because the model is billed through AWS Marketplace ([model card][card]).
  The first invocation accepts the model's end-user licence agreement.
- **While the subscription is pending,** calls return `AccessDeniedException`.
  FinGraph reports that error with what to check (`fingraph/bedrock.py`, `_explain`).

So the steps are:

1. Sign in with an identity that has Bedrock and Marketplace permissions, such as an
   administrator SSO profile.
2. Set the budget alarm (see below).
3. Make one small call; the smoke run in the README does this.
4. From then on, use the least-privilege policy.

## Prices: Bedrock vs the direct API

Source: the Anthropic tab of [Amazon Bedrock pricing][pricing], on-demand, prompts up to
100K tokens, USD per million tokens. The page showed the same figures for US East (Ohio)
and Asia Pacific (Mumbai).

| | Input | Output | Cache write (5 min) | Cache write (1 h) | Cache read |
|---|---|---|---|---|---|
| Anthropic API (`fingraph/llm.py`) | $0.10 | $0.50 | $0.125 (1.25x) | not used | $0.01 (0.1x) |
| Bedrock, global profile | $0.10 | $0.50 | $0.125 | $0.20 | $0.01 |
| Bedrock, geo and in-region profiles | $0.11 | $0.55 | $0.1375 | $0.22 | $0.011 |

- **The global profile, the one Mumbai uses, costs the same as the direct API,** with
  the same cache multipliers. So `fingraph/llm.py` prices a Bedrock call with the same
  formula, looking up `BEDROCK_PRICES` by profile ID. `report.py` prints both tables
  (`price_tables`).
- **Geo profiles cost 10% more.**
- **Prompts over 100K tokens** are billed at a separate "Haiku 5.5 - Long Context" rate
  of $0.50 / $2.50. FinGraph's prompts are far below that.
- **Batch, Priority and Flex** tiers are not offered for this model; only Standard is.

## Prompt caching on Bedrock

**Supported for Haiku 5.5, and enabled the same way FinGraph enables it on the direct
API.** Sources: [prompt caching][caching] and the [model card][card].

- **Explicit caching is supported:**
  - minimum 512 tokens per cache checkpoint, at most 4 checkpoints per request;
  - 5-minute and 1-hour TTLs;
  - checkpoints allowed in `system`, `messages` and `tools`.
- **Automatic caching:** Haiku 5.5 is on the list of models that accept "one
  cache_control field at the top level of your request instead of placing cache
  checkpoints". On Converse it goes in `additionalModelRequestFields`:
  `{"cache_control": {"type": "ephemeral"}}`. **This is exactly what FinGraph sends on
  the direct API,** so the Bedrock path sends the same thing. Other models reject the
  field with a `ValidationException`.
- **Not supported:** "implicit" prompt caching (caching without any marker), and
  caching for batch inference.
- **The Converse response reports** `usage.cacheReadInputTokens` and
  `usage.cacheWriteInputTokens`, and its `inputTokens` "covers only non-cached tokens".
  These are the same meanings as the Anthropic API's `cache_read_input_tokens`,
  `cache_creation_input_tokens` and `input_tokens`, so the mapping is one to one.

## The effort setting

**Accepted, through `additionalModelRequestFields`.** Converse has no effort parameter of
its own. Source: [adaptive thinking][thinking].

- The request carries `"output_config": {"effort": "medium"}` there. AWS's page warns
  that effort must sit in `output_config`, not inside `thinking`, or the call fails with
  a 400.
- For Haiku 5.5 the model card lists the levels low, medium, high, xhigh and max,
  default medium (FinGraph's setting).
- Adaptive thinking is on by default when `thinking` is omitted. FinGraph omits it on
  both providers.

## How the request maps

| FinGraph (Anthropic Messages shape) | Converse request |
|---|---|
| `model="claude-haiku-5-5"` | `modelId="global.anthropic.claude-haiku-5-5"` |
| `system="..."` | `system=[{"text": "..."}]` |
| `tools=[{name, description, input_schema, strict}]` | `toolConfig.tools=[{"toolSpec": {name, description, "inputSchema": {"json": ...}, strict}}]` |
| `max_tokens=4096` | `inferenceConfig.maxTokens=4096` |
| `output_config={"effort": "medium"}` | `additionalModelRequestFields.output_config` |
| `cache_control={"type": "ephemeral"}` | `additionalModelRequestFields.cache_control` |
| user text / `tool_result` blocks | `{"text"}` / `{"toolResult": {toolUseId, content, status}}` |
| the previous assistant turn, unchanged | the Converse blocks exactly as Bedrock returned them |

| Converse response | What the arms read |
|---|---|
| `text`, `toolUse`, `reasoningContent` blocks | `.content` blocks of type `text`, `tool_use`, `thinking` |
| `stopReason` | `.stop_reason`; `content_filtered` and `guardrail_intervened` become `refusal` |
| `usage.inputTokens`, `outputTokens`, `cacheReadInputTokens`, `cacheWriteInputTokens` | `.usage.input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens` |

**Thinking blocks go back exactly as they came.** On Haiku 5.5 they are bound to the
conversation prefix and to the AWS account that produced them ([model card][card]).
So every block from a Bedrock response keeps its raw Converse form, and a later request
sends that form back unchanged. The tests check this with botocore's `Stubber`, which
validates every request against the Converse schema (`tests/test_bedrock.py`).

**Why Converse and not InvokeModel:** Converse expresses everything FinGraph sends. The
Claude-specific settings (effort, request-level caching) go through
`additionalModelRequestFields`, which exists for that purpose. So `invoke_model` is not
used.

## Errors

| Bedrock error | What FinGraph does |
|---|---|
| `ThrottlingException`, `ServiceUnavailableException`, `ModelNotReadyException`, `InternalServerException` | Retries with exponential backoff and full jitter (1 s base, 20 s cap, 6 attempts), logs each retry, and records the count in the trace (`retries` on each model call). |
| `AccessDeniedException` | Fails at once, saying to check the IAM policy and the model's Marketplace subscription. |
| `ValidationException` / `ResourceNotFoundException` about the model | Fails at once, saying the profile may not be offered in `AWS_REGION`. |

boto3's own retries are switched off, so every retry is counted.

## IAM: least privilege

`infra/aws/bedrock.tf` gives the ECS task role only `bedrock:InvokeModel` on:

- the inference profile, `arn:aws:bedrock:<region>:<account>:inference-profile/global.anthropic.claude-haiku-5-5`;
- the foundation model `anthropic.claude-haiku-5-5`, in the calling region and as the
  empty-region ARN AWS uses for global routing, only when reached through that profile
  (condition `bedrock:InferenceProfileArn`).

This follows AWS's global-profile example in the [inference profile
prerequisites][prereq]. Notes:

- The Converse API is authorised by `bedrock:InvokeModel`; there is no separate
  `bedrock:Converse` action ([Converse API reference][converse]).
- ConverseStream would need `bedrock:InvokeModelWithResponseStream`. FinGraph does not
  stream, so the policy leaves it out. It has no `bedrock:*`.

For local runs, use an SSO profile (`aws configure sso`) rather than access keys.

## Budget alarm before the first call

The Terraform stack creates its own (`infra/aws/budget.tf`). For local runs, without the
stack, create a small monthly cost budget once:

```powershell
aws budgets create-budget --account-id <your account id> --budget file://budget.json --notifications-with-subscribers file://notify.json
```

`budget.json` holds:

```json
{"BudgetName": "fingraph-bedrock", "BudgetLimit": {"Amount": "5", "Unit": "USD"}, "TimeUnit": "MONTHLY", "BudgetType": "COST"}
```

`notify.json` holds the following, with your own email address:

```json
[{"Notification": {"NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN", "Threshold": 50, "ThresholdType": "PERCENTAGE"}, "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "you@example.com"}]}]
```

Budgets see costs hours late, so the real-time guard is each run's `--max-cost`. The
budget is the backstop.

[card]: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-haiku-5-5.html
[access]: https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html
[prereq]: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-prereq.html
[pricing]: https://aws.amazon.com/bedrock/pricing/
[caching]: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
[thinking]: https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-adaptive-thinking.html
[converse]: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
