# Upstream

This repository is a focused compatibility fork of DeepSeek Harness compaction-basic.

- Upstream repository: <https://github.com/deepseek-ai/deepseek-harness>
- Baseline commit: `99f6f02fecdb7dff40c3fbc9470f5907c29f74ca`
- Baseline release: `0.1.0-rc.7`
- Baseline package: `packages/compaction/compaction-basic`
- Related contracts copied or intentionally retained:
  - `packages/compaction/compaction`
  - `packages/compaction/compaction-tool-result-pruner`
  - `packages/core/session`
  - `@deepseek-ai/dsh-llm` and `@deepseek-ai/dsh-token-meter`

## Retained upstream behavior

- `CompactionEngine` public seam and `BasicCompactionEngine` compatibility export.
- Cordis config schema and optional automatic listener lifecycle.
- Routed model resolution and replay-aware `ctx.llm.stream({ purpose: 'compaction' })`.
- Durable bracket order: `compaction/start` → `compaction/summary` → replacement `user/message` → `compaction/end`.
- Manual maintenance admission, source command correlation, cancellation, flush, selected-span/whole-surface stability, shadow pricing, and canonical overflow retry protocol.
- Official tool-pair boundary predicates and tool-result `pruneContent()` service.

## Intentional differences

- The first durable append-origin non-checkpoint user message is a permanent HEAD anchor.
- The latest message plus current/recent complete turn form a protected TAIL.
- Only the continuous MIDDLE can be summarized; arbitrary `compactRegion()` subranges are rejected.
- Legacy official checkpoints recover HEAD through a deterministic version-1 envelope.
- Summary output has seven mandatory goal-continuity sections and is text-only.
- Selective pruning is protected-range aware and followed by a fresh final plan.
- The backend owns both pressure and canonical overflow listeners; host policy plugins cannot trigger compaction.
- Effective pressure is capped at 80% of the routed context window.

## Updating

For every upstream revision:

1. Diff compaction engine, region transaction, compaction/session invariants, tool pairing, token meter, pruner, agent event payloads, and preset loader behavior from the baseline commit.
2. Keep this package's `@deepseek-ai/dsh-*` peers exact and update `BACKEND_IDENTITY.harness` only after compatibility work.
3. Rerun `pnpm check`, the full fault suite, a frozen clean install, preset identity checks, real pressure/manual/overflow conversations, cold restore, restart, uninstall, and rollback.
4. Never merge an upstream change that permits HEAD, LAST, or required TAIL messages to enter a replacement or prune range.

The upstream MIT notice is retained in `LICENSE`.
