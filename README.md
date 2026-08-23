# @kuanfu0430/dsh-compaction-anchored

Contract-checked anchor-preserving compaction backend for DeepSeek Harness.

This package is a compatibility fork of `@deepseek-ai/dsh-compaction-basic`. It keeps the first durable append-origin, non-checkpoint `user/message` and the latest model-visible message complete, summarizes only the continuous middle, and makes pressure, canonical overflow, manual `/compact`, and `compactRegion()` use one planner and transaction path.

## Hard invariants

- **HEAD:** the first durable append-origin `user/message` that is not a compaction checkpoint is permanent. Its role and every content block are preserved byte-for-JSON-value; rich image blocks are never sent to the summarizer as editable output.
- **LAST / TAIL:** the latest model-visible message and the current or most recent complete turn remain outside summary replacements. Tool-call/result boundaries are never split.
- **MIDDLE only:** one continuous, tool-balanced range between HEAD and protected TAIL may be replaced.
- **Legacy recovery:** a previously shadowed HEAD is reconstructed from the append-only log inside a deterministic version-1 envelope. Unknown or malformed envelopes fail before prune or summary side effects.
- **Strict shrink:** the complete framed checkpoint, including an embedded rich HEAD, must estimate smaller than the complete shadowed range.
- **Fail closed:** if protected context cannot fit the routed model window, the backend does not delete or truncate either anchor.

Selective tool-result pruning is model-free and durable. Normal pressure may prune only MIDDLE results; canonical provider overflow may additionally prune non-LAST tool results in TAIL. Every prune pass is followed by a fresh measurement and final plan.

## Compatibility

| Component | Supported version |
| --- | --- |
| DeepSeek Harness peer range | `>=0.1.0-rc.7 <0.2.0` |
| Latest contract-tested version | `0.1.1-rc.2` |
| Upstream source baseline | `99f6f02fecdb7dff40c3fbc9470f5907c29f74ca` |
| Node.js | `^22.19.0 || >=24.0.0` |

The peer range is deliberately bounded to the current pre-1.0 API family. Every newly adopted Harness release must rerun the anchor, transaction, fault, cold-restore, trigger, and runtime suites, and the plugin-library installer must execute its backend contract probe before restart. See `UPSTREAM.md`.

## Installation

The only supported installation form is a dependency alias under the package key expected by built-in presets. Pin an exact release commit:

```json
{
  "dependencies": {
    "@deepseek-ai/dsh-compaction-basic": "github:kuanfu0430/dsh-compaction-anchored#<exact-commit>"
  }
}
```

Do **not** install the true package name beside the official backend, and do not add a host-plane `dsh.bundle.patch`. Preset rows already importing `@deepseek-ai/dsh-compaction-basic` must resolve this alias as their sole `ctx.compaction` provider.

After installation, verify:

1. the profile dependency key and exact spec;
2. `pnpm why @deepseek-ai/dsh-compaction-basic` resolves this package;
3. standard, code, cordis, and active custom presets expose `backendIdentity.name === '@kuanfu0430/dsh-compaction-anchored'`;
4. there is one automatic pressure listener and one canonical `agent/request-error` listener per preset context.

## Optional control plane

The backend reads an optional host-plane service:

```js
const value = ctx.get('kuanfuCompactionPolicy')?.resolve(session)
// frozen { mode: 'percent' | 'tokens', percent: 30..80, tokens: positiveInteger } | null
```

`null`, a missing service, or an invalid value uses backend defaults. The backend revalidates all values and caps effective pressure at 80% of the routed model window. The policy affects automatic pressure only; it cannot disable canonical overflow, manual compaction, or direct range validation.

## Checkpoint summary

The summarizer receives HEAD as read-only authority plus MIDDLE history and must output exactly these populated sections in order:

1. `## Original Goal Amendments`
2. `## Non-negotiable Requirements`
3. `## Decisions and Rationale`
4. `## Completed Work and Evidence`
5. `## Current State`
6. `## Open Issues and Risks`
7. `## Exact Next Step`

Only text output is accepted. Blank, truncated, oversized, reordered, or extra-section output does not create a summary replacement.

## Development

```sh
pnpm install --frozen-lockfile
pnpm check
```

The test suite covers repeated and legacy compaction, rich anchors, open tool loops, error results, direct ranges, selective-prune partial failures, append/flush faults, cancellation, concurrency, context-fit rechecks, cold restore, policy pressure, canonical overflow retry proof, and a ten-checkpoint continuity fixture.

## License

MIT. Fork attribution and upstream synchronization details are in `LICENSE` and `UPSTREAM.md`.
