import { isDeepStrictEqual } from 'node:util'
import { freezeMessage } from '@deepseek-ai/dsh-llm'

/**
 * Land replay-safe, protected-range-aware tool-result replacements. Every
 * marker/replacement pair is synchronous; earlier pairs remain durable if a
 * later candidate fails.
 */
export function selectivePrune(ctx, session, plan, trigger, signal) {
  const pruner = ctx.get('toolResultPruner')
  if (pruner === undefined || typeof pruner.pruneContent !== 'function') {
    return Object.freeze({ pruned: [], charsRemoved: 0 })
  }
  if (plan.migration) return Object.freeze({ pruned: [], charsRemoved: 0 })

  const nodes = [...session.surface.nodes]
  const candidateIndexes = []
  for (let index = plan.startIdx; index <= plan.endIdx; index += 1) candidateIndexes.push(index)
  if (trigger === 'context-overflow') {
    for (let index = plan.tailStartIdx; index < nodes.length; index += 1) candidateIndexes.push(index)
  }
  const excluded = new Set([
    plan.anchor.headSeq,
    plan.anchor.anchorNodeSeq,
    plan.hardLastSeq,
    nodes.at(-1),
    plan.requestedRange?.start,
    plan.requestedRange?.end,
  ].filter((value) => value !== undefined))

  const candidates = [...new Set(candidateIndexes)]
    .map((index) => nodes[index])
    .filter((seq) => !excluded.has(seq))
    .map((seq) => ({ seq, event: session.events[seq] }))
    .filter(({ event }) => event?.type === 'tool/result')

  const pruned = []
  let charsRemoved = 0
  for (const { seq, event } of candidates) {
    signal?.throwIfAborted()
    if (!session.surface.nodes.includes(seq)) continue
    const result = event.data.message.content[0]
    if (result === undefined || !Array.isArray(result.content)) {
      throw new Error(`anchored compaction: tool/result seq ${seq} has no pruneable result content`)
    }
    const content = pruner.pruneContent(result.content)
    if (content === null) continue
    if (!Array.isArray(content)) {
      throw new Error(`anchored compaction: tool-result pruner returned invalid content for seq ${seq}`)
    }
    const charsBefore = measureChars(pruner, result.content)
    const charsAfter = measureChars(pruner, content)
    if (charsAfter >= charsBefore || isDeepStrictEqual(content, result.content)) {
      throw new Error(
        `anchored compaction: tool-result prune for seq ${seq} must be strictly smaller before committing`,
      )
    }
    const message = freezeMessage({
      ...event.data.message,
      content: [{ ...result, content }],
    })
    const shadowedTokenCount = ctx.tokenMeter.estimateMessage(event.data.message)
    session.append('compaction/prune', {
      shadowedRange: { start: seq, end: seq },
      shadowedSeqs: [seq],
      shadowedTokenCount,
    })
    const replacement = session.append('tool/result', {
      ...event.data,
      message,
    }, {
      surfaceOp: { op: 'replace', start: seq, end: seq },
      sourceEventSeqs: [seq],
    })
    pruned.push({
      originalSeq: seq,
      replacementSeq: replacement.seq,
      callId: event.data.message.source.callId,
      charsBefore,
      charsAfter,
    })
    charsRemoved += charsBefore - charsAfter
  }
  return Object.freeze({ pruned, charsRemoved })
}

function measureChars(pruner, content) {
  if (typeof pruner.measureContent === 'function') return pruner.measureContent(content)
  let total = 0
  for (const block of content) {
    if (block.type === 'text') total += Array.from(block.text).length
  }
  return total
}
