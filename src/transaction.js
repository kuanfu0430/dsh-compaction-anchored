import { randomUUID } from 'node:crypto'
import { isDeepStrictEqual } from 'node:util'
import {
  CompactionId,
  ManualCompactionError,
  compactCheckpointSource,
  toolPairingBalancedAfter,
  toolPairingBalancedBefore,
} from '@deepseek-ai/dsh-compaction'
import { createUserMessage, errorChain } from '@deepseek-ai/dsh-llm'
import {
  assertAnchorProjection,
  classifyAnchor,
  frameAnchorEnvelope,
  frameNativeSummary,
  fullMessageSha256,
  lastVisibleSurfaceMessage,
} from './anchor.js'
import { buildSummarizationInput, validateSummary } from './summarizer.js'

export class SurfaceChangedError extends Error {
  constructor(message, options) {
    super(message, options)
    this.name = 'SurfaceChangedError'
  }
}

export function assertNoActiveCompaction(session, stage) {
  const entry = inspectCompactionEntryState(session.events)
  if (entry.unmatchedCompactionStart === undefined
    || (entry.latestEndSeedSeq !== undefined
      && entry.latestEndSeedSeq > entry.unmatchedCompactionStart.seq)) return
  throw new ManualCompactionError(
    'busy',
    `${stage}: compaction already in progress; the session compaction lock is active`,
  )
}

export async function executeCompaction(dependencies, session, plan, agent, options, signal) {
  signal?.throwIfAborted()
  validatePlanSelection(session, plan)
  const entry = inspectCompactionEntryState(session.events)
  assertNoActiveCompaction(session, 'anchored compaction')

  let owner
  if (options.owner === null) {
    if (entry.openTurn !== null) {
      throw new ManualCompactionError('busy', 'manual compaction: the session already has an open turn')
    }
    owner = null
  } else {
    if (entry.openTurn === null) {
      throw new Error('compactRegion: no open turn — automatic compaction must be enclosed in a turn')
    }
    owner = entry.openTurn
  }

  const compactionId = CompactionId(randomUUID())
  const lifecycle = {
    compactionId,
    ...options.sourceCommandId === undefined ? {} : { sourceCommandId: options.sourceCommandId },
    turn: owner,
  }
  signal?.throwIfAborted()
  const startEvent = session.append('compaction/start', lifecycle)
  let failure
  let flushFailure
  let result
  let closed = false
  let closing = false
  let stage = 'summary'

  try {
    const prepared = prepareCurrentSelection(dependencies.meter, session, plan)
    const summarized = await summarizeUntilValid(
      dependencies,
      session,
      prepared,
      agent,
      compactionId,
      options.sourceCommandId,
      options.contextWindow,
      options.projectedExtraTokens ?? 0,
      signal,
    )
    signal?.throwIfAborted()
    assertStable(dependencies.meter, session, prepared, options.stability)
    assertHardAnchorsStable(session, prepared)
    signal?.throwIfAborted()
    assertFinalContextFit(
      dependencies.meter,
      session,
      summarized,
      options.contextWindow,
      options.projectedExtraTokens ?? 0,
    )
    stage = 'commit'
    const pending = commitBody(session, startEvent, summarized)
    // These checks are synchronous and deterministic. They defend against a
    // malformed replacement implementation before the lock is closed.
    assertAnchorProjection(session, plan.anchor.headSeq, plan.anchor.headHash)
    assertLatestMessagePreserved(session, prepared.latestBeforeCommit)
    closing = true
    const endEvent = session.append('compaction/end', lifecycle)
    closed = true
    result = { ...pending, endSeq: endEvent.seq }
  } catch (error) {
    failure = { error, stage: closing ? 'commit' : stage }
    if (!closing) {
      closing = true
      try {
        session.append('compaction/end', { ...lifecycle, error: errorChain(error) })
        closed = true
      } catch (closeError) {
        failure = { error: closeError, stage: 'commit' }
      }
    }
  }

  if (closed && options.flush !== undefined) {
    try {
      await options.flush()
    } catch (error) {
      flushFailure = error
    }
  }

  if (options.owner === null) signal?.throwIfAborted()
  if (failure !== undefined) {
    if (options.owner === null) throwManualFailure(failure)
    throw failure.error
  }
  if (flushFailure !== undefined) {
    throw new ManualCompactionError(
      'persistence',
      'manual compaction durability checkpoint failed',
      { cause: flushFailure },
    )
  }
  if (result === undefined) throw new Error('anchored compaction committed without a result')
  return result
}

async function summarizeUntilValid(
  dependencies,
  session,
  prepared,
  agent,
  compactionId,
  sourceCommandId,
  contextWindow,
  projectedExtraTokens,
  signal,
) {
  const input = buildSummarizationInput(session, prepared.plan)
  let lastError
  for (let attempt = 0; attempt <= dependencies.config.compactionRetries; attempt += 1) {
    signal?.throwIfAborted()
    try {
      const summaryResult = await dependencies.summarize(input, agent, signal)
      signal?.throwIfAborted()
      const summary = validateSummary(summaryResult.summary)
      const summaryProbe = createUserMessage({
        content: summary,
        source: { kind: 'plugin', plugin: 'dsh-compaction-anchored-summary-budget' },
      })
      const summaryTokens = dependencies.meter.estimateMessage(summaryProbe)
      if (summaryTokens > dependencies.config.maxTokens) {
        throw new Error(
          `anchored compaction: summary exceeds output budget (${summaryTokens} estimated tokens > ${dependencies.config.maxTokens})`,
        )
      }
      const checkpointContent = prepared.plan.anchor.kind === 'native'
        ? frameNativeSummary(summary)
        : frameAnchorEnvelope(prepared.plan.anchor.headEvent, summary)
      const checkpointMessage = createUserMessage({
        content: checkpointContent,
        source: compactCheckpointSource(compactionId, sourceCommandId),
      })
      // Validate deterministic embedded anchors before any summary/replacement
      // body event is appended.
      if (prepared.plan.anchor.kind !== 'native') {
        const probe = {
          events: session.events.concat({
            type: 'user/message',
            seq: session.events.length,
            time: Date.now(),
            data: checkpointMessage,
            surfaceOp: { op: 'replace', start: prepared.start, end: prepared.end },
            sourceEventSeqs: prepared.shadowedSeqs,
          }),
          surface: { nodes: replaceNodes(session.surface.nodes, prepared, session.events.length) },
        }
        const state = classifyAnchor(probe)
        if (state.kind !== 'embedded'
          || state.headSeq !== prepared.plan.anchor.headSeq
          || state.headHash !== prepared.plan.anchor.headHash) {
          throw new Error('anchored compaction: assembled checkpoint does not preserve HEAD')
        }
      }
      const checkpointTokens = dependencies.meter.estimateMessage(checkpointMessage)
      if (checkpointTokens >= prepared.shadowedTokenCount) {
        throw new Error(
          `anchored compaction: checkpoint is not smaller than shadowed content (${checkpointTokens} estimated tokens >= ${prepared.shadowedTokenCount})`,
        )
      }
      if (contextWindow !== undefined) {
        const projectedTotal = prepared.measurement.totalTokens
          - prepared.shadowedTokenCount
          + checkpointTokens
          + projectedExtraTokens
        if (projectedTotal > contextWindow) {
          throw new Error(
            `anchored compaction: protected context would exceed the model window (${projectedTotal} projected tokens > ${contextWindow})`,
          )
        }
      }
      return {
        ...prepared,
        ...summaryResult,
        summary,
        checkpointMessage,
      }
    } catch (error) {
      lastError = error
      if (signal?.aborted) signal.throwIfAborted()
    }
  }
  throw lastError ?? new Error('anchored compaction: summarization failed without an error')
}

function prepareCurrentSelection(meter, session, plan) {
  validatePlanSelection(session, plan)
  const measurement = meter.measure(session)
  const current = currentSelection(session, plan.start, plan.end)
  const selectedNodes = measurement.nodes.slice(current.startIdx, current.endIdx + 1)
  if (selectedNodes.length !== current.shadowedSeqs.length
    || selectedNodes.some((entry, index) => entry.seq !== current.shadowedSeqs[index])) {
    throw new SurfaceChangedError('anchored compaction: selected surface changed before summarization')
  }
  const currentAnchor = classifyAnchor(session)
  if (currentAnchor.kind !== plan.anchor.kind
    || currentAnchor.headSeq !== plan.anchor.headSeq
    || currentAnchor.headHash !== plan.anchor.headHash
    || currentAnchor.anchorNodeSeq !== plan.anchor.anchorNodeSeq) {
    throw new SurfaceChangedError('anchored compaction: HEAD projection changed before summarization')
  }
  return {
    plan,
    measurement,
    ...current,
    selectedNodes,
    shadowedTokenCount: selectedNodes.reduce((sum, entry) => sum + entry.tokens, 0),
    latestBeforeCommit: lastVisibleSurfaceMessage(session),
  }
}

function assertStable(meter, session, prepared, stability) {
  if (stability === 'whole-surface') {
    const current = meter.measure(session)
    if (!isDeepStrictEqual(current.nodes, prepared.measurement.nodes)) {
      throw new SurfaceChangedError('anchored compaction: session surface changed during summarization')
    }
    return
  }
  const current = currentSelection(session, prepared.start, prepared.end)
  if (!isDeepStrictEqual(current.shadowedSeqs, prepared.shadowedSeqs)) {
    throw new SurfaceChangedError('anchored compaction: selected span changed during summarization')
  }
  const selected = meter.measure(session).nodes.slice(current.startIdx, current.endIdx + 1)
  if (!isDeepStrictEqual(selected, prepared.selectedNodes)) {
    throw new SurfaceChangedError('anchored compaction: selected span was rewritten during summarization')
  }
}

function assertHardAnchorsStable(session, prepared) {
  assertProtectedSurfaceNodes(session, prepared.plan.protectedPrefixSeqs, 'PREFIX/HEAD', 0)
  assertProtectedSurfaceNodes(session, prepared.plan.protectedTailSeqs, 'TAIL')
  const state = classifyAnchor(session)
  if (state.kind !== prepared.plan.anchor.kind
    || state.headSeq !== prepared.plan.anchor.headSeq
    || state.headHash !== prepared.plan.anchor.headHash
    || state.anchorNodeSeq !== prepared.plan.anchor.anchorNodeSeq) {
    throw new SurfaceChangedError('anchored compaction: HEAD changed during summarization')
  }
  const originalLastIndex = session.surface.nodes.indexOf(prepared.plan.hardLastSeq)
  const originalLast = originalLastIndex === -1
    ? undefined
    : lastMessageAt(session, prepared.plan.hardLastSeq)
  if (originalLast === undefined || fullMessageSha256(originalLast) !== prepared.plan.hardLastHash) {
    throw new SurfaceChangedError('anchored compaction: protected final message changed during summarization')
  }
  const latest = lastVisibleSurfaceMessage(session)
  if (latest === undefined || latest.index <= session.surface.nodes.indexOf(prepared.end)) {
    throw new SurfaceChangedError('anchored compaction: latest message would fall inside the replacement range')
  }
  prepared.latestBeforeCommit = latest
}

function assertProtectedSurfaceNodes(session, expected, label, requiredStart) {
  if (!Array.isArray(expected) || expected.length === 0) return
  const start = requiredStart ?? session.surface.nodes.indexOf(expected[0])
  if (start < 0
    || expected.some((seq, index) => session.surface.nodes[start + index] !== seq)) {
    throw new SurfaceChangedError(`anchored compaction: protected ${label} changed during summarization`)
  }
}

function assertFinalContextFit(meter, session, summarized, contextWindow, projectedExtraTokens) {
  if (contextWindow === undefined) return
  const measurement = meter.measure(session)
  const checkpointTokens = meter.estimateMessage(summarized.checkpointMessage)
  const projectedTotal = measurement.totalTokens
    - summarized.shadowedTokenCount
    + checkpointTokens
    + projectedExtraTokens
  if (projectedTotal > contextWindow) {
    throw new SurfaceChangedError(
      `anchored compaction: protected context would exceed the model window at commit (${projectedTotal} projected tokens > ${contextWindow})`,
    )
  }
}

function assertLatestMessagePreserved(session, before) {
  if (before === undefined) return
  const after = lastVisibleSurfaceMessage(session)
  if (after === undefined || after.seq !== before.seq || after.hash !== before.hash) {
    throw new Error('anchored compaction: final message changed while committing replacement')
  }
}

function commitBody(session, startEvent, summarized) {
  const callProvenance = summarized.llmStreamCall === true
    ? { rawOutput: summarized.rawOutput, llmStreamCall: true }
    : summarized.rawOutput === undefined ? {} : { rawOutput: summarized.rawOutput }
  const summaryEvent = session.append('compaction/summary', {
    compactionId: startEvent.data.compactionId,
    ...startEvent.data.sourceCommandId === undefined
      ? {}
      : { sourceCommandId: startEvent.data.sourceCommandId },
    summary: summarized.summary,
    ...callProvenance,
    shadowedRange: { start: summarized.start, end: summarized.end },
    shadowedSeqs: [...summarized.shadowedSeqs],
    shadowedTokenCount: summarized.shadowedTokenCount,
    provider: summarized.provider,
    model: summarized.model,
    ...summarized.maxTokens === undefined ? {} : { maxTokens: summarized.maxTokens },
    ...summarized.usage === undefined ? {} : { usage: summarized.usage },
  })
  const sources = uniqueSeqs([
    startEvent.seq,
    summaryEvent.seq,
    ...summarized.shadowedSeqs,
    ...summarized.plan.anchor.kind === 'native' ? [] : [summarized.plan.anchor.headSeq],
  ])
  session.append('user/message', summarized.checkpointMessage, {
    surfaceOp: { op: 'replace', start: summarized.start, end: summarized.end },
    sourceEventSeqs: sources,
  })
  return {
    compactionId: startEvent.data.compactionId,
    ...startEvent.data.sourceCommandId === undefined
      ? {}
      : { sourceCommandId: startEvent.data.sourceCommandId },
    startSeq: startEvent.seq,
    summarySeq: summaryEvent.seq,
    summary: summarized.summary,
    shadowedRange: { start: summarized.start, end: summarized.end },
    shadowedSeqs: [...summarized.shadowedSeqs],
    shadowedTokenCount: summarized.shadowedTokenCount,
  }
}

function validatePlanSelection(session, plan) {
  const current = currentSelection(session, plan.start, plan.end)
  if (!isDeepStrictEqual(current.shadowedSeqs, plan.shadowedSeqs)) {
    throw new SurfaceChangedError('anchored compaction: final plan is stale')
  }
  if (plan.shadowedSeqs.includes(plan.anchor.headSeq)
    || plan.shadowedSeqs.includes(plan.hardLastSeq)) {
    throw new Error('anchored compaction: final plan overlaps a hard anchor')
  }
}

function currentSelection(session, start, end) {
  const nodes = session.surface.nodes
  const startIdx = nodes.indexOf(start)
  const endIdx = nodes.indexOf(end)
  if (startIdx === -1 || endIdx === -1 || startIdx > endIdx) {
    throw new SurfaceChangedError('anchored compaction: selected range no longer exists')
  }
  if (!toolPairingBalancedBefore(session, start)
    || !toolPairingBalancedAfter(session, end)) {
    throw new SurfaceChangedError('anchored compaction: selected range is no longer tool-balanced')
  }
  return { start, end, startIdx, endIdx, shadowedSeqs: [...nodes.slice(startIdx, endIdx + 1)] }
}

function inspectCompactionEntryState(events) {
  let openTurn = null
  let openTurnKnown = false
  let unmatchedCompactionStart
  let compactionKnown = false
  let latestEndSeedSeq
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index]
    if (latestEndSeedSeq === undefined && event.type === 'session/end-seed') latestEndSeedSeq = event.seq
    if (!compactionKnown) {
      if (event.type === 'compaction/start') {
        unmatchedCompactionStart = event
        compactionKnown = true
      } else if (event.type === 'compaction/end') compactionKnown = true
    }
    if (!openTurnKnown) {
      if (event.type === 'turn/start') {
        openTurn = event.data.turn
        openTurnKnown = true
      } else if (event.type === 'turn/end') openTurnKnown = true
    }
    if (openTurnKnown && compactionKnown && latestEndSeedSeq !== undefined) break
  }
  return { openTurn, unmatchedCompactionStart, latestEndSeedSeq }
}

function throwManualFailure(failure) {
  if (failure.stage === 'commit') {
    throw new ManualCompactionError('commit', 'manual compaction did not commit cleanly', { cause: failure.error })
  }
  if (failure.error instanceof SurfaceChangedError) {
    throw new ManualCompactionError('changed', 'the compacted history changed during manual compaction', { cause: failure.error })
  }
  throw new ManualCompactionError('summary', 'manual compaction could not produce a smaller summary', { cause: failure.error })
}

function lastMessageAt(session, seq) {
  const event = session.events[seq]
  switch (event?.type) {
    case 'user/message': return event.data
    case 'assistant/message': return event.data.message.content.length === 0 ? undefined : event.data.message
    case 'tool/result': return event.data.message
    default: return undefined
  }
}

function uniqueSeqs(values) {
  return [...new Set(values)]
}

function replaceNodes(nodes, prepared, replacementSeq) {
  const output = [...nodes]
  output.splice(prepared.startIdx, prepared.endIdx - prepared.startIdx + 1, replacementSeq)
  return output
}
