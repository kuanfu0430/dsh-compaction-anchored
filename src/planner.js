import {
  isCompactCheckpointSource,
  toolPairingBalancedAfter,
  toolPairingBalancedBefore,
} from '@deepseek-ai/dsh-compaction'
import { classifyAnchor, eventMessage, lastVisibleSurfaceMessage } from './anchor.js'

export class PlanningError extends Error {
  constructor(code, message, options) {
    super(message, options)
    this.name = 'PlanningError'
    this.code = code
  }
}

/**
 * Build a fresh authoritative plan from the current surface. Call this once
 * before selective pruning for protection and again afterwards for commit.
 */
export function planCompaction(session, measurement, options = {}) {
  assertMeasurementMatchesSurface(session, measurement)
  const anchor = classifyAnchor(session)
  if (anchor.kind === 'missing') return null
  const last = lastVisibleSurfaceMessage(session)
  if (last === undefined) return null

  const baseTailStart = requiredTailStart(session, last.index)
  if (baseTailStart <= anchor.anchorIndex) return null
  const tailStartIdx = extendTail(
    session,
    measurement,
    baseTailStart,
    anchor.anchorIndex + 1,
    options.retainTokens ?? 0,
  )
  if (tailStartIdx <= anchor.anchorIndex) return null

  const startIdx = anchor.kind === 'native' ? anchor.anchorIndex + 1 : anchor.anchorIndex
  const endIdx = tailStartIdx - 1
  if (endIdx < startIdx) return null
  // Once an anchored envelope exists, replacing it without any newly accrued
  // middle node would only re-summarize the same checkpoint.
  if (anchor.kind === 'embedded' && endIdx === startIdx) return null

  const nodes = session.surface.nodes
  const start = nodes[startIdx]
  const end = nodes[endIdx]
  if (!toolPairingBalancedBefore(session, start)) {
    throw new PlanningError('unbalanced', `anchored compaction: start seq ${start} is not tool-balanced`)
  }
  if (!toolPairingBalancedAfter(session, end)) {
    throw new PlanningError('unbalanced', `anchored compaction: end seq ${end} is not tool-balanced`)
  }
  const shadowedSeqs = [...nodes.slice(startIdx, endIdx + 1)]
  if (shadowedSeqs.length === 1) {
    const only = session.events[shadowedSeqs[0]]
    if (only?.type === 'user/message' && isCompactCheckpointSource(only.data.source)) return null
  }
  const selectedNodes = measurement.nodes.slice(startIdx, endIdx + 1)
  if (selectedNodes.length !== shadowedSeqs.length
    || selectedNodes.some((entry, index) => entry.seq !== shadowedSeqs[index])) {
    throw new PlanningError('changed', 'anchored compaction: token-meter selection is stale')
  }
  const shadowedTokenCount = selectedNodes.reduce((sum, entry) => sum + entry.tokens, 0)
  const requested = options.requestedRange
  if (requested !== undefined && (requested.start !== start || requested.end !== end)) {
    throw new PlanningError(
      'range-policy',
      `compactRegion: requested range ${requested.start}-${requested.end} is not the complete anchored middle ${start}-${end}`,
    )
  }
  if (shadowedSeqs.includes(anchor.headSeq) || shadowedSeqs.includes(last.seq)) {
    throw new PlanningError('anchor-overlap', 'anchored compaction: planned range overlaps a hard anchor')
  }

  return Object.freeze({
    anchor,
    measurement,
    start,
    end,
    startIdx,
    endIdx,
    tailStartIdx,
    hardLastSeq: last.seq,
    hardLastHash: last.hash,
    protectedPrefixSeqs: [...nodes.slice(0, startIdx)],
    protectedTailSeqs: [...nodes.slice(tailStartIdx)],
    shadowedSeqs,
    selectedNodes,
    shadowedTokenCount,
    migration: anchor.kind === 'legacy',
    requestedRange: requested,
  })
}

export function assertRequestedRangeExists(session, start, end) {
  const nodes = session.surface.nodes
  const startIdx = nodes.indexOf(start)
  const endIdx = nodes.indexOf(end)
  if (startIdx === -1) throw new PlanningError('range-missing', `compactRegion: start seq ${start} not found in surface`)
  if (endIdx === -1) throw new PlanningError('range-missing', `compactRegion: end seq ${end} not found in surface`)
  if (startIdx > endIdx) {
    throw new PlanningError(
      'range-reversed',
      `compactRegion: start seq ${start} (position ${startIdx}) is after end seq ${end} (position ${endIdx})`,
    )
  }
  if (!toolPairingBalancedBefore(session, start)) {
    throw new PlanningError('unbalanced', `compactRegion: start seq ${start} would split a tool pair`)
  }
  if (!toolPairingBalancedAfter(session, end)) {
    throw new PlanningError('unbalanced', `compactRegion: end seq ${end} would split a tool pair or open step`)
  }
  return { startIdx, endIdx }
}

export function inspectTurnRanges(events) {
  let open
  let latestCompleted
  for (const event of events) {
    if (event.type === 'turn/start') {
      open = { turn: event.data.turn, startSeq: event.seq, endSeq: undefined }
    } else if (event.type === 'turn/end' && open?.turn === event.data.turn) {
      latestCompleted = { ...open, endSeq: event.seq }
      open = undefined
    }
  }
  return { open, latestCompleted }
}

export function requiredTailStart(session, hardLastIndex) {
  const { open, latestCompleted } = inspectTurnRanges(session.events)
  const target = open ?? latestCompleted
  if (target !== undefined) {
    const index = firstTurnSurfaceIndex(session, target)
    if (index !== undefined) return index
  }

  for (let index = hardLastIndex; index >= 0; index -= 1) {
    const event = session.events[session.surface.nodes[index]]
    if (event?.type !== 'user/message' || event.surfaceOp !== 'append') continue
    if (isCompactCheckpointSource(event.data.source)) continue
    if (eventMessage(event) !== null) return index
  }
  return hardLastIndex
}

function firstTurnSurfaceIndex(session, turnRange) {
  const endSeq = turnRange.endSeq ?? Number.POSITIVE_INFINITY
  let firstVisible
  for (let index = 0; index < session.surface.nodes.length; index += 1) {
    const seq = session.surface.nodes[index]
    if (seq <= turnRange.startSeq || seq >= endSeq) continue
    const event = session.events[seq]
    if (event?.surfaceOp !== 'append' || eventMessage(event) === null) continue
    if (firstVisible === undefined) firstVisible = index
    if (event.type === 'user/message' && !isCompactCheckpointSource(event.data.source)) return index
  }
  return firstVisible
}

function extendTail(session, measurement, initialStart, floorIndex, retainTokens) {
  let start = initialStart
  if (tokensFrom(measurement, start) >= retainTokens) return balanceTailStart(session, start, floorIndex)

  const starts = currentCompletedTurnStarts(session)
    .filter((index) => index >= floorIndex && index < start)
    .sort((a, b) => b - a)
  for (const index of starts) {
    start = index
    if (tokensFrom(measurement, start) >= retainTokens) break
  }
  // If complete visible turns cannot satisfy the budget, retaining everything
  // back to the hard-anchor boundary is safer than slicing a turn in half.
  if (tokensFrom(measurement, start) < retainTokens) start = floorIndex
  return balanceTailStart(session, start, floorIndex)
}

function currentCompletedTurnStarts(session) {
  const ranges = []
  let open
  for (const event of session.events) {
    if (event.type === 'turn/start') open = { turn: event.data.turn, startSeq: event.seq }
    if (event.type === 'turn/end' && open?.turn === event.data.turn) {
      const index = firstTurnSurfaceIndex(session, { ...open, endSeq: event.seq })
      if (index !== undefined) ranges.push(index)
      open = undefined
    }
  }
  return [...new Set(ranges)]
}

function balanceTailStart(session, initial, floorIndex) {
  let index = initial
  while (index > floorIndex && !toolPairingBalancedBefore(session, session.surface.nodes[index])) index -= 1
  if (!toolPairingBalancedBefore(session, session.surface.nodes[index])) {
    throw new PlanningError('unbalanced', 'anchored compaction: no balanced protected-tail boundary exists')
  }
  return index
}

function tokensFrom(measurement, index) {
  let total = 0
  for (let cursor = index; cursor < measurement.nodes.length; cursor += 1) {
    total += measurement.nodes[cursor].tokens
  }
  return total
}

function assertMeasurementMatchesSurface(session, measurement) {
  const nodes = session.surface.nodes
  if (!measurement || !Array.isArray(measurement.nodes)
    || nodes.length !== measurement.nodes.length
    || nodes.some((seq, index) => seq !== measurement.nodes[index]?.seq)) {
    throw new PlanningError('changed', 'anchored compaction: token-meter surface does not match current surface')
  }
}
