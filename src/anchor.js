import { createHash } from 'node:crypto'
import { isDeepStrictEqual } from 'node:util'
import { isCompactCheckpointSource } from '@deepseek-ai/dsh-compaction'

export const CHECKPOINT_PREAMBLE = 'This is an automatically generated checkpoint condensing an earlier span of the conversation to free up context. Treat the captured context as established background and build on it without restating it. Continue the task directly from the messages that follow, without acknowledging this checkpoint.'
export const SUMMARY_OPEN_TAG = '<compacted-summary>'
export const SUMMARY_CLOSE_TAG = '</compacted-summary>'
export const ENVELOPE_VERSION = 1
export const ENVELOPE_SEPARATOR = '</anchored-head>\n<compacted-summary>'
export const ENVELOPE_CLOSE = '</compacted-summary>\n</anchored-compaction>'

const ENVELOPE_MARKER = '<anchored-compaction'
const META_RE = /^<anchored-compaction version=(\d+) anchorSeq=(\d+) anchorBlocks=(\d+) anchorSha256=([0-9a-f]{64})>\n<anchored-head>$/u

export class AnchorInvariantError extends Error {
  constructor(code, message, options) {
    super(message, options)
    this.name = 'AnchorInvariantError'
    this.code = code
  }
}

/** RFC 8785-compatible canonical JSON for the lossless-JSON message domain. */
export function canonicalizeJson(value) {
  if (value === null) return 'null'
  switch (typeof value) {
    case 'boolean': return value ? 'true' : 'false'
    case 'string': return JSON.stringify(value)
    case 'number': {
      if (!Number.isFinite(value)) throw new TypeError('canonical JSON cannot encode a non-finite number')
      return JSON.stringify(value)
    }
    case 'object': {
      if (Array.isArray(value)) return `[${value.map(canonicalizeJson).join(',')}]`
      const prototype = Object.getPrototypeOf(value)
      if (prototype !== Object.prototype && prototype !== null) {
        throw new TypeError('canonical JSON requires plain objects')
      }
      const members = Object.keys(value).sort().map((key) => (
        `${JSON.stringify(key)}:${canonicalizeJson(value[key])}`
      ))
      return `{${members.join(',')}}`
    }
    default:
      throw new TypeError(`canonical JSON cannot encode ${typeof value}`)
  }
}

export function canonicalSha256(value) {
  return createHash('sha256').update(canonicalizeJson(value), 'utf8').digest('hex')
}

export function anchorSha256(message) {
  return canonicalSha256({ role: message.role, content: message.content })
}

export function fullMessageSha256(message) {
  return canonicalSha256(message)
}

export function findRawHead(session) {
  for (const event of session.events) {
    if (event.type !== 'user/message' || event.surfaceOp !== 'append') continue
    if (isCompactCheckpointSource(event.data.source)) continue
    return event
  }
  return undefined
}

export function classifyAnchor(session) {
  const headEvent = findRawHead(session)
  if (headEvent === undefined) {
    const checkpoint = [...session.surface.nodes]
      .map((seq) => session.events[seq])
      .find((event) => event?.type === 'user/message' && isCompactCheckpointSource(event.data.source))
    if (checkpoint !== undefined) {
      throw new AnchorInvariantError(
        'head-raw-missing',
        `anchored compaction: checkpoint seq ${checkpoint.seq} exists but no append-origin raw HEAD can be recovered`,
      )
    }
    return { kind: 'missing' }
  }
  const hash = anchorSha256(headEvent.data)
  const nodes = [...session.surface.nodes]
  const nativeIndex = nodes.indexOf(headEvent.seq)
  if (nativeIndex !== -1) {
    return Object.freeze({
      kind: 'native',
      headEvent,
      headSeq: headEvent.seq,
      headHash: hash,
      anchorIndex: nativeIndex,
      anchorNodeSeq: headEvent.seq,
    })
  }

  const ancestry = ancestryPredicate(session.events, headEvent.seq)
  const descendants = nodes
    .map((seq, index) => ({ seq, index }))
    .filter(({ seq }) => ancestry(seq))
  if (descendants.length !== 1) {
    throw new AnchorInvariantError(
      'head-lineage',
      `anchored compaction: raw HEAD seq ${headEvent.seq} has ${descendants.length} current surface descendants; refusing an ambiguous migration`,
    )
  }
  const [{ seq: anchorNodeSeq, index: anchorIndex }] = descendants
  const event = session.events[anchorNodeSeq]
  if (event?.type !== 'user/message') {
    throw new AnchorInvariantError(
      'head-lineage',
      `anchored compaction: HEAD descendant seq ${anchorNodeSeq} is not a user/message checkpoint`,
    )
  }
  const envelope = parseAnchorEnvelope(event.data)
  if (envelope === null) {
    return Object.freeze({
      kind: 'legacy',
      headEvent,
      headSeq: headEvent.seq,
      headHash: hash,
      anchorIndex,
      anchorNodeSeq,
    })
  }
  validateEnvelopeAgainstHead(envelope, headEvent)
  return Object.freeze({
    kind: 'embedded',
    headEvent,
    headSeq: headEvent.seq,
    headHash: hash,
    anchorIndex,
    anchorNodeSeq,
    envelope,
  })
}

export function parseAnchorEnvelope(message) {
  const blocks = message?.content
  if (!Array.isArray(blocks) || blocks.length === 0) return null
  const hasEnvelopeFragment = blocks.some((block) => block?.type === 'text'
    && typeof block.text === 'string'
    && (block.text.includes(ENVELOPE_MARKER)
      || block.text.includes('<anchored-head>')
      || block.text.includes('</anchored-head>')
      || block.text.includes('</anchored-compaction>')))
  const first = blocks[0]
  if (first?.type !== 'text' || typeof first.text !== 'string') {
    if (hasEnvelopeFragment) throw new AnchorInvariantError('envelope-marker', 'anchored compaction: misplaced anchor envelope marker')
    return null
  }
  if (!first.text.includes(ENVELOPE_MARKER)) {
    if (hasEnvelopeFragment) throw new AnchorInvariantError('envelope-marker', 'anchored compaction: misplaced anchor envelope marker')
    return null
  }
  const prefix = `${CHECKPOINT_PREAMBLE}\n\n`
  if (!first.text.startsWith(prefix)) {
    throw new AnchorInvariantError('envelope-marker', 'anchored compaction: malformed envelope preamble')
  }
  const match = META_RE.exec(first.text.slice(prefix.length))
  if (match === null) throw new AnchorInvariantError('envelope-marker', 'anchored compaction: malformed anchor envelope marker')
  const version = Number(match[1])
  const anchorSeq = Number(match[2])
  const anchorBlocks = Number(match[3])
  const anchorHash = match[4]
  if (version !== ENVELOPE_VERSION) {
    throw new AnchorInvariantError('envelope-version', `anchored compaction: unsupported anchor envelope version ${version}`)
  }
  if (!Number.isSafeInteger(anchorSeq) || anchorSeq < 0
    || !Number.isSafeInteger(anchorBlocks) || anchorBlocks < 0) {
    throw new AnchorInvariantError('envelope-marker', 'anchored compaction: invalid anchor envelope numeric metadata')
  }
  const separatorIndex = 1 + anchorBlocks
  const closeIndex = blocks.length - 1
  if (separatorIndex >= closeIndex
    || blocks[separatorIndex]?.type !== 'text'
    || blocks[separatorIndex].text !== ENVELOPE_SEPARATOR
    || blocks[closeIndex]?.type !== 'text'
    || blocks[closeIndex].text !== ENVELOPE_CLOSE) {
    throw new AnchorInvariantError('envelope-marker', 'anchored compaction: envelope markers or block count do not match')
  }
  const headBlocks = blocks.slice(1, separatorIndex)
  const summary = blocks.slice(separatorIndex + 1, closeIndex)
  if (summary.length === 0 || summary.some((block) => block.type !== 'text')) {
    throw new AnchorInvariantError('envelope-summary', 'anchored compaction: envelope summary must contain only non-empty text blocks')
  }
  if (!summary.some((block) => block.text.trim().length > 0)) {
    throw new AnchorInvariantError('envelope-summary', 'anchored compaction: envelope summary is blank')
  }
  return Object.freeze({ version, anchorSeq, anchorBlocks, anchorHash, headBlocks, summary })
}

export function validateEnvelopeAgainstHead(envelope, headEvent) {
  if (envelope.anchorSeq !== headEvent.seq) {
    throw new AnchorInvariantError(
      'envelope-anchor',
      `anchored compaction: envelope anchorSeq ${envelope.anchorSeq} does not match raw HEAD seq ${headEvent.seq}`,
    )
  }
  if (envelope.anchorBlocks !== headEvent.data.content.length
    || !isDeepStrictEqual(envelope.headBlocks, headEvent.data.content)) {
    throw new AnchorInvariantError('envelope-anchor', 'anchored compaction: envelope HEAD blocks differ from the raw HEAD')
  }
  const actual = anchorSha256(headEvent.data)
  if (envelope.anchorHash !== actual) {
    throw new AnchorInvariantError(
      'envelope-hash',
      `anchored compaction: envelope HEAD hash ${envelope.anchorHash} does not match raw HEAD hash ${actual}`,
    )
  }
}

export function frameNativeSummary(summary) {
  return [
    { type: 'text', text: `${CHECKPOINT_PREAMBLE}\n\n${SUMMARY_OPEN_TAG}` },
    ...summary.map((block) => structuredClone(block)),
    { type: 'text', text: SUMMARY_CLOSE_TAG },
  ]
}

export function frameAnchorEnvelope(headEvent, summary) {
  const hash = anchorSha256(headEvent.data)
  const opening = [
    CHECKPOINT_PREAMBLE,
    '',
    `<anchored-compaction version=${ENVELOPE_VERSION} anchorSeq=${headEvent.seq} anchorBlocks=${headEvent.data.content.length} anchorSha256=${hash}>`,
    '<anchored-head>',
  ].join('\n')
  return [
    { type: 'text', text: opening },
    ...headEvent.data.content.map((block) => structuredClone(block)),
    { type: 'text', text: ENVELOPE_SEPARATOR },
    ...summary.map((block) => structuredClone(block)),
    { type: 'text', text: ENVELOPE_CLOSE },
  ]
}

export function lastVisibleSurfaceMessage(session) {
  const nodes = session.surface.nodes
  for (let index = nodes.length - 1; index >= 0; index -= 1) {
    const seq = nodes[index]
    const event = session.events[seq]
    const message = eventMessage(event)
    if (message !== null) return { index, seq, event, message, hash: fullMessageSha256(message) }
  }
  return undefined
}

export function eventMessage(event) {
  switch (event?.type) {
    case 'user/message': return event.data
    case 'assistant/message': return event.data.message.content.length === 0 ? null : event.data.message
    case 'tool/result': return event.data.message
    default: return null
  }
}

export function assertAnchorProjection(session, expectedHeadSeq, expectedHeadHash) {
  const state = classifyAnchor(session)
  if (state.kind === 'missing'
    || state.headSeq !== expectedHeadSeq
    || state.headHash !== expectedHeadHash) {
    throw new AnchorInvariantError('anchor-postcondition', 'anchored compaction: projected HEAD changed')
  }
  if (state.kind === 'embedded') validateEnvelopeAgainstHead(state.envelope, state.headEvent)
  return state
}

function ancestryPredicate(events, targetSeq) {
  const memo = new Map()
  const visit = (seq) => {
    if (seq === targetSeq) return true
    if (memo.has(seq)) return memo.get(seq)
    const event = events[seq]
    if (event === undefined || event.seq !== seq || !Array.isArray(event.sourceEventSeqs)) {
      memo.set(seq, false)
      return false
    }
    // Source seqs are strictly earlier, so recursion cannot cycle in a valid log.
    const result = event.sourceEventSeqs.some(visit)
    memo.set(seq, result)
    return result
  }
  return visit
}
