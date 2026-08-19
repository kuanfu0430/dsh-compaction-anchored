import { deepFreeze } from '@deepseek-ai/dsh-llm'

export const DEFAULT_THRESHOLD_RATIO = 0.8
export const DEFAULT_RETAIN_RATIO = 0.16
export const SAFE_THRESHOLD_RATIO = 0.8

const POLICY_KEYS = [
  'thresholdRatio',
  'retainRatio',
  'retainTokens',
  'summarizationProvider',
  'summarizationModel',
  'maxTokens',
  'compactionRetries',
  'maxOverflowRetries',
]
const CONFIG_KEYS = new Set([...POLICY_KEYS, 'modelPolicies', 'auto'])
const MODEL_POLICY_KEYS = new Set(['provider', 'model', ...POLICY_KEYS])

export class TargetPressureConfigError extends Error {
  constructor(targetKey, message) {
    super(message)
    this.name = 'TargetPressureConfigError'
    this.targetKey = targetKey
  }
}

export function resolveConfig(config = {}) {
  assertRecord(config, 'AnchoredCompactionConfig')
  validateKeys(config, CONFIG_KEYS, 'AnchoredCompactionConfig')
  validatePolicy(config, 'AnchoredCompactionConfig')
  if (config.auto !== undefined && typeof config.auto !== 'boolean') {
    throw new Error('AnchoredCompactionConfig.auto must be a boolean')
  }

  const thresholdRatio = config.thresholdRatio ?? DEFAULT_THRESHOLD_RATIO
  const retention = resolveRetention(config, { retainRatio: DEFAULT_RETAIN_RATIO })
  validateRetentionRatio(thresholdRatio, retention, 'AnchoredCompactionConfig')
  const modelPolicies = resolveModelPolicies(config.modelPolicies)
  for (const [index, policy] of modelPolicies.entries()) {
    validateRetentionRatio(
      policy.thresholdRatio ?? thresholdRatio,
      resolveRetention(policy, retention),
      `AnchoredCompactionConfig.modelPolicies[${index}]`,
    )
  }

  return deepFreeze({
    thresholdRatio,
    ...retention,
    summarizationProvider: config.summarizationProvider ?? '',
    summarizationModel: config.summarizationModel ?? '',
    maxTokens: config.maxTokens ?? 8192,
    compactionRetries: config.compactionRetries ?? 1,
    maxOverflowRetries: config.maxOverflowRetries ?? 1,
    modelPolicies,
    auto: config.auto ?? true,
  })
}

export function resolveTargetPolicy(config, target) {
  const override = config.modelPolicies.find((entry) => (
    entry.provider === target.provider && entry.model === target.model
  ))
  const inheritedRetention = config.retainTokens === undefined
    ? { retainRatio: config.retainRatio }
    : { retainTokens: config.retainTokens }
  return deepFreeze({
    target: { provider: target.provider, model: target.model },
    thresholdRatio: override?.thresholdRatio ?? config.thresholdRatio,
    ...resolveRetention(override ?? {}, inheritedRetention),
    summarizationProvider: override?.summarizationProvider ?? config.summarizationProvider,
    summarizationModel: override?.summarizationModel ?? config.summarizationModel,
    maxTokens: override?.maxTokens ?? config.maxTokens,
    compactionRetries: override?.compactionRetries ?? config.compactionRetries,
    maxOverflowRetries: override?.maxOverflowRetries ?? config.maxOverflowRetries,
  })
}

export function normalizeControlPolicy(value) {
  if (value === null) return null
  if (value === undefined || typeof value !== 'object' || Array.isArray(value)) return undefined
  if (value.mode !== 'percent' && value.mode !== 'tokens') return undefined
  if (!Number.isInteger(value.percent) || value.percent < 30 || value.percent > 80) return undefined
  if (!Number.isSafeInteger(value.tokens) || value.tokens <= 0) return undefined
  return deepFreeze({ mode: value.mode, percent: value.percent, tokens: value.tokens })
}

export function resolveCompactSpec(policy, contextWindow, controlPolicy) {
  const targetKey = `${policy.target.provider}/${policy.target.model}`
  if (!Number.isSafeInteger(contextWindow) || contextWindow <= 0) {
    throw new TargetPressureConfigError(
      targetKey,
      `anchored compaction: contextWindow (${String(contextWindow)}) must be a positive safe integer`,
    )
  }

  const safeThresholdTokens = Math.floor(contextWindow * SAFE_THRESHOLD_RATIO)
  const normalizedControl = normalizeControlPolicy(controlPolicy)
  let configuredThresholdTokens
  let policySource
  if (normalizedControl === null || normalizedControl === undefined) {
    configuredThresholdTokens = Math.floor(contextWindow * policy.thresholdRatio)
    policySource = normalizedControl === null ? 'backend-default' : 'invalid-policy-default'
  } else if (normalizedControl.mode === 'tokens') {
    configuredThresholdTokens = normalizedControl.tokens
    policySource = 'control-tokens'
  } else {
    configuredThresholdTokens = Math.floor(contextWindow * (normalizedControl.percent / 100))
    policySource = 'control-percent'
  }
  const thresholdTokens = Math.min(configuredThresholdTokens, safeThresholdTokens)
  const retainTokens = policy.retainTokens === undefined
    ? Math.floor(contextWindow * policy.retainRatio)
    : policy.retainTokens
  if (retainTokens >= thresholdTokens) {
    throw new TargetPressureConfigError(
      targetKey,
      `anchored compaction: retainTokens (${retainTokens}) must be less than effective threshold ${thresholdTokens}`,
    )
  }

  return deepFreeze({
    ...policy,
    contextWindow,
    thresholdTokens,
    configuredThresholdTokens,
    safeThresholdTokens,
    thresholdCapped: configuredThresholdTokens > safeThresholdTokens,
    retainTokens,
    policySource,
  })
}

function resolveRetention(config, fallback) {
  if (config.retainTokens !== undefined) return { retainTokens: config.retainTokens }
  if (config.retainRatio !== undefined) return { retainRatio: config.retainRatio }
  return fallback
}

function resolveModelPolicies(value) {
  if (value === undefined) return []
  if (!Array.isArray(value)) throw new Error('AnchoredCompactionConfig.modelPolicies must be an array')
  const seen = new Set()
  return value.map((entry, index) => {
    const label = `AnchoredCompactionConfig.modelPolicies[${index}]`
    assertRecord(entry, label)
    validateKeys(entry, MODEL_POLICY_KEYS, label)
    assertNonEmptyString(entry.provider, `${label}.provider`)
    assertNonEmptyString(entry.model, `${label}.model`)
    validatePolicy(entry, label)
    const key = `${entry.provider}\u0000${entry.model}`
    if (seen.has(key)) throw new Error(`AnchoredCompactionConfig has duplicate model policy for ${entry.provider}/${entry.model}`)
    seen.add(key)
    return { ...entry }
  })
}

function validatePolicy(config, label) {
  if (config.thresholdRatio !== undefined) assertRatio(config.thresholdRatio, `${label}.thresholdRatio`)
  if (config.retainRatio !== undefined) assertRatio(config.retainRatio, `${label}.retainRatio`)
  if (config.retainTokens !== undefined) assertNonNegativeInteger(config.retainTokens, `${label}.retainTokens`)
  if (config.retainRatio !== undefined && config.retainTokens !== undefined) {
    throw new Error(`${label}: retainRatio and retainTokens are mutually exclusive`)
  }
  if (config.maxTokens !== undefined) assertPositiveInteger(config.maxTokens, `${label}.maxTokens`)
  if (config.compactionRetries !== undefined) assertNonNegativeInteger(config.compactionRetries, `${label}.compactionRetries`)
  if (config.maxOverflowRetries !== undefined) assertNonNegativeInteger(config.maxOverflowRetries, `${label}.maxOverflowRetries`)
  const provider = config.summarizationProvider
  const model = config.summarizationModel
  if (provider !== undefined && typeof provider !== 'string') throw new Error(`${label}.summarizationProvider must be a string`)
  if (model !== undefined && typeof model !== 'string') throw new Error(`${label}.summarizationModel must be a string`)
  if ((provider === undefined) !== (model === undefined)
    || (typeof provider === 'string' && typeof model === 'string' && (provider.length === 0) !== (model.length === 0))) {
    throw new Error(`${label}: summarizationProvider and summarizationModel must be set together as an empty or non-empty pair`)
  }
}

function validateRetentionRatio(thresholdRatio, retention, label) {
  const effectiveThresholdRatio = Math.min(thresholdRatio, SAFE_THRESHOLD_RATIO)
  if (retention.retainRatio !== undefined && retention.retainRatio >= effectiveThresholdRatio) {
    throw new Error(
      `${label}: retainRatio (${retention.retainRatio}) must be less than the safety-capped thresholdRatio (${effectiveThresholdRatio})`,
    )
  }
}

function validateKeys(value, keys, label) {
  for (const key of Object.keys(value)) {
    if (!keys.has(key)) throw new Error(`${label}: unknown key "${key}"`)
  }
}

function assertRecord(value, label) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${label} must be an object`)
}

function assertNonEmptyString(value, label) {
  if (typeof value !== 'string' || value.length === 0) throw new Error(`${label} must be a non-empty string`)
}

function assertRatio(value, label) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0 || value > 1) {
    throw new Error(`${label} (${String(value)}) must be in (0, 1]`)
  }
}

function assertPositiveInteger(value, label) {
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`${label} (${String(value)}) must be a positive safe integer`)
}

function assertNonNegativeInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error(`${label} (${String(value)}) must be a non-negative safe integer`)
}
