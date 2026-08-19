import z from '@deepseek-ai/schemastery'
import {
  CompactionEngine,
  ManualCompactionError,
} from '@deepseek-ai/dsh-compaction'
import {
  CONTEXT_WINDOW_EXCEEDED_CODE,
} from '@deepseek-ai/dsh-llm'
import {
  normalizeControlPolicy,
  resolveCompactSpec,
  resolveConfig,
  resolveTargetPolicy,
  TargetPressureConfigError,
} from './config.js'
import { assertRequestedRangeExists, planCompaction, PlanningError } from './planner.js'
import { selectivePrune } from './pruner.js'
import { summarizeWithLlm } from './summarizer.js'
import { assertNoActiveCompaction, executeCompaction } from './transaction.js'

export const BACKEND_IDENTITY = Object.freeze({
  name: '@kuanfu0430/dsh-compaction-anchored',
  version: '0.1.0',
  harness: '0.1.0-rc.7',
  anchored: true,
})

const thresholdRatioSchema = z.number()
const retainRatioSchema = z.number()
const retainTokensSchema = z.number().step(1).min(0)
const summarizationProviderSchema = z.string()
const summarizationModelSchema = z.string()
const maxTokensSchema = z.number().step(1).min(1)
const compactionRetriesSchema = z.number().step(1).min(0)
const maxOverflowRetriesSchema = z.number().step(1).min(0)
const modelPolicySchema = z.object({
  provider: z.string().required(),
  model: z.string().required(),
  thresholdRatio: thresholdRatioSchema,
  retainRatio: retainRatioSchema,
  retainTokens: retainTokensSchema,
  summarizationProvider: summarizationProviderSchema,
  summarizationModel: summarizationModelSchema,
  maxTokens: maxTokensSchema,
  compactionRetries: compactionRetriesSchema,
  maxOverflowRetries: maxOverflowRetriesSchema,
})

export class AnchoredCompactionEngine extends CompactionEngine {
  static inject = ['llm', 'tokenMeter', 'sessions']
  static backendIdentity = BACKEND_IDENTITY
  static Config = z.object({
    thresholdRatio: thresholdRatioSchema,
    retainRatio: retainRatioSchema,
    retainTokens: retainTokensSchema,
    summarizationProvider: summarizationProviderSchema,
    summarizationModel: summarizationModelSchema,
    maxTokens: maxTokensSchema,
    compactionRetries: compactionRetriesSchema,
    maxOverflowRetries: maxOverflowRetriesSchema,
    modelPolicies: z.array(modelPolicySchema),
    auto: z.boolean(),
  })

  constructor(ctx, config = {}) {
    super(ctx)
    this.config = resolveConfig(config)
    this.backendIdentity = BACKEND_IDENTITY
    this.warnedPressureTargets = new Set()
    this.warnedPolicySessions = new WeakSet()
    this.overflowRetries = new WeakMap()
    this.overflowAgents = new WeakMap()
    if (this.config.auto) this.registerAutomaticCompaction()
  }

  async summarize(input, agent, signal, policy = this.config) {
    return summarizeWithLlm(this.ctx, policy, input, agent, signal)
  }

  async compactIfNeeded(agent, trigger, signal) {
    if (trigger === 'context-overflow') {
      throw new Error('anchored compaction: context-overflow is reserved for the canonical agent/request-error listener')
    }
    if (trigger !== 'pressure') throw new Error(`anchored compaction: unsupported trigger ${String(trigger)}`)
    return this.compactAutomatic(agent, 'pressure', signal, [])
  }

  async compactRegion(start, end, agent, signal) {
    signal?.throwIfAborted()
    assertRequestedRangeExists(agent.session, start, end)
    assertCurrentTurnOpen(agent.session)
    assertNoActiveCompaction(agent.session, 'compactRegion')
    const targetState = await this.resolveTargetState(agent, signal, 'direct')
    signal?.throwIfAborted()
    const retainTokens = targetState?.spec.retainTokens ?? 0
    const requestedRange = { start, end }
    const provisional = planCompaction(
      agent.session,
      this.ctx.tokenMeter.measure(agent.session),
      { retainTokens, requestedRange },
    )
    if (provisional === null) {
      throw new PlanningError('range-policy', 'compactRegion: no complete anchored middle is currently compactable')
    }
    selectivePrune(this.ctx, agent.session, provisional, 'pressure', signal)
    signal?.throwIfAborted()
    let finalPlan
    try {
      finalPlan = planCompaction(
        agent.session,
        this.ctx.tokenMeter.measure(agent.session),
        { retainTokens, requestedRange },
      )
    } catch (error) {
      if (error instanceof PlanningError) {
        throw new PlanningError(
          'changed',
          'compactRegion: the authorized anchored middle changed during maintenance',
          { cause: error },
        )
      }
      throw error
    }
    if (finalPlan === null) {
      throw new PlanningError('changed', 'compactRegion: the authorized anchored middle disappeared after maintenance')
    }
    return this.execute(finalPlan, agent, targetState?.policy ?? this.policyForAgent(agent), {
      owner: 'current-turn',
      stability: 'whole-surface',
      contextWindow: targetState?.spec.contextWindow,
    }, signal)
  }

  compactNow(agent, signal, sourceCommandId) {
    signal.throwIfAborted()
    try {
      return agent.runMaintenance(async (agentSignal) => {
        const operationSignal = AbortSignal.any([agentSignal, signal])
        try {
          operationSignal.throwIfAborted()
          assertNoActiveCompaction(agent.session, 'manual compaction')
          const targetState = await this.resolveTargetState(agent, operationSignal, 'manual')
          operationSignal.throwIfAborted()
          const retainTokens = targetState?.spec.retainTokens ?? 0
          const provisional = planCompaction(
            agent.session,
            this.ctx.tokenMeter.measure(agent.session),
            { retainTokens },
          )
          if (provisional === null) return null
          const eventCountBeforePrune = agent.session.events.length
          let pruneFailure
          try {
            selectivePrune(this.ctx, agent.session, provisional, 'pressure', operationSignal)
          } catch (error) {
            pruneFailure = error
          }
          if (agent.session.events.length > eventCountBeforePrune) {
            try {
              await this.ctx.sessions.flush(agent.session)
            } catch (error) {
              throw new ManualCompactionError(
                'persistence',
                'manual compaction prune durability checkpoint failed',
                { cause: error },
              )
            }
          }
          if (pruneFailure !== undefined) throw pruneFailure
          operationSignal.throwIfAborted()
          const finalPlan = planCompaction(
            agent.session,
            this.ctx.tokenMeter.measure(agent.session),
            { retainTokens },
          )
          if (finalPlan === null) return null
          return await this.execute(finalPlan, agent, targetState?.policy ?? this.policyForAgent(agent), {
            owner: null,
            stability: 'selected-span',
            sourceCommandId,
            contextWindow: targetState?.spec.contextWindow,
            flush: async () => this.ctx.sessions.flush(agent.session),
          }, operationSignal)
        } catch (error) {
          if (agentSignal.aborted && operationSignal.reason === agentSignal.reason) {
            throw new ManualCompactionError(
              'cancelled',
              'manual compaction was cancelled',
              { cause: error },
            )
          }
          operationSignal.throwIfAborted()
          throw error
        }
      })
    } catch (error) {
      throw new ManualCompactionError(
        'busy',
        'manual compaction requires an idle agent with no waking queued work',
        { cause: error },
      )
    }
  }

  registerAutomaticCompaction() {
    const logResult = (result, trigger) => {
      this.ctx.logger.info(
        `anchored compaction (${trigger}): shadowed ${result.shadowedSeqs.length} surface nodes `
        + `(seqs ${result.shadowedRange.start}-${result.shadowedRange.end}, ~${result.shadowedTokenCount} tokens)`,
      )
    }

    this.ctx.on('agent/pre-step', async ({ agent, messages, signal }, next) => {
      if (!signal.aborted) {
        try {
          const result = await this.compactAutomatic(agent, 'pressure', signal, messages)
          if (result !== null) logResult(result, 'step pressure')
        } catch (error) {
          if (error instanceof TargetPressureConfigError) {
            if (this.warnedPressureTargets.has(error.targetKey)) return next()
            this.warnedPressureTargets.add(error.targetKey)
          }
          this.ctx.logger.warn(`anchored step compaction failed: ${errorMessage(error)}; continuing the turn`)
        }
      }
      return next()
    })

    this.ctx.on('agent/status', ({ agent, status }) => {
      if (status === 'idle') this.overflowRetries.delete(agent)
    })

    this.ctx.on('session/event', (session, event) => {
      if (event.type !== 'assistant/message') return
      const agent = this.overflowAgents.get(session)
      if (agent !== undefined) this.overflowRetries.delete(agent)
    })

    this.ctx.on('agent/request-error', async ({ agent, failure, signal }, next) => {
      if (failure.code !== CONTEXT_WINDOW_EXCEEDED_CODE || signal.aborted) return next()
      this.overflowAgents.set(agent.session, agent)
      const target = routedTarget(agent.session)
      if (target === undefined) return next()
      const policy = resolveTargetPolicy(this.config, target)
      const retries = this.overflowRetries.get(agent) ?? 0
      if (retries >= policy.maxOverflowRetries) return next()
      const generation = agent.session.surface.replaceGeneration
      let result
      try {
        result = await this.compactAutomatic(agent, 'context-overflow', signal, [], true)
      } catch (error) {
        if (!signal.aborted && agent.session.surface.replaceGeneration > generation) {
          this.ctx.logger.warn(
            `anchored context-overflow recovery failed after durable surface progress: ${errorMessage(error)}; retrying replacement surface`,
          )
          this.overflowRetries.set(agent, retries + 1)
          return { kind: 'retry' }
        }
        this.ctx.logger.warn(
          `anchored context-overflow recovery failed: ${errorMessage(error)}; ${signal.aborted
            ? 'cancellation prevents retry'
            : 'preserving the original request error'}`,
        )
        return next()
      }
      if (signal.aborted || agent.session.surface.replaceGeneration <= generation) return next()
      if (result !== null) logResult(result, 'context overflow recovery')
      this.overflowRetries.set(agent, retries + 1)
      return { kind: 'retry' }
    })
  }

  async compactAutomatic(agent, trigger, signal, pendingMessages, canonicalOverflow = false) {
    if (trigger === 'context-overflow' && !canonicalOverflow) {
      throw new Error('anchored compaction: non-canonical context-overflow trigger rejected')
    }
    const targetState = await this.resolveTargetState(agent, signal, trigger)
    if (targetState === null) return null
    assertNoActiveCompaction(agent.session, `automatic ${trigger} compaction`)
    const pendingTokens = estimatePending(this.ctx.tokenMeter, pendingMessages)
    let measurement = this.ctx.tokenMeter.measure(agent.session)
    if (trigger === 'pressure'
      && measurement.totalTokens + pendingTokens < targetState.spec.thresholdTokens) return null

    const attempts = trigger === 'pressure' ? targetState.policy.compactionRetries + 1 : 1
    let latest = null
    for (let attempt = 0; attempt < attempts; attempt += 1) {
      signal.throwIfAborted()
      const provisional = planCompaction(agent.session, measurement, {
        retainTokens: targetState.spec.retainTokens,
      })
      if (provisional === null) return latest
      selectivePrune(this.ctx, agent.session, provisional, trigger, signal)
      signal.throwIfAborted()
      measurement = this.ctx.tokenMeter.measure(agent.session)
      if (trigger === 'pressure'
        && measurement.totalTokens + pendingTokens < targetState.spec.thresholdTokens) return latest
      const finalPlan = planCompaction(agent.session, measurement, {
        retainTokens: targetState.spec.retainTokens,
      })
      if (finalPlan === null) return latest
      latest = await this.execute(finalPlan, agent, targetState.policy, {
        owner: 'current-turn',
        stability: 'whole-surface',
        contextWindow: targetState.spec.contextWindow,
        projectedExtraTokens: pendingTokens,
      }, signal)
      measurement = this.ctx.tokenMeter.measure(agent.session)
      if (trigger === 'context-overflow'
        || measurement.totalTokens + pendingTokens < targetState.spec.thresholdTokens) return latest
    }
    throw new Error(
      `anchored compaction remains above threshold after ${attempts} attempts `
      + `(${measurement.totalTokens + pendingTokens} projected tokens >= ${targetState.spec.thresholdTokens})`,
    )
  }

  execute(plan, agent, policy, options, signal) {
    return executeCompaction({
      meter: this.ctx.tokenMeter,
      config: policy,
      summarize: (input, owner, abort) => this.summarize(input, owner, abort, policy),
    }, agent.session, plan, agent, options, signal)
  }

  policyForAgent(agent) {
    const target = routedTarget(agent.session) ?? conversationTarget(agent)
    return target === undefined ? this.config : resolveTargetPolicy(this.config, target)
  }

  async resolveTargetState(agent, signal, purpose) {
    const target = routedTarget(agent.session) ?? conversationTarget(agent)
    if (target === undefined) {
      if (purpose === 'pressure' || purpose === 'context-overflow') return null
      throw new TargetPressureConfigError(
        'unrouted',
        `anchored compaction: no provider/model target is available for ${purpose} compaction`,
      )
    }
    const policy = resolveTargetPolicy(this.config, target)
    const info = await this.ctx.llm.resolveModelInfo(target.provider, target.model, signal)
    const contextWindow = info?.context?.contextWindow
    if (!Number.isSafeInteger(contextWindow) || contextWindow <= 0) {
      if (purpose === 'pressure') {
        throw new TargetPressureConfigError(
          `${target.provider}/${target.model}`,
          `anchored compaction: no context capacity for ${target.provider}/${target.model}`,
        )
      }
      // Canonical overflow cannot prove protected-context fit without the
      // provider capacity, so it must preserve the original error. Manual and
      // direct requests fail before maintenance for the same reason.
      if (purpose === 'context-overflow') return null
      throw new TargetPressureConfigError(
        `${target.provider}/${target.model}`,
        `anchored compaction: no context capacity for ${target.provider}/${target.model}`,
      )
    }
    // The control-plane threshold governs only automatic pressure. Manual and
    // canonical-overflow entries share retention/invariants but do not need a
    // user threshold to authorize work, so an unusually low token threshold
    // cannot disable recovery or /compact.
    const control = purpose === 'pressure' ? this.controlPolicy(agent.session) : null
    const spec = resolveCompactSpec(policy, contextWindow, control)
    return { policy, spec }
  }

  controlPolicy(session) {
    const service = lookupControlPolicyService(this.ctx)
    if (service === undefined || typeof service.resolve !== 'function') return null
    try {
      const value = service.resolve(session)
      if (value !== null && normalizeControlPolicy(value) === undefined) {
        if (!this.warnedPolicySessions.has(session)) {
          this.warnedPolicySessions.add(session)
          this.ctx.logger.warn('anchored compaction: invalid control-plane policy; using backend defaults')
        }
        return undefined
      }
      return value
    } catch (error) {
      if (!this.warnedPolicySessions.has(session)) {
        this.warnedPolicySessions.add(session)
        this.ctx.logger.warn(`anchored compaction: policy resolve failed: ${errorMessage(error)}; using defaults`)
      }
      return undefined
    }
  }
}

function lookupControlPolicyService(ctx) {
  const seen = new Set()
  for (const candidate of [ctx, ctx?.root, ctx?.reflect]) {
    if (candidate === undefined || candidate === null || seen.has(candidate)) continue
    seen.add(candidate)
    if (typeof candidate.get !== 'function') continue
    try {
      const value = candidate.get('kuanfuCompactionPolicy')
      if (value !== undefined && typeof value.resolve === 'function') return value
    } catch {
      // A rebound isolate context may refuse names outside its realm.
    }
  }
  return undefined
}

function routedTarget(session) {
  const config = session.requestHeader()?.config
  if (typeof config?.provider !== 'string' || config.provider.length === 0
    || typeof config?.model !== 'string' || config.model.length === 0) return undefined
  return { provider: config.provider, model: config.model }
}

function conversationTarget(agent) {
  if (typeof agent.options?.provider !== 'string' || agent.options.provider.length === 0
    || typeof agent.options?.model !== 'string' || agent.options.model.length === 0) return undefined
  return { provider: agent.options.provider, model: agent.options.model }
}

function estimatePending(meter, messages) {
  if (!Array.isArray(messages)) return 0
  return messages.reduce((sum, message) => sum + meter.estimateMessage(message), 0)
}

function assertCurrentTurnOpen(session) {
  for (let index = session.events.length - 1; index >= 0; index -= 1) {
    const event = session.events[index]
    if (event.type === 'turn/start') return
    if (event.type === 'turn/end') break
  }
  throw new Error('compactRegion: no open turn')
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error)
}

export { lookupControlPolicyService }
export { AnchoredCompactionEngine as BasicCompactionEngine }
export default AnchoredCompactionEngine
