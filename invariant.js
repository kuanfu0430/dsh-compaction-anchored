const PACKAGE_NAME = '@deepseek-ai/dsh-compaction-basic'

export const name = 'compaction-anchored-invariant'
export const inject = ['invariants']

// The authoritative compaction event and surface invariants remain owned by
// @deepseek-ai/dsh-compaction and @deepseek-ai/dsh-session. This companion only
// reserves the aliased provider package identity, matching the upstream seam.
const install = () => {}

export const apply = (ctx) => Promise.resolve(ctx.invariants.register(PACKAGE_NAME, install))
