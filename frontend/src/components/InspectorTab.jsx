import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, BarChart, Bar, Cell, Legend,
} from 'recharts'
import EvaluationTable from './EvaluationTable'
import { useEffect, useRef, useState } from 'react'

/* ── Mini force-directed SVG entity graph ────────────────────────────── */
function EntityGraph({ graphData }) {
  const svgRef = useRef(null)

  useEffect(() => {
    if (!svgRef.current || !graphData) return
    const svg = svgRef.current
    const W = svg.clientWidth || 700
    const H = 250

    const nodes = (graphData.nodes || []).slice(0, 40)
    const edges = (graphData.edges || []).slice(0, 80)
    if (!nodes.length) return

    nodes.forEach((n, i) => {
      const angle = (2 * Math.PI * i) / nodes.length - Math.PI / 2
      const r = Math.min(W, H) * 0.32
      n.x = W / 2 + r * Math.cos(angle)
      n.y = H / 2 + r * Math.sin(angle)
      n.vx = 0; n.vy = 0
    })
    const byId = Object.fromEntries(nodes.map(n => [n.id, n]))

    for (let iter = 0; iter < 80; iter++) {
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x, dy = nodes[i].y - nodes[j].y
          const dist = Math.sqrt(dx * dx + dy * dy) || 1
          const f = 500 / (dist * dist)
          nodes[i].vx += dx / dist * f; nodes[i].vy += dy / dist * f
          nodes[j].vx -= dx / dist * f; nodes[j].vy -= dy / dist * f
        }
      }
      edges.forEach(e => {
        const a = byId[e.source], b = byId[e.target]
        if (!a || !b) return
        const dx = b.x - a.x, dy = b.y - a.y
        const dist = Math.sqrt(dx * dx + dy * dy) || 1
        const f = dist * 0.007
        a.vx += dx / dist * f; a.vy += dy / dist * f
        b.vx -= dx / dist * f; b.vy -= dy / dist * f
      })
      nodes.forEach(n => {
        n.vx += (W / 2 - n.x) * 0.003; n.vy += (H / 2 - n.y) * 0.003
        n.vx *= 0.75; n.vy *= 0.75
        n.x = Math.max(20, Math.min(W - 20, n.x + n.vx))
        n.y = Math.max(16, Math.min(H - 16, n.y + n.vy))
      })
    }

    let html = ''
    edges.forEach(e => {
      const a = byId[e.source], b = byId[e.target]
      if (!a || !b) return
      html += `<line x1="${a.x.toFixed(1)}" y1="${a.y.toFixed(1)}" x2="${b.x.toFixed(1)}" y2="${b.y.toFixed(1)}" stroke="#D4D4D8" stroke-width="${0.8 + (e.weight || 1) * 0.4}" stroke-opacity="0.7"/>`
    })
    nodes.forEach(n => {
      const r = 9 + Math.min(n.entity_count * 1.4, 10)
      html += `<circle cx="${n.x.toFixed(1)}" cy="${n.y.toFixed(1)}" r="${r.toFixed(1)}" fill="rgba(255,230,0,0.20)" stroke="#111" stroke-width="1.4"/>
        <text x="${n.x.toFixed(1)}" y="${(n.y + 4).toFixed(1)}" text-anchor="middle" fill="#111" font-size="9.5" font-weight="600" font-family="Inter,sans-serif">${n.label}</text>`
    })
    svg.innerHTML = html
  }, [graphData])

  if (!graphData || !(graphData.nodes?.length)) {
    return <div style={{ textAlign: 'center', padding: '48px 0', color: 'var(--text-muted)', fontSize: 13 }}>No entity connections detected.</div>
  }

  return (
    <svg
      ref={svgRef}
      className="entity-graph-svg"
      viewBox={`0 0 700 250`}
      preserveAspectRatio="xMidYMid meet"
    />
  )
}

/* ── Stage rows ──────────────────────────────────────────────────────── */
function StageRow({ num, name, detail }) {
  return (
    <div className="stage-row-inspector">
      <div className="stage-num-badge">{num}</div>
      <div>
        <div className="stage-name-inspector">{name}</div>
        {detail && (
          <div
            className="stage-detail-inspector"
            dangerouslySetInnerHTML={{ __html: detail }}
          />
        )}
      </div>
    </div>
  )
}

/* ── Chart tooltip ───────────────────────────────────────────────────── */
const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: '#fff', border: '1px solid var(--border)',
      borderRadius: 8, padding: '8px 12px',
      fontSize: 12, boxShadow: 'var(--shadow)',
    }}>
      <div style={{ fontWeight: 700, marginBottom: 4, color: 'var(--gray-700)' }}>{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color }}>
          {p.name}: <strong>{typeof p.value === 'number' ? p.value.toFixed(4) : p.value}</strong>
        </div>
      ))}
    </div>
  )
}

const S3_BOUNDARY_MEANINGS = {
  hard: 'Strong topic or structure shift. Stage 3 kept this boundary.',
  soft: 'Moderate shift. Stage 3 kept it, but later stages may still merge it.',
  merged: 'Weak boundary. Stage 3 merged this chunk with the next candidate.',
  end: 'Final chunk in the method output.',
  single: 'Only one chunk was available for this method.',
}

function fmtPct(value) {
  const n = Number(value || 0)
  return `${Math.round(n * 100)}%`
}

function Stage3ChunkRow({ chunk, thresholds }) {
  const [open, setOpen] = useState(false)
  const type = chunk.boundary_type || 'unknown'
  const signal = Number(chunk.boundary_signal || 0)
  const reason = chunk.merge_reason || ''
  const isProtected = reason === 'protected_structure_boundary'

  return (
    <div className={`s3-chunk-row ${type}`} onClick={() => setOpen(o => !o)}>
      <div className="s3-chunk-main">
        <div className="s3-chunk-index">S3-{chunk.index}</div>
        <div className="s3-chunk-text">
          <div className="s3-chunk-title">
            <span className={`s3-boundary-pill ${type}`}>{type.replace(/_/g, ' ')}</span>
            {isProtected && <span className="s3-boundary-pill protected">protected</span>}
            <span>{chunk.token_count} tokens</span>
            <span>signal {fmtPct(signal)}</span>
          </div>
          <div className="s3-chunk-preview">{chunk.preview || 'No preview available.'}</div>
        </div>
      </div>

      <div className={`s3-chunk-details${open ? ' open' : ''}`}>
        <div className="s3-meaning">{S3_BOUNDARY_MEANINGS[type] || 'Boundary label from Stage 3.'}</div>
        <div className="s3-feature-grid">
          {[
            ['Selected metric', chunk.features?.selected_metric],
            ['JSD', chunk.features?.jsd],
            ['Hellinger', chunk.features?.hellinger],
            ['Token overlap', chunk.features?.overlap],
            ['Entropy delta', chunk.features?.entropy_delta],
            ['Low threshold', thresholds?.low],
            ['High threshold', thresholds?.high],
          ].map(([label, value]) => (
            <div className="s3-feature" key={label}>
              <div>{Number(value || 0).toFixed(4)}</div>
              <span>{label}</span>
            </div>
          ))}
        </div>
        {reason && (
          <div className="s3-reason">Decision reason: <strong>{reason.replace(/_/g, ' ')}</strong></div>
        )}
      </div>
    </div>
  )
}

/* ── Strategy score bar chart ────────────────────────────────────────── */
function StrategyScoreChart({ scores }) {
  if (!scores || Object.keys(scores).length === 0) return null
  const data = Object.entries(scores)
    .map(([name, s]) => ({ name: name.replace(/_/g, ' '), score: s.score }))
    .sort((a, b) => b.score - a.score)

  return (
    <div className="card">
      <div className="card-title">Strategy Score Comparison</div>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 16, left: -8, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--gray-200)" vertical={false} />
            <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
            <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
            <Tooltip content={<ChartTooltip />} />
            <Bar dataKey="score" name="Score" radius={[4, 4, 0, 0]}>
              {data.map((entry, i) => (
                <Cell key={i} fill={i === 0 ? '#FFE600' : '#E4E4E7'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

/* ── Main inspector ───────────────────────────────────────────────────── */
export default function InspectorTab({ results, config }) {
  if (!results) {
    return (
      <div className="empty-state">
        <div className="empty-icon">◈</div>
        <p>Run the pipeline to inspect each stage.</p>
      </div>
    )
  }

  const details  = results.stage_details || {}
  const s8       = details.s8 || {}
  const s7       = details.s7 || {}
  const evalScores = s8.scores || {}
  const perStrategy = details.s3_s6 || {}
  const s7Strategies = s7.strategies || {}
  const strategyNames = Object.keys(perStrategy)
  const defaultStrategy = s7.winner || s7.strategy || s8.winner || strategyNames[0] || ''
  const [activeStrategy, setActiveStrategy] = useState(defaultStrategy)
  const selectedStrategy = strategyNames.includes(activeStrategy) ? activeStrategy : defaultStrategy
  const selectedDetails = perStrategy[selectedStrategy] || {}
  const selectedS7 = s7Strategies[selectedStrategy] || {}
  const selectedS3Chunks = selectedDetails.s3_chunks || []
  const selectedS3Stats = selectedDetails.s3_stats || {}

  const getS3S6Detail = name => {
    const d = perStrategy[name]
    if (!d) return ''
    return `${d.chunk_count} chunks · S3 ${d.s3_chunk_count ?? '—'} · S4 ${d.s4_chunk_count ?? '—'} · Metric: ${(d.metric || 'jsd').toUpperCase()}`
  }

  const stages = [
    {
      num: 'S1', name: 'Document Profiler',
      detail: details.s1 ? `Type: <strong>${details.s1.type}</strong> · Domain: ${details.s1.domain} · ${details.s1.token_count} tokens` : '',
    },
    {
      num: 'S2', name: 'Parallel Chunkers',
      detail: details.s2
        ? Object.entries(details.s2.strategies || {}).map(([k, v]) => `${k}: ${v}`).join(' · ')
          + ` · Evaluating: <strong>${(details.s2.evaluating || []).join(', ')}</strong>`
        : '',
    },
    {
      num: 'S3–S6', name: 'Full Pipeline Per Strategy',
      detail: strategyNames.map(n => `${n}: ${getS3S6Detail(n)}`).join('<br/>'),
    },
    {
      num: 'S8', name: 'Strategy Evaluator',
      detail: s8.winner
        ? `Winner: <strong>${s8.winner}</strong> · Ranked: ${(s8.ranked || []).map(([n, sc]) => `${n} (${(sc * 100).toFixed(1)}%)`).join(' › ')}`
        : '',
    },
    {
      num: 'S7', name: 'RL Reward Calibration',
      detail: Object.keys(s7Strategies).length
        ? `Ran on <strong>${Object.keys(s7Strategies).length}</strong> methods · Winner: <strong>${(s7.winner || s7.strategy || '').replace(/_/g, ' ')}</strong> · Final: ${(results.reward_history?.at(-1) || 0).toFixed(4)}`
        : s7.iterations
        ? `${s7.iterations} iterations on <strong>${s7.strategy}</strong> · Final: ${(results.reward_history?.at(-1) || 0).toFixed(4)}`
        : '',
    },
  ]

  const winnerName = s7.winner || s7.strategy || s8.winner
  const selectedJsd = selectedDetails.jsd_series || details.s3?.jsd_series || []

  const jsdData = selectedJsd.map((v, i) => ({ idx: `C${i}`, score: v }))
  const activeRewardHistory = selectedS7.reward_history || results.reward_history || []
  const rlData  = activeRewardHistory.map((v, i) => ({ iter: `Iter ${i}`, reward: v }))

  const tauLow  = config.tau_jsd_low  ?? 0.15
  const tauHigh = config.tau_jsd_high ?? 0.45

  const breakdown = selectedS7.reward_breakdown || s7.reward_breakdown || {}
  const graphData = details.s5?.entity_graphs?.[selectedStrategy] || details.s5?.entity_graph || null

  return (
    <div>
      {/* Stage overview */}
      <div className="card" style={{ marginBottom: 14 }}>
        <div className="card-title">Pipeline Stage Overview</div>
        <div className="stage-list-inspector">
          {stages.map(s => <StageRow key={s.num} {...s} />)}
        </div>
      </div>

      {/* Strategy score bar + evaluation table */}
      <StrategyScoreChart scores={evalScores} />

      {s8.table?.length > 0 && (
        <div style={{ marginTop: 14 }}>
          <EvaluationTable table={s8.table} winner={s8.winner} />
        </div>
      )}

      {s7.table?.length > 0 && (
        <div style={{ marginTop: 14 }}>
          <EvaluationTable
            table={s7.table}
            winner={s7.winner || s7.strategy}
            title="S7 RL Evaluation"
            note="This ranks the chunking methods after RL calibration. The score is the final multi-objective reward for each method."
          />
        </div>
      )}

      {strategyNames.length > 0 && (
        <div className="card" style={{ marginTop: 14 }}>
          <div className="card-title">
            Method Results
            {winnerName && <span className="badge">Winner: {winnerName.replace(/_/g, ' ')}</span>}
          </div>
          <div className="method-tabs">
            {strategyNames.map(name => (
              <button
                key={name}
                className={`method-tab${selectedStrategy === name ? ' active' : ''}`}
                onClick={() => setActiveStrategy(name)}
              >
                {name.replace(/_/g, ' ')}
              </button>
            ))}
          </div>
          <div className="method-detail-grid">
            {[
              ['Initial', selectedDetails.initial_chunk_count],
              ['After S3', selectedDetails.s3_chunk_count],
              ['S3 merged', selectedDetails.s3_merge_count],
              ['S3 hard', selectedDetails.s3_hard_count],
              ['Protected', selectedDetails.s3_protected_count],
              ['After S4', selectedDetails.s4_chunk_count],
              ['Final', selectedDetails.chunk_count],
              ['Mean tokens', selectedDetails.mean_tokens],
              ['Entities', selectedDetails.entity_count],
              ['Embedding dim', selectedDetails.embedding_dim],
              ['Boundary', selectedDetails.mean_boundary_score != null ? (selectedDetails.mean_boundary_score * 100).toFixed(1) + '%' : '—'],
              ['S7 reward', selectedS7.final_reward != null ? selectedS7.final_reward.toFixed(4) : '—'],
            ].map(([label, value]) => (
              <div className="method-detail-card" key={label}>
                <div className="method-detail-value">{value ?? '—'}</div>
                <div className="method-detail-label">{label}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {selectedS3Chunks.length > 0 && (
        <div className="card" style={{ marginTop: 14 }}>
          <div className="card-title">
            Stage 3 Boundary Refinement
            {selectedStrategy && <span className="badge">{selectedStrategy.replace(/_/g, ' ')}</span>}
          </div>
          <div className="s3-explain-grid">
            <div>
              <strong>Hard</strong>
              <span>clear boundary kept</span>
            </div>
            <div>
              <strong>Soft</strong>
              <span>uncertain boundary kept for S4</span>
            </div>
            <div>
              <strong>Merged</strong>
              <span>weak boundary removed</span>
            </div>
            <div>
              <strong>Protected</strong>
              <span>article/section boundary preserved</span>
            </div>
          </div>
          <div className="s3-summary-strip">
            {[
              ['Initial', selectedS3Stats.initial_count],
              ['After S3', selectedS3Stats.final_count],
              ['Merged', selectedS3Stats.merged_count],
              ['Hard', selectedS3Stats.hard_count],
              ['Soft', selectedS3Stats.soft_count],
              ['Protected', selectedS3Stats.protected_count],
              ['Mean signal', selectedS3Stats.mean_signal != null ? fmtPct(selectedS3Stats.mean_signal) : '—'],
            ].map(([label, value]) => (
              <div key={label}>
                <strong>{value ?? '—'}</strong>
                <span>{label}</span>
              </div>
            ))}
          </div>
          <div className="s3-chunk-list">
            {selectedS3Chunks.map(chunk => (
              <Stage3ChunkRow
                key={`${selectedStrategy}-${chunk.index}`}
                chunk={chunk}
                thresholds={selectedDetails.thresholds}
              />
            ))}
          </div>
        </div>
      )}

      {/* Charts */}
      <div className="inspector-grid" style={{ marginTop: 14 }}>
        {/* JSD */}
        <div className="card">
          <div className="card-title">
            Boundary Signal
            {selectedStrategy && <span className="badge">{selectedStrategy.replace(/_/g, ' ')}</span>}
          </div>
          <div className="chart-wrap">
            {jsdData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={jsdData} margin={{ top: 4, right: 16, left: -8, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--gray-200)" />
                  <XAxis dataKey="idx" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                  <Tooltip content={<ChartTooltip />} />
                  <ReferenceLine y={tauLow}  stroke="var(--green)" strokeDasharray="4 3" label={{ value: `τ⁻ ${tauLow}`, position: 'right', fontSize: 9, fill: 'var(--green)' }} />
                  <ReferenceLine y={tauHigh} stroke="var(--amber)" strokeDasharray="4 3" label={{ value: `τ⁺ ${tauHigh}`, position: 'right', fontSize: 9, fill: 'var(--amber)' }} />
                  <Line type="monotone" dataKey="score" name="Boundary signal" stroke="#111" strokeWidth={2} dot={{ r: 2, fill: '#FFE600', stroke: '#111', strokeWidth: 1 }} activeDot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-state" style={{ padding: '32px 0' }}>
                <p>No JSD series available.</p>
              </div>
            )}
          </div>
        </div>

        {/* RL reward */}
        <div className="card">
          <div className="card-title">
            RL Reward Curve
            {selectedStrategy && <span className="badge">{selectedStrategy.replace(/_/g, ' ')}</span>}
          </div>
          <div className="chart-wrap">
            {rlData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={rlData} margin={{ top: 4, right: 16, left: -8, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--gray-200)" />
                  <XAxis dataKey="iter" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                  <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                  <Tooltip content={<ChartTooltip />} />
                  <Line type="monotone" dataKey="reward" name="Reward" stroke="#111" strokeWidth={2.5} dot={{ r: 3, fill: '#FFE600', stroke: '#111', strokeWidth: 1 }} activeDot={{ r: 5 }} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-state" style={{ padding: '32px 0' }}>
                <p>No reward history available.</p>
              </div>
            )}
          </div>
          {Object.keys(breakdown).length > 0 && (
            <div style={{ marginTop: 12, fontSize: 11, color: 'var(--text-muted)', display: 'flex', gap: 14, flexWrap: 'wrap' }}>
              {Object.entries(breakdown).filter(([k]) => k !== 'total').map(([k, v]) => (
                <span key={k}>
                  <strong style={{ color: 'var(--text-secondary)', textTransform: 'capitalize' }}>{k}:</strong> {(v * 100).toFixed(1)}%
                </span>
              ))}
              {breakdown.total != null && (
                <span style={{ fontWeight: 700, color: 'var(--yellow-dark)' }}>Total: {breakdown.total.toFixed(4)}</span>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Embedding info */}
      {(details.s6 || selectedDetails.embedding_dim) && (
        <div className="card" style={{ marginTop: 14 }}>
          <div className="card-title">Embedding Ensemble</div>
          <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
            {details.s6.ensemble_models?.length
              ? <>Models: <strong>{details.s6.ensemble_models.join(', ')}</strong> · {selectedStrategy ? <>Selected method: <strong>{selectedStrategy.replace(/_/g, ' ')}</strong> · </> : null}Projection dim: {selectedDetails.embedding_dim || details.s6.embedding_dim}</>
              : <>Fallback model: {details.s6.model || 'n/a'}</>
            }
          </div>
        </div>
      )}

      {/* Entity graph */}
      <div className="card" style={{ marginTop: 14 }}>
        <div className="card-title">
          Entity Network
          {selectedStrategy && <span className="badge">{selectedStrategy.replace(/_/g, ' ')}</span>}
        </div>
        <EntityGraph graphData={graphData} />
        <div style={{ marginTop: 8, textAlign: 'center', fontSize: 11, color: 'var(--text-muted)' }}>
          Nodes = chunks · Edges = shared named entities · Node size ∝ entity count
        </div>
      </div>
    </div>
  )
}
