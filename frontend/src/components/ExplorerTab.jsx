import { useState } from 'react'

function esc(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function ChunkCard({ chunk }) {
  const [open, setOpen] = useState(false)
  const [expanded, setExpanded] = useState(false)

  const score = chunk.chunk_score || 0
  const qClass = score >= 0.75 ? 'q-high' : score >= 0.5 ? 'q-med' : 'q-low'
  const scoreClass = score >= 0.75 ? 'score-high' : score >= 0.5 ? 'score-med' : 'score-low'
  const text = chunk.text || ''
  const MAX = 280
  const truncated = !expanded && text.length > MAX
  const displayText = truncated ? text.slice(0, MAX) + '…' : text

  const entities = (chunk.entities || []).slice(0, 8)
  const neighbors = (chunk.graph_neighbors || []).join(', ') || '—'

  return (
    <div className={`chunk-card ${qClass}`} onClick={() => setOpen(o => !o)}>
      {/* Header */}
      <div className="chunk-header">
        <div className="chunk-meta">
          <span className="chunk-idx">#{chunk.chunk_index}</span>
          <span className="metric-pill">🔤 {chunk.token_count || 0}</span>
          <span className="metric-pill">JSD {(chunk.jsd_score || 0).toFixed(3)}</span>
          <span className="metric-pill">BLEU {(chunk.boundary_score || 0).toFixed(3)}</span>
          {chunk.boundary_type && (
            <span className="metric-pill">{chunk.boundary_type}</span>
          )}
        </div>
        <span className={`score-badge ${scoreClass}`}>
          {(score * 100).toFixed(1)}%
        </span>
      </div>

      {/* Text */}
      <div className="chunk-text">{displayText}</div>
      {truncated && (
        <span
          className="chunk-expand"
          onClick={e => { e.stopPropagation(); setExpanded(true) }}
        >
          ▼ show more
        </span>
      )}

      {/* Entities */}
      {entities.length > 0 && (
        <div className="entity-tags">
          {entities.map((e, i) => (
            <span className="entity-tag" key={i}>
              {e.text} <span style={{ opacity: 0.6, fontSize: 9 }}>{e.label}</span>
            </span>
          ))}
        </div>
      )}

      {/* Expandable detail */}
      <div className={`chunk-detail${open ? ' open' : ''}`}>
        <div className="detail-grid">
          <div className="detail-item">
            <div className="detail-key">Score</div>
            <div className="detail-val">{(score * 100).toFixed(2)}%</div>
          </div>
          <div className="detail-item">
            <div className="detail-key">JSD</div>
            <div className="detail-val">{(chunk.jsd_score || 0).toFixed(4)}</div>
          </div>
          <div className="detail-item">
            <div className="detail-key">Boundary</div>
            <div className="detail-val">{(chunk.boundary_score || 0).toFixed(4)}</div>
          </div>
          <div className="detail-item">
            <div className="detail-key">ICC</div>
            <div className="detail-val">{(chunk.icc || 0).toFixed(4)}</div>
          </div>
          <div className="detail-item">
            <div className="detail-key">Type</div>
            <div className="detail-val">{chunk.boundary_type || '—'}</div>
          </div>
          <div className="detail-item">
            <div className="detail-key">Method</div>
            <div className="detail-val">{chunk.method || '—'}</div>
          </div>
          <div className="detail-item">
            <div className="detail-key">Neighbors</div>
            <div className="detail-val">{neighbors}</div>
          </div>
          <div className="detail-item">
            <div className="detail-key">Entities</div>
            <div className="detail-val">{(chunk.entities || []).length}</div>
          </div>
        </div>
        {chunk.context_header && (
          <div className="context-header-box">{chunk.context_header}</div>
        )}
      </div>
    </div>
  )
}

export default function ExplorerTab({ chunks }) {
  const [query, setQuery]       = useState('')
  const [qualFilter, setQualFilter] = useState('')
  const [boundFilter, setBoundFilter] = useState('')

  if (!chunks.length) {
    return (
      <div className="empty-state">
        <div className="empty-icon">🔍</div>
        <p>Run the pipeline to explore chunks.</p>
      </div>
    )
  }

  const filtered = chunks.filter(c => {
    const score = c.chunk_score || 0
    const qual  = score >= 0.75 ? 'high' : score >= 0.5 ? 'med' : 'low'
    const text  = (c.text || '').toLowerCase()
    return (
      (!query || text.includes(query.toLowerCase())) &&
      (!qualFilter || qual === qualFilter) &&
      (!boundFilter || (c.boundary_type || '') === boundFilter)
    )
  })

  return (
    <div>
      <div className="explorer-controls">
        <input
          className="search-input"
          type="text"
          placeholder="Search chunk text…"
          value={query}
          onChange={e => setQuery(e.target.value)}
        />
        <select className="filter-select" value={qualFilter} onChange={e => setQualFilter(e.target.value)}>
          <option value="">All quality</option>
          <option value="high">High (≥ 75%)</option>
          <option value="med">Medium (50–74%)</option>
          <option value="low">Low (&lt; 50%)</option>
        </select>
        <select className="filter-select" value={boundFilter} onChange={e => setBoundFilter(e.target.value)}>
          <option value="">All boundaries</option>
          <option value="hard">Hard</option>
          <option value="soft">Soft</option>
          <option value="merged">Merged</option>
          <option value="end">End</option>
        </select>
        <span className="count-label">
          {filtered.length} / {chunks.length} chunk{chunks.length !== 1 ? 's' : ''}
        </span>
      </div>

      {filtered.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">🔍</div>
          <p>No chunks match the filter.</p>
        </div>
      ) : (
        <div className="chunk-list">
          {filtered.map(c => <ChunkCard key={c.chunk_index} chunk={c} />)}
        </div>
      )}
    </div>
  )
}
