function ExportCard({ icon, title, desc, label, onClick, disabled }) {
  return (
    <div className={`export-card${disabled ? ' disabled' : ''}`} style={disabled ? { opacity: 0.5, pointerEvents: 'none' } : {}}>
      <div className="export-icon">{icon}</div>
      <div className="export-title">{title}</div>
      <div className="export-desc">{desc}</div>
      <button className="btn btn-primary" onClick={onClick} disabled={disabled}>
        {label}
      </button>
    </div>
  )
}

export default function ExportTab({ jobId, results, apiBase }) {
  if (!results || !jobId) {
    return (
      <div className="empty-state">
        <div className="empty-icon">↓</div>
        <p>Run the pipeline first to enable export.</p>
      </div>
    )
  }

  const s = results.summary || {}

  function doExport(fmt) {
    window.open(`${apiBase}/export/${jobId}/${fmt}`, '_blank')
  }

  return (
    <div>
      {/* Summary bar */}
      <div className="card" style={{ marginBottom: 16, display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
        <div style={{ flex: 1, minWidth: 200 }}>
          <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: 4 }}>
            Ready to export
          </div>
          <div style={{ fontSize: 22, fontWeight: 800, color: 'var(--gray-900)', letterSpacing: '-0.02em' }}>
            {s.chunk_count} chunks
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2 }}>
            {s.doc_type} · {s.domain} · strategy: <strong>{(s.winning_strategy || '').replace(/_/g, ' ')}</strong>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          {[
            ['Score', ((s.mean_chunk_score || 0) * 100).toFixed(1) + '%'],
            ['Tokens', (s.token_count || 0).toLocaleString()],
            ['RL iters', s.rl_iterations ?? '—'],
          ].map(([label, val]) => (
            <div key={label} style={{
              textAlign: 'center',
              padding: '10px 16px',
              background: 'var(--yellow-light)',
              borderRadius: 'var(--radius)',
              border: '1px solid rgba(255,230,0,0.35)',
              minWidth: 80,
            }}>
              <div style={{ fontSize: 18, fontWeight: 800, color: 'var(--yellow-dark)' }}>{val}</div>
              <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginTop: 2 }}>{label}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="export-grid">
        <ExportCard
          icon="{ }"
          title="JSON"
          desc="Complete results with metadata, stage details, entity data, and full strategy evaluation."
          label="↓ Download JSON"
          onClick={() => doExport('json')}
        />
        <ExportCard
          icon="CSV"
          title="CSV Scorecard"
          desc="Tabular summary of all chunks: scores, boundaries, entities, text preview."
          label="↓ Download CSV"
          onClick={() => doExport('csv')}
        />
        <ExportCard
          icon="MD"
          title="Markdown"
          desc="Formatted chunk output with metadata comments for downstream review or RAG ingestion."
          label="↓ Download Markdown"
          onClick={() => doExport('markdown')}
        />
      </div>

      <div style={{ marginTop: 20, fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.6 }}>
        Files are generated on the fly from job <code style={{ fontFamily: 'JetBrains Mono', background: 'var(--gray-100)', padding: '1px 5px', borderRadius: 4 }}>{jobId.slice(0, 8)}</code>.
        JSON includes the full strategy evaluation table from S8.
      </div>
    </div>
  )
}
