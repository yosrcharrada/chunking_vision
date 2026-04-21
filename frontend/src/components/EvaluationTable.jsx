const METRIC_COLS = [
  { key: 'mean_icc',               label: 'Coherence' },
  { key: 'inter_separation',       label: 'Separation' },
  { key: 'entropy_strength',       label: 'Entropy' },
  { key: 'boundary_distinctiveness', label: 'Boundary' },
  { key: 'size_fit',               label: 'Size Fit' },
]

export default function EvaluationTable({ table, winner, title = 'Strategy Evaluation', note }) {
  if (!table || table.length === 0) return null

  return (
    <div className="card">
      <div className="card-title">
        {title}
        {winner && (
          <span className="badge">
            Winner: {winner.replace(/_/g, ' ')}
          </span>
        )}
      </div>

      <div className="eval-table">
        {table.map(row => (
          <div key={row.strategy} className={`eval-row${row.winner ? ' winner' : ''}`}>
            {/* Rank */}
            <div className="eval-rank">
              {row.winner ? '🏆' : `#${row.rank}`}
            </div>

            {/* Strategy name */}
            <div>
              <div className="eval-strategy-name">
                {row.strategy.replace(/_/g, ' ')}
              </div>
              <div className="eval-strategy-sub">
                {row.chunk_count} chunks · ~{row.mean_tokens} tokens
              </div>
            </div>

            {/* Score */}
            <div className="eval-score">
              {(row.score * 100).toFixed(1)}%
            </div>

            {/* Mini metric bars */}
            <div className="eval-metrics-mini">
              {METRIC_COLS.map(({ key, label }) => {
                const val = row[key] || 0
                return (
                  <div className="eval-mini-bar-row" key={key}>
                    <div className="eval-mini-label">{label}</div>
                    <div className="eval-mini-track">
                      <div className="eval-mini-fill" style={{ width: (val * 100) + '%' }} />
                    </div>
                    <div className="eval-mini-val">{Math.round(val * 100)}%</div>
                  </div>
                )
              })}
            </div>

            {/* Counts */}
            <div className="eval-chunks-info">
              <div className="eval-chunks-val">{row.chunk_count}</div>
              <div className="eval-chunks-sub">chunks</div>
            </div>
          </div>
        ))}
      </div>

      {note && (
        <div style={{ marginTop: 12, fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.6 }}>
          {note}
        </div>
      )}
    </div>
  )
}
