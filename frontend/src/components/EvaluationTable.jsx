// ─── EvaluationTable.jsx ──────────────────────────────────────────────────
// Shows strategy comparison with TABLE I retrieval metrics for S8,
// and internal quality metrics for S7 (passed via metricCols prop).

// S8 default: paper TABLE I retrieval metrics
const DEFAULT_METRIC_COLS = [
  { key: 'mrr',       label: 'MRR' },
  { key: 'ndcg',      label: 'NDCG@5' },
  { key: 'precision', label: 'Precision' },
  { key: 'recall',    label: 'Recall' },
  { key: 'ss2fd',     label: 'ss2fd' },
  { key: 'qcs',       label: 'QCS' },
]

// Clamp any score/metric to [0, 1] so it can never render above 100%.
const pct01 = v => Math.min(Math.max(Number(v) || 0, 0), 1)

export default function EvaluationTable({
  table,
  winner,
  title = 'Strategy Evaluation',
  note,
  metricCols = DEFAULT_METRIC_COLS,
}) {
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

            {/* Strategy name + chunk info */}
            <div>
              <div className="eval-strategy-name">
                {row.strategy.replace(/_/g, ' ')}
              </div>
              <div className="eval-strategy-sub">
                {row.chunk_count} chunks · ~{row.mean_tokens} tokens
              </div>
            </div>

            {/* Score — show internal + rag breakdown if available */}
            <div className="eval-score" title={
              row.internal_score != null && row.rag_score != null
                ? `Internal: ${(pct01(row.internal_score) * 100).toFixed(1)}%  |  RAG: ${(pct01(row.rag_score) * 100).toFixed(1)}%`
                : undefined
            }>
              {(pct01(row.score) * 100).toFixed(1)}%
              {row.internal_score != null && row.rag_score != null && (
                <div style={{ fontSize: 9, color: 'var(--text-muted)', fontWeight: 400, marginTop: 2 }}>
                  int {(pct01(row.internal_score) * 100).toFixed(0)}% · rag {(pct01(row.rag_score) * 100).toFixed(0)}%
                </div>
              )}
            </div>

            {/* Mini metric bars */}
            <div className="eval-metrics-mini">
              {metricCols.map(({ key, label }) => {
                const val = pct01(row[key])
                return (
                  <div className="eval-mini-bar-row" key={key}>
                    <div className="eval-mini-label">{label}</div>
                    <div className="eval-mini-track">
                      <div
                        className="eval-mini-fill"
                        style={{ width: (val * 100) + '%' }}
                      />
                    </div>
                    <div className="eval-mini-val">{Math.round(val * 100)}%</div>
                  </div>
                )
              })}

              {/* Token cost + chunking time (TABLE I — shown when available) */}
              {(row.retrieval_token_cost > 0 || row.chunking_time > 0) && (
                <div style={{
                  fontSize: 9.5,
                  color: 'var(--text-muted)',
                  marginTop: 3,
                  fontFamily: "'JetBrains Mono', monospace",
                  whiteSpace: 'nowrap',
                }}>
                  {row.retrieval_token_cost > 0 && `~${Math.round(row.retrieval_token_cost)} tok/q`}
                  {row.retrieval_token_cost > 0 && row.chunking_time > 0 && ' · '}
                  {row.chunking_time > 0 && `${row.chunking_time.toFixed(2)}s`}
                </div>
              )}
            </div>

            {/* Chunk count */}
            <div className="eval-chunks-info">
              <div className="eval-chunks-val">{row.chunk_count}</div>
              <div className="eval-chunks-sub">chunks</div>
            </div>

          </div>
        ))}
      </div>

      {note && (
        <div style={{
          marginTop: 12,
          fontSize: 11,
          color: 'var(--text-muted)',
          lineHeight: 1.6,
        }}>
          {note}
        </div>
      )}
    </div>
  )
}
