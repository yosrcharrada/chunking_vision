/**
 * QEntropyMetrics.jsx — the authoritative evaluation panel.
 *
 * Renders `results.p2_evaluation`, produced by backend pipeline/evaluation.py
 * (engine.metrics, Table I).  It shows, for every GA-tuned strategy:
 *   • the qentropy fingerprint (q, D_q, Shannon/Tsallis bits, entropy rate),
 *   • the Table-I retrieval metrics (Precision/Recall/MRR/NDCG/ss2fd/SRGT/QCS/
 *     token cost/time), and
 *   • the overall winner + how it was decided (primary signal vs token-cost tie).
 */

function fmt(v, d = 3) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '—'
  return Number(v).toFixed(d)
}

export default function QEntropyMetrics({ p2 }) {
  if (!p2 || !p2.metrics || Object.keys(p2.metrics).length === 0) return null

  const info = p2.metric_info || []
  const winner = p2.summary?.overall?.winner_id || p2.winner
  // Sort rows by the cosine-rank mean, with the decided winner pinned to the top
  // so the ★ row is always first (it may win on the token-cost tie-break).
  const rankKeys = ['mrr', 'ndcg', 'srgt', 'qcs']
  const rankScore = m => rankKeys.reduce((s, k) => s + (Number(m?.[k]) || 0), 0) / rankKeys.length
  const strategies = Object.keys(p2.metrics).sort((a, b) => {
    if (a === winner) return -1
    if (b === winner) return 1
    return rankScore(p2.metrics[b]) - rankScore(p2.metrics[a])
  })
  const overall = p2.summary?.overall
  const bestPer = p2.summary?.best_per_metric || {}
  const qstats = p2.qentropy || {}
  // When any strategy ran the q-log entropy family, the S_q column shows the
  // q-logarithm entropy value (nats) instead of the Tsallis bits, and the
  // header is relabelled so it is unambiguous which entropy is on screen.
  const anyQlog = strategies.some(n => (qstats[n] || {}).entropy_mode === 'qlog')

  return (
    <div className="card" style={{ marginBottom: 14 }}>
      <div className="card-title">
        qEntropy · Table-I Evaluation
        {winner && <span className="badge">Winner: {String(winner).replace(/_/g, ' ')}</span>}
      </div>

      <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10 }}>
        {p2.n_queries > 0
          ? <>Scored on <strong>{p2.n_queries}</strong> auto-generated questions ·
              ranked by <strong>{overall?.ranked_by === 'answer_correct' ? 'LLM answer-correctness' : 'cosine-rank mean'}</strong>
              {overall?.decided_by === 'token_cost_among_ties' && ' · tie broken by token cost'} ·
              rel-threshold {fmt(p2.rel_threshold, 2)}</>
          : <>No evaluation questions (offline) — showing query-free metrics only (ss2fd, chunk stats).
              Set <code>OPENAI_API_KEY</code> for the full Table-I metrics.</>}
        {overall?.note && <div style={{ marginTop: 4, fontStyle: 'italic' }}>{overall.note}</div>}
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table className="qe-table">
          <thead>
            <tr>
              <th style={{ textAlign: 'left' }}>Strategy</th>
              <th>q</th>
              <th>D_q</th>
              <th>H (bits)</th>
              <th title={anyQlog
                ? 'q-logarithm entropy S_q^K (nats) — the q-deformed-log entropy driving the boundary count'
                : 'Tsallis q-entropy S_q (bits)'}>
                {anyQlog ? 'S_q^K (q-log)' : 'S_q (bits)'}
              </th>
              <th>h_K</th>
              {info.map(m => <th key={m.key} title={m.desc}>{m.label}</th>)}
            </tr>
          </thead>
          <tbody>
            {strategies.map(name => {
              const m = p2.metrics[name] || {}
              const q = qstats[name] || {}
              const isWinner = name === winner
              return (
                <tr key={name} className={isWinner ? 'qe-winner-row' : ''}>
                  <td style={{ textAlign: 'left', fontWeight: isWinner ? 700 : 500 }}>
                    {isWinner && '★ '}{name.replace(/_/g, ' ')}
                  </td>
                  <td>{fmt(q.q, 2)}</td>
                  <td>{fmt(q.diversity_number, 2)}</td>
                  <td>{fmt(q.shannon_bits, 2)}</td>
                  <td>{fmt(q.entropy_mode === 'qlog' ? q.qlog_bits : q.tsallis_bits, 2)}</td>
                  <td>{fmt(q.entropy_rate, 3)}</td>
                  {info.map(col => {
                    const v = m[col.key]
                    const best = bestPer[col.key] === name
                    const decimals = col.key === 'retrieval_token_cost' || col.key === 'chunking_time_ms' ? 1 : 3
                    return (
                      <td key={col.key} style={best ? { fontWeight: 700, color: 'var(--yellow-dark)' } : undefined}>
                        {fmt(v, decimals)}
                      </td>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
