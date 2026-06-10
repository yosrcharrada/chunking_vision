import { useState } from 'react'

// ── Metric column sets ─────────────────────────────────────────────────────

// These 5 metrics compose BOTH the S8 pre-RL score and the S7 RL reward
const STRUCTURAL_COLS = [
  { key: 'mean_icc',                 label: 'Coherence',  tip: 'Intra-chunk coherence (ICC). For regulatory prose ~18-21% is normal.' },
  { key: 'inter_separation',         label: 'Separation', tip: 'Inter-chunk topic separation — how distinct adjacent chunks are.' },
  { key: 'entropy_strength',         label: 'Entropy',    tip: 'Boundary entropy signal strength.' },
  { key: 'boundary_distinctiveness', label: 'Boundary',   tip: 'Boundary distinctiveness (JSD + Hellinger + drift).' },
  { key: 'size_fit',                 label: 'Size Fit',   tip: '1.0 = all chunks in [n_min, n_max]. 0.0 = all out of range.' },
]

const RETRIEVAL_COLS = [
  { key: 'mrr',       label: 'MRR',       tip: 'Mean Reciprocal Rank (BM25 leave-one-out).' },
  { key: 'ndcg',      label: 'NDCG@5',    tip: 'Normalised Discounted Cumulative Gain @ 5.' },
  { key: 'precision', label: 'Precision', tip: 'Precision@5 (BM25 leave-one-out).' },
  { key: 'recall',    label: 'Recall',    tip: 'Recall@5 — fraction of queries answered in top-5.' },
]

const REWARD_BREAKDOWN_COLS = [
  { key: 'reward_quality',     label: 'Quality'     },
  { key: 'reward_coverage',    label: 'Coverage'    },
  { key: 'reward_consistency', label: 'Consistency' },
  { key: 'reward_efficiency',  label: 'Efficiency'  },
]

// ── Mini bar ──────────────────────────────────────────────────────────────────

function MiniBar({ label, value, tip, isWinner }) {
  const pct = Math.round((Number(value) || 0) * 100)
  return (
    <div className="eval-mini-bar-row" title={tip || label}>
      <div className="eval-mini-label">{label}</div>
      <div className="eval-mini-track">
        <div className="eval-mini-fill" style={{ width: pct + '%' }} />
      </div>
      <div className="eval-mini-val">{pct}%</div>
    </div>
  )
}

// ── Single evaluation row ─────────────────────────────────────────────────────

function EvalRow({ row, isS7 }) {
  const isWinner = row.winner

  // Retrieval metrics are real only when explicitly computed (method set + mrr > 0)
  const retrievalComputed = row.retrieval_method
    && row.retrieval_method !== 'rank_bm25_not_installed_and_no_embeddings'
    && row.retrieval_method !== 'too_few_chunks'
    && row.retrieval_method !== 'no_valid_queries'
    && (row.mrr > 0 || row.ndcg > 0 || row.precision > 0)

  // S7-only: reward breakdown populated
  const hasBreakdown = isS7
    && (row.reward_quality > 0 || row.reward_coverage > 0
        || row.reward_consistency > 0 || row.reward_efficiency > 0)

  // Sentinel detection for sliding_window (flat reward, fallback values)
  const hasSentinel = isS7
    && row.entropy_strength === 0
    && Math.abs((row.mean_icc || 0) - 0.5) < 0.01

  const scoreTitle = isS7 ? 'RL Reward (post-tuning)' : 'Pre-RL structural composite'

  return (
    <div className={`eval-row${isWinner ? ' winner' : ''}`}>

      {/* Rank */}
      <div className="eval-rank">
        {isWinner ? '🏆' : `#${row.rank}`}
      </div>

      {/* Name + sub-info */}
      <div>
        <div className="eval-strategy-name">
          {row.strategy.replace(/_/g, ' ')}
        </div>
        <div className="eval-strategy-sub">
          {row.chunk_count} chunks · ~{row.mean_tokens} tokens
        </div>
        {isS7 && isWinner && (
          <div style={{ fontSize: 9, color: 'var(--green)', marginTop: 2 }}>
            ← Final output saved to chunks[ ]
          </div>
        )}
        {isS7 && row.size_fit === 0 && (
          <div style={{ fontSize: 9, color: 'var(--red)', marginTop: 2 }}>
            ⚠ size_fit = 0 — all chunks exceed n_max
          </div>
        )}
      </div>

      {/* Score */}
      <div className="eval-score" title={scoreTitle}>
        {((row.score || 0) * 100).toFixed(1)}%
      </div>

      {/* Metric bars */}
      <div className="eval-metrics-mini">

        {hasSentinel ? (
          <div style={{ fontSize: 9, color: 'var(--amber)', fontStyle: 'italic', lineHeight: 1.5 }}>
            ⚠ RL produced no improvement — reward was flat across all iterations.
          </div>
        ) : (
          STRUCTURAL_COLS.map(({ key, label, tip }) => (
            <MiniBar key={key} label={label} value={row[key]} tip={tip} isWinner={isWinner} />
          ))
        )}

        {/* ss2fd and QCS — shown when non-zero */}
        {(row.ss2fd > 0) && (
          <MiniBar label="ss2fd" value={row.ss2fd} tip="Semantic similarity to full document (no query set needed)." isWinner={isWinner} />
        )}
        {(row.qcs > 0) && (
          <div className="eval-mini-bar-row" style={{ opacity: 0.75 }}>
            <div className="eval-mini-label" style={{ fontSize: 9 }}>QCS</div>
            <div className="eval-mini-track">
              <div className="eval-mini-fill" style={{ width: ((row.qcs || 0) * 100) + '%' }} />
            </div>
            <div className="eval-mini-val" style={{ fontSize: 9 }}>
              {((row.qcs || 0) * 100).toFixed(1)}%
            </div>
          </div>
        )}

        {/* Retrieval metrics: only when actually computed */}
        {retrievalComputed
          ? RETRIEVAL_COLS.map(({ key, label, tip }) => (
              <MiniBar key={key} label={label} value={row[key]} tip={tip} isWinner={isWinner} />
            ))
          : (
            <div style={{ marginTop: 4, fontSize: 9, color: 'var(--text-muted)', fontStyle: 'italic' }}>
              {_BM25_NOT_INSTALLED_NOTE}
            </div>
          )
        }

        {/* Reward breakdown (S7 only, when populated) */}
        {hasBreakdown && (
          <>
            <div style={{
              marginTop: 5, fontSize: 9, color: 'var(--text-muted)',
              fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em',
            }}>
              Reward breakdown
            </div>
            {REWARD_BREAKDOWN_COLS.map(({ key, label }) => (
              <MiniBar key={key} label={label} value={row[key]} isWinner={isWinner} />
            ))}
          </>
        )}
        {isS7 && !hasBreakdown && (
          <div style={{ marginTop: 4, fontSize: 9, color: 'var(--text-muted)', fontStyle: 'italic' }}>
            Reward sub-components not yet computed.
          </div>
        )}
      </div>

      {/* Chunk count badge */}
      <div className="eval-chunks-info">
        <div className="eval-chunks-val">{row.chunk_count}</div>
        <div className="eval-chunks-sub">chunks</div>
      </div>

    </div>
  )
}

const _BM25_NOT_INSTALLED_NOTE = 'Retrieval metrics (MRR/NDCG/Precision/Recall) require: pip install rank_bm25'

// ── Main export ───────────────────────────────────────────────────────────────

export default function EvaluationTable({
  table,
  winner,
  title = 'Strategy Evaluation',
  note,
  isS7 = false,
  metricCols,   // ignored — bar set determined by isS7 flag
}) {
  if (!table || table.length === 0) return null

  const scoreNote = isS7
    ? 'Score = RL final reward after GA parameter tuning. Bars show the 5 structural metrics the reward function optimises. The winner\'s chunks are the final pipeline output.'
    : 'Score = pre-RL structural composite (Coherence + Boundary + Size Fit + Separation + Entropy). Retrieval metrics (MRR/NDCG/Precision) appear only when rank_bm25 is installed.'

  return (
    <div className="card">
      <div className="card-title">
        {title}
        {winner && (
          <span className="badge">
            Winner: {winner.replace(/_/g, ' ')}
          </span>
        )}
        <span style={{
          fontSize: 10,
          fontWeight: 400,
          color: 'var(--text-muted)',
          marginLeft: 8,
          fontStyle: 'italic',
        }}>
          {isS7 ? 'RL Reward (post-tuning)' : 'Pre-RL Score'}
        </span>
      </div>

      <div className="eval-table">
        {table.map(row => (
          <EvalRow key={row.strategy} row={row} isS7={isS7} />
        ))}
      </div>

      <div style={{ marginTop: 12, fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.6 }}>
        {note || scoreNote}
      </div>
    </div>
  )
}
