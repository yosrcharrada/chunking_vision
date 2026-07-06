/**
 * Sidebar.jsx — qentropy chunker configuration panel
 *
 * Post-merge: S3 is driven solely by the qentropy engine (Tsallis q-entropy +
 * Hill diversity number D_q), so the old JSD/Hellinger/PMI threshold controls,
 * the PPL toggle, the bespoke RL reward weights and the multi-model ensemble are
 * gone.  What remains are the knobs that actually move the qentropy pipeline:
 *   • S2 sizing (strategy, N_min, N_max)
 *   • S3 qentropy (Tsallis q, branching K, min/max chunk tokens, shift window)
 *   • S4 merge threshold (τ_sem)
 *   • Embedding backend (engine.embeddings) + optional LLM answerability judge
 *   • Auto-QA count (engine.qagen) for the Table-I metrics
 *   • S7 GA budget
 */
import { useEffect, useState } from 'react'

function SliderRow({ label, min, max, step = 1, value, decimals = 0, onChange }) {
  const display = decimals > 0 ? Number(value).toFixed(decimals) : value
  return (
    <div className="cfg-row">
      <div className="cfg-label">
        <span>{label}</span>
        <span className="cfg-val">{display}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(decimals > 0 ? parseFloat(e.target.value) : parseInt(e.target.value))}
      />
    </div>
  )
}

export default function Sidebar({ config, setConfig, apiBase, setApiBase }) {
  const set = (key, val) => setConfig(c => ({ ...c, [key]: val }))
  const [backends, setBackends] = useState([])
  const [caps, setCaps] = useState({ openai_configured: false })

  // Discover the embedding backends the engine actually has available.
  useEffect(() => {
    let alive = true
    fetch(`${apiBase}/backends`)
      .then(r => r.json())
      .then(d => { if (alive) { setBackends(d.backends || []); setCaps(d) } })
      .catch(() => { if (alive) setBackends([]) })
    return () => { alive = false }
  }, [apiBase])

  return (
    <aside className="sidebar">
      <div className="sidebar-inner">

        <div className="sidebar-logo">
          <div className="logo-mark">qH</div>
          <div>
            <div className="logo-name">qEntropy Chunker</div>
            <div className="logo-sub">Tsallis · diversity-number chunking</div>
          </div>
        </div>

        {/* ── Chunking (S2 sizing) ───────────────────────────────────────── */}
        <div className="cfg-section">
          <div className="cfg-section-title">Chunking</div>
          <div className="cfg-row">
            <div className="cfg-label"><span>Strategy</span></div>
            <select value={config.chunking_strategy} onChange={e => set('chunking_strategy', e.target.value)}>
              <option value="auto">Auto (best quality)</option>
              <option value="structure">Structure-based</option>
              <option value="semantic_boundaries">Semantic boundaries</option>
              <option value="sentence_clustering">Sentence clustering</option>
              <option value="paragraph_pack">Paragraph pack</option>
              <option value="legal_articles">Legal articles</option>
              <option value="recursive">Recursive character</option>
              <option value="sliding_window">Sliding window</option>
            </select>
          </div>
          <SliderRow label="N_min (words)" min={20} max={400} value={config.n_min} onChange={v => set('n_min', v)} />
          <SliderRow label="N_max (words)" min={100} max={1200} value={config.n_max} onChange={v => set('n_max', v)} />
        </div>

        {/* ── qEntropy (S3) ──────────────────────────────────────────────── */}
        <div className="cfg-section">
          <div className="cfg-section-title">qEntropy (S3)</div>

          {/* Tsallis q ∈ [-1, 1]. q<1 → finer (keeps weak shifts), q>1 → coarser,
              q=1 → Shannon perplexity exp(H). D_q = round(diversity number) sets
              the boundary count. Auto-tuned by the S7 GA. */}
          <SliderRow
            label="Tsallis q"
            min={-1.0} max={1.0} step={0.05} decimals={2}
            value={config.q_entropy_param ?? 1.0}
            onChange={v => set('q_entropy_param', v)}
          />
          <SliderRow label="Branching K" min={2} max={8} value={config.K ?? 4} onChange={v => set('K', v)} />
          <SliderRow label="Min chunk tokens" min={4} max={200} value={config.min_chunk_tokens ?? 20}
                     onChange={v => set('min_chunk_tokens', v)} />
          <SliderRow label="Max chunk tokens" min={64} max={2000} step={16} value={config.max_chunk_tokens ?? 320}
                     onChange={v => set('max_chunk_tokens', v)} />
          <SliderRow label="Shift window" min={0} max={5} value={config.window ?? 1} onChange={v => set('window', v)} />

          {/* S4 semantic-merge threshold */}
          <SliderRow label="τ_sem (S4 merge)" min={0.40} max={0.95} step={0.01} decimals={2}
                     value={config.tau_sem} onChange={v => set('tau_sem', v)} />

          {/* S4 similarity kernel: classical cosine vs Fitouhi–Bouzeffour q-cosine (base=q²).
              q-cosine deforms the inter-chunk angle similarity with the same
              Tsallis q used by S3; q=±1 recovers the classical angular cosine. */}
          <div className="cfg-row">
            <div className="cfg-label"><span>S4 similarity</span></div>
            <select
              value={config.s4_similarity ?? 'cosine'}
              onChange={e => set('s4_similarity', e.target.value)}
            >
              <option value="cosine">Cosine (classical)</option>
              <option value="qcosine">q-Cosine (Fitouhi–Bouzeffour)</option>
            </select>
          </div>
        </div>

        {/* ── Evaluation (Table I) ───────────────────────────────────────── */}
        <div className="cfg-section">
          <div className="cfg-section-title">Evaluation</div>
          <SliderRow label="Auto-QA questions" min={2} max={40} value={config.qa_count ?? 12}
                     onChange={v => set('qa_count', v)} />
          <div className="cfg-row">
            <div className="cfg-label"><span>LLM answerability judge</span></div>
            <input
              type="checkbox"
              checked={!!config.judge_answerability}
              disabled={!caps.openai_configured}
              onChange={e => set('judge_answerability', e.target.checked)}
            />
          </div>
          {!caps.openai_configured && (
            <div className="cfg-hint">No OPENAI_API_KEY — QA + judge disabled; cosine-rank winner is used.</div>
          )}
          {caps.openai_configured && (
            <div className="cfg-hint">
              LLM provider: {caps.provider === 'azure' ? 'Azure / EY endpoint' : 'OpenAI'} ✓
            </div>
          )}
        </div>

        {/* ── Embedding backend (engine.embeddings) ──────────────────────── */}
        <div className="cfg-section">
          <div className="cfg-section-title">Embedding Backend</div>
          <div className="cfg-row">
            <div className="cfg-label"><span>Backend</span></div>
            <select
              value={config.embedding_backend ?? ''}
              onChange={e => set('embedding_backend', e.target.value || null)}
            >
              <option value="">Auto (default)</option>
              {backends.map(b => (
                <option key={b.id} value={b.id} disabled={!b.available}>
                  {b.label}{b.available ? '' : ' — unavailable'}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* ── S7 Genetic Algorithm ───────────────────────────────────────── */}
        <div className="cfg-section">
          <div className="cfg-section-title">S7 Genetic Algorithm</div>
          <SliderRow label="Population" min={2} max={24} value={config.ga_population ?? 6}
                     onChange={v => set('ga_population', v)} />
          <SliderRow label="Generations" min={1} max={12} value={config.ga_generations ?? 3}
                     onChange={v => set('ga_generations', v)} />
        </div>

        {/* ── Service endpoint ───────────────────────────────────────────── */}
        <div className="cfg-section">
          <div className="cfg-section-title">Service Endpoint</div>
          <div className="cfg-row">
            <div className="cfg-label"><span>Backend URL</span></div>
            <input
              type="text" className="sidebar-text-input" value={apiBase}
              onChange={e => setApiBase(e.target.value.replace(/\/$/, ''))}
              placeholder="http://localhost:8000"
            />
          </div>
        </div>

      </div>
    </aside>
  )
}
