/**
 * Sidebar.jsx — AutoChunker configuration panel
 *
 * CHANGES FROM ORIGINAL:
 * 1. Entropy metric dropdown: added "pmi", "depth", "drift" options
 * 2. "Hybrid λ" slider: now only shows when entropy_metric === "hybrid"
 *    (it is meaningless for the other modes)
 * 3. RL section: renamed labels to match what s7_rl.py actually uses
 *
 * Replace your entire Sidebar.jsx with this file.
 */

/**
 * SliderRow — reusable slider component with label + live value display.
 * No changes from the original.
 */
function SliderRow({ label, id, min, max, step = 1, value, decimals = 0, onChange }) {
  // Format the displayed value: show decimals only when requested
  const display = decimals > 0 ? Number(value).toFixed(decimals) : value

  return (
    <div className="cfg-row">
      <div className="cfg-label">
        <span>{label}</span>
        {/* Show the current numeric value to the right of the label */}
        <span className="cfg-val">{display}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e =>
          // Parse as float when decimals are needed, otherwise as integer
          onChange(decimals > 0 ? parseFloat(e.target.value) : parseInt(e.target.value))
        }
      />
    </div>
  )
}

/**
 * Sidebar — main configuration panel component.
 *
 * Props:
 *   config    — the current config object (controlled by App.jsx state)
 *   setConfig — state setter for config (passed down from App.jsx)
 *   apiBase   — backend URL string
 *   setApiBase — setter for backend URL
 */
export default function Sidebar({ config, setConfig, apiBase, setApiBase }) {
  // Helper: update a single key in the config object without touching other keys
  const set = (key, val) => setConfig(c => ({ ...c, [key]: val }))

  return (
    <aside className="sidebar">
      <div className="sidebar-inner">

        {/* ── Logo ───────────────────────────────────────────────────────── */}
        <div className="sidebar-logo">
          <div className="logo-mark">AC</div>
          <div>
            <div className="logo-name">AutoChunker</div>
            <div className="logo-sub">Document intelligence platform</div>
          </div>
        </div>

        {/* ── Chunking section ────────────────────────────────────────────── */}
        <div className="cfg-section">
          <div className="cfg-section-title">Chunking</div>

          {/* Strategy selector: controls which S2 chunker wins the selection */}
          <div className="cfg-row">
            <div className="cfg-label"><span>Strategy</span></div>
            <select
              value={config.chunking_strategy}
              onChange={e => set('chunking_strategy', e.target.value)}
            >
              {/* "auto" lets s2_chunkers.py pick the best strategy by quality score */}
              <option value="auto">Auto (best quality)</option>
              <option value="structure">Structure-based</option>
              <option value="semantic_boundaries">Semantic boundaries</option>
              <option value="sentence_clustering">Sentence clustering</option>
              <option value="paragraph_pack">Paragraph pack</option>
              {/* legal_articles uses LEGAL_BOUNDARY_RE — best for treaty/regulation docs */}
              <option value="legal_articles">Legal articles</option>
              <option value="recursive">Recursive character</option>
              <option value="sliding_window">Sliding window</option>
            </select>
          </div>

          {/* N_min: minimum chunk size in tokens — chunks smaller than this get merged */}
          <SliderRow
            label="N_min (tokens)"
            min={20} max={400}
            value={config.n_min}
            onChange={v => set('n_min', v)}
          />

          {/* N_max: maximum chunk size in tokens — chunks larger than this get split */}
          <SliderRow
            label="N_max (tokens)"
            min={100} max={1200}
            value={config.n_max}
            onChange={v => set('n_max', v)}
          />
        </div>

        {/* ── Boundary Controls section ───────────────────────────────────── */}
        <div className="cfg-section">
          <div className="cfg-section-title">Boundary Controls</div>

          {/*
            τ JSD low — the MERGE threshold.
            Boundaries with combined signal BELOW this value are merged into the
            previous chunk (the two chunks are too similar to be separated).
            Lower = more merging.
          */}
          <SliderRow
            label="τ JSD low"
            min={0.05} max={0.40} step={0.01} decimals={2}
            value={config.tau_jsd_low}
            onChange={v => set('tau_jsd_low', v)}
          />

          {/*
            τ JSD high — the HARD SPLIT threshold.
            Boundaries with combined signal ABOVE this value become hard splits
            (even if not a protected Article/Section boundary).
            Higher = fewer hard splits.
          */}
          <SliderRow
            label="τ JSD high"
            min={0.20} max={0.80} step={0.01} decimals={2}
            value={config.tau_jsd_high}
            onChange={v => set('tau_jsd_high', v)}
          />

          {/*
            Entropy metric selector.
            Controls which signal(s) are used to compute the raw boundary score
            BEFORE the LSTM combines them.  The LSTM always receives all 5 signals
            regardless of this setting — this only affects the raw weighted sum.

            Options:
              hybrid  — weighted combination of all 5 (recommended for financial docs)
              jsd     — Jensen-Shannon Divergence only (classic, good baseline)
              hellinger — Hellinger distance (more sensitive to rare-term changes)
              pmi     — PMI-drop: key concept shift (best for legal/regulatory text)
              depth   — Structure depth change (Article/Chapter/Section transitions)
              drift   — Embedding drift from local baseline (detects slow topic migration)
          */}
          <div className="cfg-row">
            <div className="cfg-label"><span>Entropy metric</span></div>
            <select
              value={config.entropy_metric}
              onChange={e => set('entropy_metric', e.target.value)}
            >
              {/* ORIGINAL OPTIONS (kept as-is) */}
              <option value="jsd">JSD only</option>
              <option value="hellinger">Hellinger only</option>
              <option value="hybrid">Hybrid (all 5 signals)</option>

              {/* NEW OPTIONS — added to match the 5-signal s3_entropy.py */}
              <option value="pmi">PMI-drop (concept shift)</option>
              <option value="depth">Structure depth</option>
              <option value="drift">Embedding drift</option>
            </select>
          </div>

          {/*
            Hybrid λ — only shown when metric is "hybrid".
            Controls the JSD vs Hellinger blend inside the hybrid formula:
              hybrid_signal = λ * hellinger + (1-λ) * jsd
            Note: in s3_entropy.py the full hybrid formula uses 5 signals with
            fixed weights; this λ is passed as config.hybrid_lambda and used
            by the "_select_metric" logic if it remains from a previous version.
            We keep the slider for backward compatibility.
          */}
          {config.entropy_metric === 'hybrid' && (
            <SliderRow
              label="Hybrid λ"
              min={0.10} max={0.90} step={0.01} decimals={2}
              value={config.hybrid_lambda}
              onChange={v => set('hybrid_lambda', v)}
            />
          )}

          {/*
            Threshold mode:
              fixed      — use τ_low and τ_high exactly as set by the sliders
              percentile — compute τ_low and τ_high from the document's own
                           signal distribution (25th and 75th percentile by default).
                           This adapts automatically to each document.
          */}
          <div className="cfg-row">
            <div className="cfg-label"><span>Threshold mode</span></div>
            <select
              value={config.threshold_mode}
              onChange={e => set('threshold_mode', e.target.value)}
            >
              <option value="fixed">Fixed</option>
              {/* Percentile is recommended for financial/regulatory docs */}
              <option value="percentile">Percentile (adaptive)</option>
            </select>
          </div>

          {/*
            Percentile sliders — only shown when mode is "percentile".
            Low percentile  → τ_low  (merge threshold, e.g. 25th pct)
            High percentile → τ_high (hard split threshold, e.g. 75th pct)
          */}
          {config.threshold_mode === 'percentile' && (
            <>
              <SliderRow
                label="Low percentile"
                min={5} max={45}
                value={config.tau_percentile_low}
                onChange={v => set('tau_percentile_low', v)}
              />
              <SliderRow
                label="High percentile"
                min={55} max={95}
                value={config.tau_percentile_high}
                onChange={v => set('tau_percentile_high', v)}
              />
            </>
          )}

          {/*
            τ_sem similarity — used in S4 (Boundary Quality Filter).
            Two adjacent chunks with semantic similarity ABOVE this threshold
            get merged regardless of their entropy score.
            Higher = less merging in S4.
          */}
          <SliderRow
            label="τ_sem similarity"
            min={0.40} max={0.95} step={0.01} decimals={2}
            value={config.tau_sem}
            onChange={v => set('tau_sem', v)}
          />
        </div>

        {/* ── RL Optimization section ─────────────────────────────────────── */}
        <div className="cfg-section">
          {/* Label updated: this is DQN-based RL, not simple alpha/beta tuning */}
          <div className="cfg-section-title">RL Optimization (DQN)</div>

          {/* Max iterations: how many S2→S6 re-runs the DQN agent performs */}
          <SliderRow
            label="Max iterations"
            min={1} max={20}
            value={config.max_iterations}
            onChange={v => set('max_iterations', v)}
          />

          {/*
            α — quality weight in the multi-objective reward.
            Reward = α·quality + β·coverage + consistency_weight·consistency + efficiency_weight·efficiency
            quality = 1 - mean(boundary_score) across all chunks
          */}
          <SliderRow
            label="α quality weight"
            min={0.1} max={0.8} step={0.05} decimals={2}
            value={config.alpha}
            onChange={v => set('alpha', v)}
          />

          {/*
            β — coverage (recall proxy) weight in the reward.
            coverage = fraction of probe queries answered by at least one chunk
          */}
          <SliderRow
            label="β coverage weight"
            min={0.1} max={0.8} step={0.05} decimals={2}
            value={config.beta}
            onChange={v => set('beta', v)}
          />

          {/*
            λ — count penalty weight in the reward.
            Penalizes producing too many or too few chunks relative to target.
          */}
          <SliderRow
            label="λ count penalty"
            min={0.05} max={0.50} step={0.05} decimals={2}
            value={config.lambda}
            onChange={v => set('lambda', v)}
          />
        </div>

        {/* ── Embedding Model section ─────────────────────────────────────── */}
        <div className="cfg-section">
          <div className="cfg-section-title">Embedding Model</div>
          <div className="cfg-row">
            <div className="cfg-label"><span>Primary model</span></div>
            <select
              value={config.embedding_model}
              onChange={e => set('embedding_model', e.target.value)}
            >
              {/* all-MiniLM-L6-v2: fast, 384-dim, good for most docs */}
              <option value="all-MiniLM-L6-v2">all-MiniLM-L6-v2</option>
              {/* all-mpnet-base-v2: slower, higher quality, 768-dim */}
              <option value="all-mpnet-base-v2">all-mpnet-base-v2</option>
            </select>
          </div>
        </div>

        {/* ── Service Endpoint section ────────────────────────────────────── */}
        <div className="cfg-section">
          <div className="cfg-section-title">Service Endpoint</div>
          <div className="cfg-row">
            <div className="cfg-label"><span>Backend URL</span></div>
            <input
              type="text"
              className="sidebar-text-input"
              value={apiBase}
              onChange={e => setApiBase(e.target.value.replace(/\/$/, ''))}
              placeholder="http://localhost:8000"
            />
          </div>
        </div>

      </div>
    </aside>
  )
}
