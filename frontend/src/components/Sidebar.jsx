function SliderRow({ label, id, min, max, step = 1, value, decimals = 0, onChange }) {
  const display = decimals > 0 ? Number(value).toFixed(decimals) : value
  return (
    <div className="cfg-row">
      <div className="cfg-label">
        <span>{label}</span>
        <span className="cfg-val">{display}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(decimals > 0 ? parseFloat(e.target.value) : parseInt(e.target.value))}
      />
    </div>
  )
}

export default function Sidebar({ config, setConfig, apiBase, setApiBase }) {
  const set = (key, val) => setConfig(c => ({ ...c, [key]: val }))

  return (
    <aside className="sidebar">
      <div className="sidebar-inner">
        {/* Logo */}
        <div className="sidebar-logo">
          <div className="logo-mark">AC</div>
          <div>
            <div className="logo-name">AutoChunker</div>
            <div className="logo-sub">Document intelligence platform</div>
          </div>
        </div>

        {/* Chunking */}
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

          <SliderRow label="N_min (tokens)" min={20} max={300} value={config.n_min} onChange={v => set('n_min', v)} />
          <SliderRow label="N_max (tokens)" min={100} max={1000} value={config.n_max} onChange={v => set('n_max', v)} />
        </div>

        {/* Boundary */}
        <div className="cfg-section">
          <div className="cfg-section-title">Boundary Controls</div>

          <SliderRow label="τ JSD low" min={0.05} max={0.40} step={0.01} decimals={2} value={config.tau_jsd_low} onChange={v => set('tau_jsd_low', v)} />
          <SliderRow label="τ JSD high" min={0.20} max={0.80} step={0.01} decimals={2} value={config.tau_jsd_high} onChange={v => set('tau_jsd_high', v)} />

          <div className="cfg-row">
            <div className="cfg-label"><span>Entropy metric</span></div>
            <select value={config.entropy_metric} onChange={e => set('entropy_metric', e.target.value)}>
              <option value="jsd">JSD</option>
              <option value="hellinger">Hellinger</option>
              <option value="hybrid">Hybrid</option>
            </select>
          </div>

          <SliderRow label="Hybrid λ" min={0.10} max={0.90} step={0.01} decimals={2} value={config.hybrid_lambda} onChange={v => set('hybrid_lambda', v)} />

          <div className="cfg-row">
            <div className="cfg-label"><span>Threshold mode</span></div>
            <select value={config.threshold_mode} onChange={e => set('threshold_mode', e.target.value)}>
              <option value="fixed">Fixed</option>
              <option value="percentile">Percentile</option>
            </select>
          </div>

          {config.threshold_mode === 'percentile' && (
            <>
              <SliderRow label="Low percentile" min={5} max={45} value={config.tau_percentile_low} onChange={v => set('tau_percentile_low', v)} />
              <SliderRow label="High percentile" min={55} max={95} value={config.tau_percentile_high} onChange={v => set('tau_percentile_high', v)} />
            </>
          )}

          <SliderRow label="τ_sem similarity" min={0.40} max={0.95} step={0.01} decimals={2} value={config.tau_sem} onChange={v => set('tau_sem', v)} />
        </div>

        {/* Optimization */}
        <div className="cfg-section">
          <div className="cfg-section-title">RL Optimization</div>
          <SliderRow label="Max iterations" min={1} max={20} value={config.max_iterations} onChange={v => set('max_iterations', v)} />
          <SliderRow label="α CodeBLEU weight" min={0.1} max={0.8} step={0.05} decimals={2} value={config.alpha} onChange={v => set('alpha', v)} />
          <SliderRow label="β recall weight" min={0.1} max={0.8} step={0.05} decimals={2} value={config.beta} onChange={v => set('beta', v)} />
          <SliderRow label="λ count penalty" min={0.05} max={0.50} step={0.05} decimals={2} value={config.lambda} onChange={v => set('lambda', v)} />
        </div>

        {/* Embedding */}
        <div className="cfg-section">
          <div className="cfg-section-title">Embedding Model</div>
          <div className="cfg-row">
            <div className="cfg-label"><span>Primary model</span></div>
            <select value={config.embedding_model} onChange={e => set('embedding_model', e.target.value)}>
              <option value="all-MiniLM-L6-v2">all-MiniLM-L6-v2</option>
              <option value="all-mpnet-base-v2">all-mpnet-base-v2</option>
            </select>
          </div>
        </div>

        {/* API */}
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
