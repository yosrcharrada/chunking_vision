import { useState, useRef, useEffect } from 'react'
import EvaluationTable from './EvaluationTable'

const STAGES = ['S1', 'S2', 'S3-S6', 'S8', 'S7', 'DONE']
const STAGE_LABELS = {
  S1: 'S1 Profile', S2: 'S2 Chunk', 'S3-S6': 'S3–S6 Pipeline', S8: 'S8 Evaluate', S7: 'S7 RL', DONE: 'Done',
}
const DOC_METRICS = [
  { key: 'RC', label: 'Retrieval Cues' },
  { key: 'ICC', label: 'Local Cohesion' },
  { key: 'DCC', label: 'Block Coherence' },
  { key: 'BI', label: 'Block Integrity' },
  { key: 'SC', label: 'Size Regularity' },
]
const STAGE2_METRICS = [
  { key: 'size_fit', label: 'Size Fit' },
  { key: 'count_fit', label: 'Count Fit' },
  { key: 'boundary_quality', label: 'Boundary' },
  { key: 'structure_integrity', label: 'Structure' },
  { key: 'cohesion', label: 'Cohesion' },
  { key: 'distinctness', label: 'Distinct' },
  { key: 'time_efficiency', label: 'Speed' },
]

function normalizeStageKey(stage) {
  if (['S3', 'S4', 'S5', 'S6', 'S3-S6'].includes(stage)) return 'S3-S6'
  return stage || ''
}

export default function UploadTab({
  documentId, setDocumentId,
  jobId, setJobId,
  results, setResults,
  isRunning, setIsRunning,
  progress, setProgress,
  config, apiBase, showToast, setActiveTab,
}) {
  const [dragging, setDragging]       = useState(false)
  const [fileName, setFileName]       = useState(null)
  const [tokenCount, setTokenCount]   = useState(null)
  const [showProgress, setShowProgress] = useState(false)
  const fileInputRef  = useRef(null)
  const pollTimerRef  = useRef(null)

  useEffect(() => () => { if (pollTimerRef.current) clearInterval(pollTimerRef.current) }, [])

  /* ── File handling ───────────────────────────────────────────── */
  async function handleFile(file) {
    setFileName(file.name)
    setTokenCount(null)
    setDocumentId(null)

    const fd = new FormData()
    fd.append('file', file)
    try {
      const res = await fetch(`${apiBase}/upload`, { method: 'POST', body: fd })
      if (!res.ok) throw new Error((await res.json()).detail || 'Upload failed')
      const data = await res.json()
      setDocumentId(data.document_id)
      setTokenCount(data.token_count ?? null)
      const tokenLabel = data.token_count != null
        ? ` — ${data.token_count.toLocaleString()} tokens`
        : ' — text extraction will run with analysis'
      showToast(`✓ Uploaded ${file.name}${tokenLabel}`, 'success')
    } catch (err) {
      showToast(`Upload failed: ${err.message}`, 'error')
      setFileName(null)
    }
  }

  function onDrop(e) {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }

  /* ── Pipeline ────────────────────────────────────────────────── */
  async function runPipeline() {
    if (!documentId) return
    setIsRunning(true)
    setShowProgress(true)
    setResults(null)
    setProgress({ stage: 'S1', pct: 0, message: 'Starting…', stageKey: 'S1' })

    const fd = new FormData()
    fd.append('config', JSON.stringify(config))

    try {
      const res = await fetch(`${apiBase}/run/${documentId}`, { method: 'POST', body: fd })
      if (!res.ok) throw new Error((await res.json()).detail || 'Pipeline failed to start')
      const { job_id } = await res.json()
      setJobId(job_id)
      startPolling(job_id)
    } catch (err) {
      showToast(`Failed: ${err.message}`, 'error')
      setIsRunning(false)
    }
  }

  function startPolling(jid) {
    if (pollTimerRef.current) clearInterval(pollTimerRef.current)
    pollTimerRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${apiBase}/status/${jid}`)
        if (!res.ok) return
        const data = await res.json()
        setProgress({ stage: data.stage, pct: data.progress || 0, message: data.message || '', stageKey: normalizeStageKey(data.stage) })

        if (data.status === 'complete') {
          clearInterval(pollTimerRef.current)
          await loadResults(jid)
        } else if (data.status === 'error') {
          clearInterval(pollTimerRef.current)
          showToast(`Pipeline error: ${data.message}`, 'error')
          setIsRunning(false)
        }
      } catch (_) {}
    }, 1000)
  }

  async function loadResults(jid) {
    try {
      const res = await fetch(`${apiBase}/results/${jid}`)
      if (!res.ok) throw new Error('Failed to fetch results')
      const data = await res.json()
      setResults(data)
      setProgress({ stage: 'DONE', pct: 100, message: 'Pipeline complete!', stageKey: 'DONE' })
      showToast('Pipeline complete! Explore your chunks.', 'success')
    } catch (err) {
      showToast(`Results error: ${err.message}`, 'error')
    } finally {
      setIsRunning(false)
    }
  }

  function reset() {
    if (pollTimerRef.current) clearInterval(pollTimerRef.current)
    setDocumentId(null)
    setJobId(null)
    setResults(null)
    setFileName(null)
    setTokenCount(null)
    setShowProgress(false)
    setIsRunning(false)
    setProgress({ stage: '', pct: 0, message: '', stageKey: '' })
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const stageIdx = STAGES.indexOf(progress.stageKey)

  return (
    <div>
      {/* Drop zone */}
      <div
        className={`drop-zone${dragging ? ' drag-over' : ''}`}
        onClick={() => fileInputRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          hidden
          accept=".pdf,.txt,.md,.py,.js,.ts,.java,.cpp,.c,.go,.rs,.html"
          onChange={e => e.target.files[0] && handleFile(e.target.files[0])}
        />
        <div className="drop-icon">📄</div>
        {fileName
          ? <div className="drop-primary">
              <span className="drop-file">{fileName}</span>
              {tokenCount != null && <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}> — {tokenCount.toLocaleString()} tokens</span>}
            </div>
          : <div className="drop-primary">Drop a file or click to upload</div>
        }
        <div className="drop-secondary">PDF · TXT · Markdown · Python · JavaScript · Java · C/C++</div>
      </div>

      {/* Actions */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 20, alignItems: 'center' }}>
        <button className="btn btn-primary" disabled={!documentId || isRunning} onClick={runPipeline}>
          {isRunning ? '⏳ Running…' : '▶ Start Analysis'}
        </button>
        <button className="btn btn-secondary" onClick={reset}>
          ↺ Reset
        </button>
        {documentId && !isRunning && !results && (
          <span style={{ fontSize: 12, color: 'var(--green)', fontWeight: 600 }}>✓ Ready to run</span>
        )}
      </div>

      {/* Progress */}
      {showProgress && (
        <div className="card" style={{ marginBottom: 14 }}>
          <div className="card-title" style={{ marginBottom: 8 }}>
            Pipeline Status
            {progress.pct === 100 && <span className="badge badge-green">Complete</span>}
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>
            <span>{progress.message || progress.stage}</span>
            <span style={{ fontWeight: 700, fontFamily: 'JetBrains Mono', color: 'var(--yellow-dark)' }}>{progress.pct}%</span>
          </div>
          <div className="progress-bar-track">
            <div className="progress-bar-fill" style={{ width: progress.pct + '%' }} />
          </div>

          <div className="stage-pills">
            {STAGES.map((s, i) => (
              <span
                key={s}
                className={`stage-pill${i < stageIdx ? ' done' : i === stageIdx ? ' active' : ''}`}
              >
                {i < stageIdx ? '✓ ' : ''}{STAGE_LABELS[s]}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Summary */}
      {results && <SummarySection results={results} setActiveTab={setActiveTab} />}
    </div>
  )
}

function SummarySection({ results, setActiveTab }) {
  const s    = results.summary || {}
  const prof = results.doc_profile || {}
  const metrics = prof.metrics || {}
  const metricDetails = prof.metric_details || {}
  const evalTable = results.stage2_evaluation || []
  const stage2Winner = s.stage2_winning_strategy || results.stage_details?.s2?.selected_strategy

  const stats = [
    { label: 'Doc Type',      value: s.doc_type || '—' },
    { label: 'Domain',        value: s.domain || '—' },
    { label: 'Chunks',        value: s.chunk_count ?? '—' },
    { label: 'Mean Score',    value: s.mean_chunk_score != null ? (s.mean_chunk_score * 100).toFixed(1) + '%' : '—' },
    { label: 'S2 Winner',     value: stage2Winner ? stage2Winner.replace(/_/g, ' ') : '—' },
    { label: 'RL Iterations', value: s.rl_iterations ?? '—' },
    { label: 'Tokens',        value: (s.token_count || 0).toLocaleString() },
    { label: 'Final Reward',  value: results.reward_history?.length ? results.reward_history.at(-1).toFixed(4) : '—' },
  ]

  return (
    <>
      <div className="card" style={{ marginBottom: 14 }}>
        <div className="card-title">
          Executive Summary
          <span className="badge badge-green">Complete</span>
        </div>

        <div className="stat-grid" style={{ marginBottom: 16 }}>
          {stats.map(({ label, value }) => (
            <div className="stat-card" key={label}>
              <div className="stat-value">{value}</div>
              <div className="stat-label">{label}</div>
            </div>
          ))}
        </div>

        {Object.keys(metrics).length > 0 && (
          <>
            <hr className="divider" />
            <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: 10 }}>
              Document Quality Metrics
            </div>
            {DOC_METRICS.map(({ key, label }) => {
              const v = metrics[key] || 0
              const detail = metricDetails[key] || {}
              const pct = Math.round(v * 100)
              const color = pct >= 70 ? 'var(--green)' : pct >= 40 ? 'var(--amber)' : 'var(--red)'
              return (
                <div className="metric-block" key={key}>
                  <div className="metric-bar-row">
                    <div className="metric-bar-key">
                      <span>{detail.label || label}</span>
                      <small>{key}</small>
                    </div>
                    <div className="metric-bar-track">
                      <div className="metric-bar-fill" style={{ width: pct + '%', background: color }} />
                    </div>
                    <div className="metric-bar-val">{pct}%</div>
                  </div>
                  {detail.reason && <div className="metric-reason">{detail.reason}</div>}
                </div>
              )
            })}
          </>
        )}

        <div style={{ marginTop: 16, display: 'flex', gap: 10 }}>
          <button className="btn btn-secondary" style={{ fontSize: 12 }} onClick={() => setActiveTab('explorer')}>
            View Chunks →
          </button>
          <button className="btn btn-secondary" style={{ fontSize: 12 }} onClick={() => setActiveTab('inspector')}>
            Inspect Pipeline →
          </button>
        </div>
      </div>

      {/* Strategy evaluation table */}
      {evalTable.length > 0 && (
        <EvaluationTable
          table={evalTable}
          winner={stage2Winner}
          title="Stage 2 Chunking Evaluation"
          metricCols={STAGE2_METRICS}
          note="This scores raw Stage 2 chunk candidates before entropy, embeddings, retrieval, RL, or ground-truth QA. Score blends size distribution, expected chunk count, boundary integrity, structure preservation, local cohesion, token distinctness, coverage sanity, and chunking speed."
        />
      )}
    </>
  )
}
