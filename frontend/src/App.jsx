import { useState, useRef, useCallback } from 'react'
import Sidebar from './components/Sidebar'
import UploadTab from './components/UploadTab'
import ExplorerTab from './components/ExplorerTab'
import InspectorTab from './components/InspectorTab'
import ExportTab from './components/ExportTab'

const DEFAULT_CONFIG = {
  chunking_strategy: 'auto',
  n_min: 80,
  n_max: 500,
  tau_jsd_low: 0.15,
  tau_jsd_high: 0.45,
  entropy_metric: 'hybrid',
  hybrid_lambda: 0.60,
  threshold_mode: 'percentile',
  tau_percentile_low: 25,
  tau_percentile_high: 75,
  tau_sem: 0.75,
  max_iterations: 30,
  ga_population: 8,       // NEW
  ga_generations: 5,      // NEW
  ga_workers: 4,          // NEW
  alpha: 0.4,
  beta: 0.4,
  lambda: 0.2,
  embedding_model: 'mxbai-embed-large',  // NEW: Updated to use mxbai-embed-large
  ensemble_models: ['mxbai-embed-large', 'all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'jina-embeddings-v2-base-en'],
}

const TABS = [
  { id: 'upload',    label: 'Submit & Analyze',  icon: '↑' },
  { id: 'explorer',  label: 'Review Findings',    icon: '⊞' },
  { id: 'inspector', label: 'Process Controls',   icon: '◈' },
  { id: 'export',    label: 'Deliverables',        icon: '↓' },
]

export default function App() {
  const [config, setConfig]       = useState(DEFAULT_CONFIG)
  const [apiBase, setApiBase]     = useState('http://localhost:8000')
  const [documentId, setDocumentId] = useState(null)
  const [jobId, setJobId]         = useState(null)
  const [results, setResults]     = useState(null)
  const [activeTab, setActiveTab] = useState('upload')
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress]   = useState({ stage: '', pct: 0, message: '', stageKey: '' })
  const [toast, setToast]         = useState(null)
  const toastRef = useRef(null)

  const showToast = useCallback((message, type = 'info') => {
    setToast({ message, type })
    if (toastRef.current) clearTimeout(toastRef.current)
    toastRef.current = setTimeout(() => setToast(null), 4500)
  }, [])

  return (
    <div className="app-layout">
      <Sidebar
        config={config}
        setConfig={setConfig}
        apiBase={apiBase}
        setApiBase={setApiBase}
      />

      <div className="main-area">
        {/* Workspace header */}
        <div className="workspace-header">
          <div>
            <div className="workspace-kicker">AI Document Intelligence</div>
            <h1 className="workspace-title">Adaptive PDF Chunking Platform</h1>
            <p className="workspace-copy">
              Upload any document. All chunking methods run through the pipeline, then the evaluator compares the full result set.
            </p>
          </div>
          <div className="workspace-chips">
            <span className="chip"><strong>8</strong> stages</span>
            <span className="chip"><strong>All</strong> methods</span>
            <span className="chip"><strong>S8</strong> evaluation</span>
          </div>
        </div>

        {/* Tab bar */}
        <nav className="tab-bar">
          {TABS.map(t => (
            <button
              key={t.id}
              className={`tab-btn${activeTab === t.id ? ' active' : ''}`}
              onClick={() => setActiveTab(t.id)}
            >
              <span className="tab-icon">{t.icon}</span>
              {t.label}
            </button>
          ))}
        </nav>

        {/* Tab content */}
        <div className="tab-content">
          {activeTab === 'upload' && (
            <UploadTab
              documentId={documentId}
              setDocumentId={setDocumentId}
              jobId={jobId}
              setJobId={setJobId}
              results={results}
              setResults={setResults}
              isRunning={isRunning}
              setIsRunning={setIsRunning}
              progress={progress}
              setProgress={setProgress}
              config={config}
              apiBase={apiBase}
              showToast={showToast}
              setActiveTab={setActiveTab}
            />
          )}
          {activeTab === 'explorer' && (
            <ExplorerTab chunks={results?.chunks || []} />
          )}
          {activeTab === 'inspector' && (
            <InspectorTab results={results} config={config} />
          )}
          {activeTab === 'export' && (
            <ExportTab jobId={jobId} results={results} apiBase={apiBase} />
          )}
        </div>
      </div>

      {/* Toast */}
      {toast && (
        <div className={`toast toast-${toast.type}`}>
          {toast.message}
        </div>
      )}
    </div>
  )
}
