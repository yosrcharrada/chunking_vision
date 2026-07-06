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
  // ── S3 qentropy (Tsallis q + diversity number D_q) ──
  q_entropy_param: 1.0,     // q ∈ [-1, 1]; 1.0 == Shannon baseline
  K: 4,
  min_chunk_tokens: 20,
  max_chunk_tokens: 320,
  window: 1,
  // ── S4 ──
  tau_sem: 0.75,
  s4_similarity: 'cosine',  // 'cosine' (classical) | 'qcosine' (Fitouhi–Bouzeffour q-cosine, base=q²)
  // ── Evaluation (engine.metrics, Table I) ──
  qa_count: 12,
  judge_answerability: false,
  // ── Embedding backend (engine.embeddings) ──
  embedding_backend: null,  // null → openai if key else multilingual
  // ── S7 GA ──
  ga_population: 6,
  ga_generations: 3,
  ga_workers: 4,
  max_iterations: 30,
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
            <div className="workspace-kicker">qEntropy Document Intelligence</div>
            <h1 className="workspace-title">Tsallis qEntropy Chunking Platform</h1>
            <p className="workspace-copy">
              Upload any document. The Tsallis q-entropy diversity number drives the S3 boundaries,
              the genetic algorithm tunes q, and every method is scored with the Table-I retrieval metrics.
            </p>
          </div>
          <div className="workspace-chips">
            <span className="chip"><strong>q</strong>-entropy S3</span>
            <span className="chip"><strong>GA</strong>-tuned q</span>
            <span className="chip"><strong>Table I</strong> metrics</span>
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
