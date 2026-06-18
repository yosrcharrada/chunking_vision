import React, { useState } from 'react'
import '../styles/ScoringFormulas.css'

export default function ScoringFormulas() {
  const [expandedFormula, setExpandedFormula] = useState(null)

  const formulas = {
    tsallis: {
      title: 'S3 · Tsallis q-Entropy',
      formula: 'S_q(p) = (1 − Σ pᵢ^q) / (q − 1)   →   Shannon H = −Σ pᵢ log pᵢ as q → 1',
      description: 'Generalised entropy of the boundary distribution p (normalised semantic-shift peaks). q ∈ [−1, 1]; the platform recovers the Zhong et al. Shannon model exactly at q = 1.',
      interpretation: {
        'q < 1': 'amplifies weak/rare shifts → finer chunks',
        'q = 1': 'Shannon entropy (perplexity baseline)',
        'q > 1': 'suppresses weak shifts → coarser chunks',
      },
    },
    diversity: {
      title: 'S3 · Hill Diversity Number (boundary count driver)',
      formula: 'D_q = (Σ pᵢ^q)^(1/(1−q))   ·   D_1 = exp(H)   ·   target_boundaries = round(D_q)',
      description: 'The "effective number of real boundaries" at granularity q. D_q is non-increasing in q, so q alone decides HOW MANY of the semantic-shift peaks become chunk boundaries. This is what makes Shannon (q=1) and Tsallis (q≠1) produce genuinely different chunkings.',
    },
    lstm: {
      title: 'S3 · LSTM Boundary Refinement',
      formula: 'salience_i = 0.65·d̂_i + 0.35·σ(Wₚ·hᵢ),   hᵢ = LSTM(x₀..xᵢ)',
      description: 'A deterministic forward LSTM (fixed seed) runs over a 5-dim qentropy feature sequence — normalised shift d̂, probability mass pᵢ, excess-over-baseline, pointwise Tsallis weight pᵢ^q, structural flag — to add document-order context before the top D_q gaps are opened.',
      components: [
        { name: 'd̂ (shift)', weight: 0.65, description: '1 − cos(unitᵢ, unitᵢ₊₁), windowed & normalised' },
        { name: 'LSTM score', weight: 0.35, description: 'context-aware salience over the qentropy features' },
      ],
    },
    treeEntropy: {
      title: 'S3 · Tree Entropy (paper-faithful fingerprint)',
      formula: 'H(N) = Σ_internal log Z_K(size),   Z_K(n) = C(n+K−1, n−1),   h_K = H(N)/N',
      description: 'Zhong et al. recursive K-ary segmentation entropy over the resulting tree, with a Tsallis-q generalisation S_q that → H as q → 1. Reported per document as Shannon nats, entropy rate h_K, and Tsallis nats.',
    },
    tableMetrics: {
      title: 'Evaluation · Table-I Retrieval Metrics',
      formula: 'winner = argmax primary  →  bootstrap-CI ties  →  lowest retrieval token cost',
      description: 'Every chunking (each strategy + the GA winner) is scored with the qentropy project metrics. Ranking uses the cosine rank mean (or LLM answer-correctness when the judge runs); precision/recall/f1 are shown as diagnostics only.',
      components: [
        { name: 'Precision/Recall/F1', weight: 0, description: 'cosine-threshold diagnostics (not used for ranking)' },
        { name: 'MRR', weight: 0, description: 'mean reciprocal rank of the first relevant chunk' },
        { name: 'NDCG@5', weight: 0, description: 'graded relevance = max(0, cos(chunk, ground-truth))' },
        { name: 'ss2fd', weight: 0, description: 'mean cos(chunk, full-document embedding)' },
        { name: 'SRGT', weight: 0, description: 'mean cos(top-5 retrieved chunk, ground-truth)' },
        { name: 'QCS', weight: 0, description: 'mean cos(query, top-5 retrieved chunks)' },
        { name: 'Token Cost', weight: 0, description: 'mean tokens across top-5 retrieved (lower is cheaper)' },
      ],
    },
    gaFitness: {
      title: 'S7 · Genetic Algorithm Fitness',
      formula: 'fitness = mean(MRR, NDCG, SRGT, QCS)   [or LLM answer-correct]   ·   q tuned in [−1, 1]',
      description: 'The GA evolves S2 sizing, the S4 merge threshold, the qentropy min-token floor and the Tsallis q, scoring each individual with the same Table-I signal used for the final ranking. Offline (no API key) it falls back to a label-free coherence/separation/balance quality.',
    },
  }

  return (
    <div className="scoring-formulas">
      <h2>Scoring Formulas & Calculations</h2>
      
      <div className="formula-cards">
        {Object.entries(formulas).map(([key, formula]) => (
          <div key={key} className="formula-card">
            <div 
              className="formula-header"
              onClick={() => setExpandedFormula(expandedFormula === key ? null : key)}
            >
              <h3>{formula.title}</h3>
              <span className="expand-icon">{expandedFormula === key ? '▼' : '▶'}</span>
            </div>
            
            {expandedFormula === key && (
              <div className="formula-content">
                <div className="formula-box">
                  <code>{formula.formula}</code>
                </div>
                
                {formula.model && (
                  <div className="info-section">
                    <strong>Model:</strong> {formula.model}
                  </div>
                )}
                
                {formula.threshold && (
                  <div className="info-section">
                    <strong>Threshold:</strong> {formula.threshold}
                  </div>
                )}
                
                {formula.description && (
                  <div className="info-section">
                    <strong>Description:</strong> {formula.description}
                  </div>
                )}
                
                {formula.components && (
                  <div className="components-section">
                    <strong>Components:</strong>
                    <table className="components-table">
                      <thead>
                        <tr>
                          <th>Component</th>
                          <th>Weight</th>
                          <th>Description</th>
                        </tr>
                      </thead>
                      <tbody>
                        {formula.components.map((comp, i) => (
                          <tr key={i}>
                            <td><strong>{comp.name}</strong></td>
                            <td>{comp.weight ? `${(comp.weight * 100).toFixed(0)}%` : '—'}</td>
                            <td>{comp.description}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                
                {formula.example && (
                  <div className="example-section">
                    <strong>Example:</strong>
                    <p>0.24×{formula.example.values[0]} + 0.18×{formula.example.values[1]} + ... = {formula.example.result.toFixed(3)}</p>
                  </div>
                )}
                
                {formula.interpretation && (
                  <div className="interpretation-section">
                    <strong>Interpretation:</strong>
                    <ul>
                      {Object.entries(formula.interpretation).map(([key, val]) => (
                        <li key={key}><strong>{key}:</strong> {val}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
