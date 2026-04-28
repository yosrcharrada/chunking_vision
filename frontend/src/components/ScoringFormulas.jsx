import React, { useState } from 'react'
import '../styles/ScoringFormulas.css'

export default function ScoringFormulas() {
  const [expandedFormula, setExpandedFormula] = useState(null)

  const formulas = {
    strategyScore: {
      title: 'S2 Strategy Quality Score',
      formula: 'Score = 0.24×Fit + 0.18×Stability + 0.18×InRange + 0.17×BoundDiv + 0.13×Integrity + 0.10×Completion',
      components: [
        { name: 'Fit', weight: 0.24, description: 'How close avg chunk size is to 72% of n_max' },
        { name: 'Stability', weight: 0.18, description: 'How uniform are chunk sizes (1 - std/avg)' },
        { name: 'InRange', weight: 0.18, description: 'Percentage of chunks within [n_min, n_max]' },
        { name: 'BoundDiv', weight: 0.17, description: 'Mean JSD between consecutive chunks' },
        { name: 'Integrity', weight: 0.13, description: 'Fraction of structurally complete chunks' },
        { name: 'Completion', weight: 0.10, description: 'Percentage of boundaries not breaking sentences' },
      ],
      example: {
        values: [0.85, 0.92, 0.88, 0.72, 0.95, 0.91],
        result: 0.865,
      },
    },
    boundaryScore: {
      title: 'S4 Boundary Similarity Score',
      formula: 'Score = 0.25×Lexical + 0.20×Syntactic + 0.15×TokenType + 0.20×Structural + 0.20×Semantic',
      components: [
        { name: 'Lexical', weight: 0.25, description: 'BLEU-style n-gram overlap' },
        { name: 'Syntactic', weight: 0.20, description: 'Stopword overlap' },
        { name: 'TokenType', weight: 0.15, description: 'POS tag set Jaccard' },
        { name: 'Structural', weight: 0.20, description: 'Bracket/brace balance' },
        { name: 'Semantic', weight: 0.20, description: 'Embedding cosine + multi-scale' },
      ],
      interpretation: {
        high: '>0.75 - Chunks too similar, merge candidate',
        med: '0.25-0.75 - Valid boundary',
        low: '<0.25 - Strong boundary',
      },
    },
    totalReward: {
      title: 'S7 Total Reward',
      formula: 'Reward = 0.35×Quality + 0.30×Coverage + 0.20×Consistency + 0.15×Efficiency',
      components: [
        { name: 'Quality', weight: 0.35, description: 'Boundary integrity, ICC, structure compliance' },
        { name: 'Coverage', weight: 0.30, description: 'Text covered, chunk count vs target' },
        { name: 'Consistency', weight: 0.20, description: 'Size uniformity, metric stability' },
        { name: 'Efficiency', weight: 0.15, description: 'Distance from target chunk size (300 words)' },
      ],
    },
    ppl: {
      title: 'S3 Perplexity Validation',
      formula: 'Merge valid if: PPL_merged < max(PPL_a, PPL_b) × threshold',
      model: 'DistilGPT2 (~82M parameters)',
      threshold: 'Default: 1.1 (allow 10% PPL increase)',
      description: 'Validates that merged chunks maintain coherence in language model',
    },
    entropyRate: {
      title: 'S3 Entropy Rate (Intra-chunk)',
      formula: 'EntropyRate = sigmoid(mean(JSD(sent_i, sent_{i+1})))',
      description: 'Measures sentence-level vocabulary divergence within chunks',
      interpretation: {
        low: '0.0-0.3 - Highly coherent sentences',
        med: '0.3-0.7 - Mixed coherence',
        high: '0.7-1.0 - Diverse/incoherent sentences',
      },
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
                            <td>{(comp.weight * 100).toFixed(0)}%</td>
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
