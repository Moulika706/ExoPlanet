import React from 'react'

const CLASS_COLOR = {
  CONFIRMED: 'var(--confirmed)',
  CANDIDATE: 'var(--candidate)',
  'FALSE POSITIVE': 'var(--false-positive)',
}

/** The verdict: the class, the probability, and the review badge. */
export function Verdict({ result }) {
  const ambiguous = result.prediction_set.length > 1
  return (
    <div>
      <div className="verdict">
        <span className="class" style={{ color: CLASS_COLOR[result.predicted_class] }}>
          {result.predicted_class}
        </span>
        <span className="prob">{(result.probability * 100).toFixed(1)}%</span>
        <span className={`badge ${ambiguous ? 'review' : 'clear'}`}>
          <span className="dot" />
          {ambiguous ? 'Needs manual review' : 'Confident'}
        </span>
      </div>
      <div className="sets">
        {result.prediction_set.map((cls) => (
          <span key={cls} className="set-chip in">
            {cls}
          </span>
        ))}
        <span className="set-chip">
          coverage target {(result.target_coverage * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  )
}

/** The plain-English report the model wrote from the top features. */
export function Report({ text }) {
  if (!text) return null
  return (
    <div className="report">
      <p>{text}</p>
    </div>
  )
}

/** The warnings that travel with every answer. */
export function Warnings({ warnings }) {
  if (!warnings?.length) return null
  return (
    <>
      {warnings.map((warning) => (
        <div className="warn" key={warning}>
          {warning}
        </div>
      ))}
    </>
  )
}
