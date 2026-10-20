import React from 'react'

const CLASS_COLOR = {
  CONFIRMED: 'var(--confirmed)',
  CANDIDATE: 'var(--candidate)',
  'FALSE POSITIVE': 'var(--false-positive)',
}

export default function ProbabilityBar({ probabilities }) {
  const entries = Object.entries(probabilities || {})
  if (!entries.length) return null

  const total = entries.reduce((sum, [, value]) => sum + value, 0) || 1

  return (
    <div>
      <div className="probbar">
        {entries.map(([name, value]) => (
          <div
            key={name}
            className={name}
            style={{ width: `${(value / total) * 100}%` }}
            title={`${name}: ${(value * 100).toFixed(1)}%`}
          >
            {value >= 0.12 ? `${(value * 100).toFixed(0)}%` : ''}
          </div>
        ))}
      </div>
      <div className="probbar-key">
        {entries.map(([name, value]) => (
          <span key={name} className={name.replace(' ', '-')}>
            {name} {(value * 100).toFixed(1)}%
          </span>
        ))}
      </div>
    </div>
  )
}

export { CLASS_COLOR }
