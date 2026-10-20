import React from 'react'

/**
 * The measurement form.
 *
 * Every field is optional. A blank field is imputed with the training
 * median inside the pipeline, so a partial set of measurements still
 * produces a verdict - the model will just lean more on the features it
 * was given.
 */
export default function FeatureForm({ features, values, onChange, onSubmit, busy }) {
  const handle = (name) => (event) => {
    const raw = event.target.value
    onChange(name, raw === '' ? null : Number(raw))
  }

  return (
    <form
      className="panel"
      onSubmit={(event) => {
        event.preventDefault()
        onSubmit()
      }}
    >
      <h2>Measured properties</h2>
      <p className="hint">
        Leave a field blank if it was not measured. The model will say so in the
        report.
      </p>

      {features.map((feature) => (
        <div className="field" key={feature.name}>
          <label htmlFor={feature.name}>
            <span>{feature.display_name}</span>
            {feature.unit && <span className="unit">{feature.unit}</span>}
          </label>
          <input
            id={feature.name}
            type="number"
            step="any"
            inputMode="decimal"
            placeholder="not measured"
            value={values[feature.name] ?? ''}
            onChange={handle(feature.name)}
          />
          {feature.description && <div className="desc">{feature.description}</div>}
        </div>
      ))}

      <div className="actions">
        <button type="submit" className="primary" disabled={busy}>
          {busy ? 'Vetting...' : 'Vet this object'}
        </button>
      </div>
    </form>
  )
}
