import React, { useEffect, useState } from 'react'
import { getFeatures, getHealth, getModel, predict } from './api.js'
import FeatureForm from './components/FeatureForm.jsx'
import ProbabilityBar from './components/ProbabilityBar.jsx'
import ShapChart from './components/ShapChart.jsx'
import Feedback from './components/Feedback.jsx'
import { Report, Verdict, Warnings } from './components/Verdict.jsx'

// Real objects from the Kepler KOI table, one per class, so the demo
// starts from something the archive actually contains.
const EXAMPLES = {
  'Confirmed planet (K00752.01)': {
    koi_period: 9.488,
    koi_duration: 2.9575,
    koi_depth: 615.8,
    koi_prad: 2.26,
    koi_teq: 793.0,
    koi_impact: 0.146,
    koi_insol: 93.59,
    koi_steff: 5455.0,
    koi_slogg: 4.467,
    koi_srad: 0.927,
    koi_kepmag: 15.347,
  },
  'Candidate (K00753.01)': {
    koi_period: 19.8991,
    koi_duration: 1.7822,
    koi_depth: 10829.0,
    koi_prad: 14.6,
    koi_teq: 638.0,
    koi_impact: 0.969,
    koi_insol: 39.3,
    koi_steff: 5853.0,
    koi_slogg: 4.544,
    koi_srad: 0.868,
    koi_kepmag: 15.436,
  },
  'False positive (K00754.01)': {
    koi_period: 1.737,
    koi_duration: 2.4064,
    koi_depth: 8079.2,
    koi_prad: 33.46,
    koi_teq: 1395.0,
    koi_impact: 1.276,
    koi_insol: 891.96,
    koi_steff: 5805.0,
    koi_slogg: 4.564,
    koi_srad: 0.791,
    koi_kepmag: 15.597,
  },
}

export default function App() {
  const [features, setFeatures] = useState([])
  const [values, setValues] = useState({})
  const [result, setResult] = useState(null)
  const [meta, setMeta] = useState(null)
  const [health, setHealth] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    getFeatures()
      .then(setFeatures)
      .catch((err) => setError(err.message))
    getModel()
      .then(setMeta)
      .catch(() => setMeta(null))
    getHealth()
      .then(setHealth)
      .catch(() => setHealth(null))
  }, [])

  const setValue = (name, value) => setValues((previous) => ({ ...previous, [name]: value }))

  const loadExample = (name) => setValues({ ...EXAMPLES[name] })

  const submit = async () => {
    setBusy(true)
    setError(null)
    try {
      const payload = Object.fromEntries(
        Object.entries(values).filter(([, value]) => value !== null && value !== undefined),
      )
      setResult(await predict(payload))
    } catch (err) {
      setError(err.message)
      setResult(null)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app">
      <header className="masthead">
        <h1>ExoVet</h1>
        <p>
          Vet a transiting signal and get a class, a calibrated probability, a
          prediction set with a coverage guarantee, and a plain-English report
          built only from the features that drove the decision.
        </p>
        <div className="meta">
          {meta && (
            <>
              <span className="chip">
                model <strong>{meta.model}</strong>
              </span>
              <span className="chip">
                macro-F1 <strong>{meta.metrics.macro_f1?.toFixed(3)}</strong>
              </span>
              <span className="chip">
                coverage target <strong>{(meta.target_coverage * 100).toFixed(0)}%</strong>
              </span>
            </>
          )}
          {health && (
            <span className="chip">
              service <strong>{health.status}</strong>
            </span>
          )}
        </div>
      </header>

      <div className="layout">
        <div className="stack">
          <FeatureForm
            features={features}
            values={values}
            onChange={setValue}
            onSubmit={submit}
            busy={busy}
          />
          <div className="panel">
            <h2>Examples</h2>
            <p className="hint">Real objects from the Kepler KOI table.</p>
            <div className="feedback-row">
              {Object.keys(EXAMPLES).map((name) => (
                <button key={name} onClick={() => loadExample(name)}>
                  {name}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="stack">
          {error && <div className="error">{error}</div>}

          {!result && !error && (
            <div className="panel">
              <h2>Verdict</h2>
              <p className="empty">
                Fill in the measurements, or load one of the examples, and the
                assessment will appear here.
              </p>
            </div>
          )}

          {result && (
            <>
              <div className="panel">
                <h2>Verdict</h2>
                <Verdict result={result} />
                <div style={{ marginTop: 18 }}>
                  <ProbabilityBar probabilities={result.probabilities} />
                </div>
                <Warnings warnings={result.warnings} />
              </div>

              <div className="panel">
                <h2>Why</h2>
                <p className="hint">
                  The features that moved the decision, largest contribution first.
                </p>
                <ShapChart features={result.top_features} />
                <div style={{ marginTop: 16 }}>
                  <Report text={result.report} />
                </div>
              </div>

              <div className="panel">
                <h2>Correct the model</h2>
                <Feedback result={result} values={values} objectId={null} />
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
