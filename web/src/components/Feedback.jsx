import React, { useState } from 'react'
import { sendFeedback } from '../api.js'

const CLASSES = ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']

/** Lets a user correct a verdict. The correction is logged for later. */
export default function Feedback({ result, values, objectId }) {
  const [corrected, setCorrected] = useState(result.predicted_class)
  const [comment, setComment] = useState('')
  const [state, setState] = useState('idle')

  const submit = async () => {
    setState('sending')
    try {
      await sendFeedback({
        values,
        predicted_class: result.predicted_class,
        corrected_class: corrected,
        comment: comment || null,
        object_id: objectId ?? null,
      })
      setState('sent')
      setComment('')
    } catch (error) {
      setState(`error: ${error.message}`)
    }
  }

  return (
    <div>
      <p className="hint">
        Disagree with the verdict? Send the correction back. It is logged with the
        measurements, so the model can be checked against what the experts said.
      </p>
      <div className="feedback-row">
        <select value={corrected} onChange={(event) => setCorrected(event.target.value)}>
          {CLASSES.map((cls) => (
            <option key={cls} value={cls}>
              {cls}
            </option>
          ))}
        </select>
        <input
          type="text"
          placeholder="Comment (optional)"
          value={comment}
          onChange={(event) => setComment(event.target.value)}
          style={{
            flex: 2,
            minWidth: 160,
            padding: '7px 9px',
            background: 'var(--bg)',
            border: '1px solid var(--line)',
            borderRadius: 6,
            color: 'var(--text)',
            fontFamily: 'inherit',
            fontSize: 13.5,
          }}
        />
        <button onClick={submit} disabled={state === 'sending'}>
          {state === 'sending' ? 'Sending...' : 'Send correction'}
        </button>
      </div>
      {state === 'sent' && <div className="warn">Correction logged. Thank you.</div>}
      {state.startsWith('error') && <div className="error">{state}</div>}
    </div>
  )
}
