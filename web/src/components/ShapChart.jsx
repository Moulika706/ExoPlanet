import React from 'react'

/**
 * A diverging bar chart of the SHAP contributions.
 *
 * The bar for each feature is drawn from the centre line: to the right
 * when the feature pushes the assessment towards the predicted class, to
 * the left when it counts against it. The width is the magnitude of the
 * contribution relative to the largest one, so the chart is readable
 * whatever the scale.
 */
export default function ShapChart({ features }) {
  if (!features?.length) return null

  const largest = Math.max(...features.map((f) => Math.abs(f.shap_value)), 1e-9)

  return (
    <div>
      {features.map((feature) => {
        const fraction = Math.abs(feature.shap_value) / largest
        const positive = feature.shap_value >= 0
        const width = `${Math.max(fraction * 50, 0.6)}%`
        const left = positive ? '50%' : `${50 - fraction * 50}%`
        return (
          <div className="shap-row" key={feature.feature}>
            <div className="name">
              {feature.display_name}
              <span>{feature.formatted_value}</span>
            </div>
            <div className="shap-track">
              <div className="mid" style={{ left: '50%' }} />
              <div
                className={`bar ${positive ? 'pos' : 'neg'}`}
                style={{ left, width }}
                title={`${feature.shap_value >= 0 ? 'pushes towards' : 'counts against'} the predicted class`}
              />
            </div>
            <div className="value">
              {feature.shap_value >= 0 ? '+' : ''}
              {feature.shap_value.toFixed(3)}
            </div>
          </div>
        )
      })}
    </div>
  )
}
