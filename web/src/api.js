const BASE = import.meta.env.VITE_API_BASE ?? ''

async function request(path, options = {}) {
  const response = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      if (body.detail) detail = body.detail
    } catch {
      // the body was not JSON; keep the status line
    }
    throw new Error(detail)
  }
  return response.json()
}

export const getHealth = () => request('/health')
export const getModel = () => request('/model')
export const getFeatures = () => request('/features')

export const predict = (values, objectId) =>
  request('/predict', {
    method: 'POST',
    body: JSON.stringify({ values, object_id: objectId ?? null }),
  })

export const sendFeedback = (payload) =>
  request('/feedback', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
