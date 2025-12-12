// API route to proxy sign language prediction to HuggingFace Space
const HF_BASE_URL = 'https://khalood619-signbridge-api.hf.space'
const SIGN_PREDICT_URL = `${HF_BASE_URL}/sign/predict`

export const config = {
  api: {
    bodyParser: {
      sizeLimit: '50mb',
    },
    responseLimit: false,
  },
}

export default async function handler(req, res) {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Origin', '*')
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS')
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type')

  if (req.method === 'OPTIONS') {
    return res.status(200).end()
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const { videoBlob, topK = 5 } = req.body

    if (!videoBlob) {
      return res.status(400).json({ error: 'No video data provided' })
    }

    // Convert base64 to buffer
    const videoBuffer = Buffer.from(videoBlob, 'base64')
    console.log('Video buffer size:', videoBuffer.length)

    // Create form data for the HF API
    const FormData = (await import('form-data')).default
    const formData = new FormData()
    formData.append('video', videoBuffer, {
      filename: 'clip.webm',
      contentType: 'video/webm',
    })
    formData.append('top_k', String(topK))

    // Send to HuggingFace Space
    console.log('Sending to HF API...')
    const response = await fetch(SIGN_PREDICT_URL, {
      method: 'POST',
      body: formData,
      headers: formData.getHeaders(),
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error('HF API error:', response.status, errorText)
      return res.status(response.status).json({ 
        error: `API error: ${response.status}`,
        details: errorText.slice(0, 200)
      })
    }

    const result = await response.json()
    console.log('HF API result:', result)
    return res.status(200).json(result)

  } catch (error) {
    console.error('Sign prediction error:', error)
    return res.status(500).json({ error: 'Failed to process sign prediction', details: error.message })
  }
}
