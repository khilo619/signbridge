import Pusher from 'pusher'

const pusher = new Pusher({
  appId: process.env.PUSHER_APP_ID,
  key: process.env.PUSHER_KEY,
  secret: process.env.PUSHER_SECRET,
  cluster: process.env.PUSHER_CLUSTER,
  useTLS: true,
})

export default function handler(req, res) {
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

  // Handle both JSON and form-urlencoded
  const socketId = req.body.socket_id
  const channelName = req.body.channel_name

  if (!socketId || !channelName) {
    console.error('Missing socket_id or channel_name:', req.body)
    return res.status(400).json({ error: 'Missing socket_id or channel_name' })
  }

  try {
    const authResponse = pusher.authorizeChannel(socketId, channelName)
    res.status(200).json(authResponse)
  } catch (error) {
    console.error('Pusher auth error:', error)
    res.status(403).json({ error: 'Forbidden' })
  }
}
