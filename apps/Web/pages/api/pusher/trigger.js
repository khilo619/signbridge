import Pusher from 'pusher'

const pusher = new Pusher({
  appId: process.env.PUSHER_APP_ID,
  key: process.env.PUSHER_KEY,
  secret: process.env.PUSHER_SECRET,
  cluster: process.env.PUSHER_CLUSTER,
  useTLS: true,
})

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  const { channel, event, data } = req.body

  try {
    await pusher.trigger(channel, event, data)
    res.status(200).json({ success: true })
  } catch (error) {
    console.error('Pusher trigger error:', error)
    res.status(500).json({ error: 'Failed to trigger event' })
  }
}
