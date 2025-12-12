# SignBridge - Sign Language Video Call

A video calling application with **real-time sign language recognition**. Enables communication between deaf/hard-of-hearing users and hearing users by translating sign language to text.

## Live Demo

🔗 [video-call-webrtc.vercel.app](https://video-call-webrtc.vercel.app)

## Features

- **Sign Language Recognition**: Real-time translation of sign language to text using I3D model
- **Role-based UI**: Choose between Deaf/HoH or Hearing user modes
- **Live Transcript**: Signs are translated and displayed as text for both users
- **Video Calling**: Peer-to-peer video/audio streaming with WebRTC
- **Cross-network Support**: Works across different networks with TURN servers
- **Modern UI**: Responsive design with Tailwind CSS

## Setup

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure Pusher (Required for signaling)

1. Go to [pusher.com](https://pusher.com) and create a free account
2. Create a new Channels app
3. Go to App Keys and copy your credentials

### 3. Configure Metered TURN Server (Required for cross-network calls)

1. Go to [metered.ca](https://www.metered.ca/stun-turn) and create a free account
2. Create a new app and generate TURN credentials
3. Update the `ICE_SERVERS` config in `pages/room/[roomId].js` with your credentials

### 4. Environment Variables

Create a `.env.local` file in the project root:

```env
PUSHER_APP_ID=your_app_id
PUSHER_KEY=your_key
PUSHER_SECRET=your_secret
PUSHER_CLUSTER=your_cluster

NEXT_PUBLIC_PUSHER_KEY=your_key
NEXT_PUBLIC_PUSHER_CLUSTER=your_cluster
```

### 5. Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## Deploy to Vercel

1. Push your code to GitHub
2. Go to [vercel.com](https://vercel.com) and import your repository
3. Add the environment variables in Vercel's project settings
4. Deploy!

## How It Works

1. **Select Role**: Choose if you're a Deaf/HoH user (sign language) or Hearing user (voice)
2. **Create a Room**: Click "Create New Room" to generate a unique room code
3. **Share the Link**: Copy the room link and send it to the person you want to call
4. **Connect**: When both users are in the room, the video call starts automatically
5. **Sign & Translate**: Deaf users click the 🤟 button to start signing - translations appear in real-time

## Architecture

- **Sign Language Model**: I3D-based model hosted on HuggingFace Space for sign recognition
- **Signaling**: Pusher Channels handles the WebRTC signaling (offer/answer/ICE candidates exchange)
- **STUN**: Google STUN servers for NAT discovery
- **TURN**: Metered TURN servers for relay when direct connection fails
- **Media**: WebRTC handles peer-to-peer video/audio streaming

## Tech Stack

- **Next.js 14** - React framework
- **WebRTC** - Peer-to-peer video/audio
- **Pusher Channels** - Real-time signaling & translation sync
- **HuggingFace Spaces** - Sign language recognition API
- **Metered** - TURN server for NAT traversal
- **Tailwind CSS** - Styling
- **Vercel** - Deployment

## Limits

- Metered free tier: 500MB/month (~100-150 minutes of video calls through TURN)
- Direct peer-to-peer connections don't count against this limit
- Sign language recognition requires stable internet for API calls
