import { useEffect, useRef, useState, useCallback } from 'react'
import { useRouter } from 'next/router'
import Head from 'next/head'
import Pusher from 'pusher-js'

const ICE_SERVERS = {
  iceServers: [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun.relay.metered.ca:80' },
    {
      urls: 'turn:global.relay.metered.ca:80',
      username: '7077e854820e6a3bc31dd457',
      credential: '191EIP8HkhA/mCxF',
    },
    {
      urls: 'turn:global.relay.metered.ca:80?transport=tcp',
      username: '7077e854820e6a3bc31dd457',
      credential: '191EIP8HkhA/mCxF',
    },
    {
      urls: 'turn:global.relay.metered.ca:443',
      username: '7077e854820e6a3bc31dd457',
      credential: '191EIP8HkhA/mCxF',
    },
    {
      urls: 'turns:global.relay.metered.ca:443?transport=tcp',
      username: '7077e854820e6a3bc31dd457',
      credential: '191EIP8HkhA/mCxF',
    },
  ],
  iceCandidatePoolSize: 10,
}

// Sign language recognition config
const CLIP_FRAMES = 32        // Model trained on 32 frames
const MIN_CONFIDENCE = 0.6    // Minimum confidence to accept prediction
const MIN_MOTION_SCORE = 5.0  // Motion threshold to START capturing
const LOW_MOTION_FRAMES = 2   // Frames of low motion to STOP capturing (reduced for speed)
const MAX_CAPTURE_SECONDS = 3.5 // Max capture duration (32 frames at 10fps = 3.2s)
const REQUEST_INTERVAL = 0.5  // Cooldown between requests (reduced for speed)

export default function Room() {
  const router = useRouter()
  const { roomId, role } = router.query

  const [isConnected, setIsConnected] = useState(false)
  const [isMuted, setIsMuted] = useState(false)
  const [isVideoOff, setIsVideoOff] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState('Connecting...')
  const [copied, setCopied] = useState(false)
  
  // Sign language states
  const [currentSign, setCurrentSign] = useState(null)
  const [signConfidence, setSignConfidence] = useState(0)
  const [transcript, setTranscript] = useState([])
  const [refinedSentence, setRefinedSentence] = useState('')
  const [isCapturing, setIsCapturing] = useState(false)
  const [signStatus, setSignStatus] = useState('idle') // idle, ready, capturing, processing
  const [peerRole, setPeerRole] = useState(null)
  const [motionScore, setMotionScore] = useState(0)
  const [captureFrameCount, setCaptureFrameCount] = useState(0)
  const [cooldownTime, setCooldownTime] = useState(0)
  const [isRefining, setIsRefining] = useState(false)
  
  // Speech-to-text states (for hearing users)
  const [isListening, setIsListening] = useState(false)
  const [speechText, setSpeechText] = useState('')
  const [speechTranscript, setSpeechTranscript] = useState([])
  const recognitionRef = useRef(null)
  const isListeningRef = useRef(false)

  const localVideoRef = useRef(null)
  const remoteVideoRef = useRef(null)
  const peerConnectionRef = useRef(null)
  const localStreamRef = useRef(null)
  const pusherRef = useRef(null)
  const channelRef = useRef(null)
  const userIdRef = useRef(null)
  const canvasRef = useRef(null)
  const motionCanvasRef = useRef(null)
  const frameBufferRef = useRef([])
  const captureIntervalRef = useRef(null)
  const prevGrayRef = useRef(null)
  const lowMotionCountRef = useRef(0)
  const captureStartTimeRef = useRef(0)
  const isProcessingRef = useRef(false)
  const isCapturingRef = useRef(false) // Use ref to avoid closure issues
  const lastRequestTimeRef = useRef(0) // For REQUEST_INTERVAL cooldown

  const triggerEvent = useCallback(async (event, data) => {
    try {
      await fetch('/api/pusher/trigger', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          channel: `private-room-${roomId}`,
          event,
          data: { ...data, senderId: userIdRef.current },
        }),
      })
    } catch (error) {
      console.error('Failed to trigger event:', error)
    }
  }, [roomId])

  const createPeerConnection = useCallback(() => {
    const pc = new RTCPeerConnection(ICE_SERVERS)

    pc.onicecandidate = (event) => {
      if (event.candidate) {
        triggerEvent('ice-candidate', { candidate: event.candidate })
      }
    }

    pc.ontrack = (event) => {
      if (remoteVideoRef.current && event.streams[0]) {
        remoteVideoRef.current.srcObject = event.streams[0]
        setIsConnected(true)
        setConnectionStatus('Connected')
      }
    }

    pc.onconnectionstatechange = () => {
      switch (pc.connectionState) {
        case 'connected':
          setConnectionStatus('Connected')
          setIsConnected(true)
          break
        case 'disconnected':
        case 'failed':
          setConnectionStatus('Disconnected')
          setIsConnected(false)
          break
        case 'connecting':
          setConnectionStatus('Connecting...')
          break
      }
    }

    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach((track) => {
        pc.addTrack(track, localStreamRef.current)
      })
    }

    return pc
  }, [triggerEvent])

  const handleOffer = useCallback(async (offer, senderId) => {
    if (senderId === userIdRef.current) return

    peerConnectionRef.current = createPeerConnection()
    await peerConnectionRef.current.setRemoteDescription(new RTCSessionDescription(offer))
    const answer = await peerConnectionRef.current.createAnswer()
    await peerConnectionRef.current.setLocalDescription(answer)
    triggerEvent('answer', { answer })
  }, [createPeerConnection, triggerEvent])

  const handleAnswer = useCallback(async (answer, senderId) => {
    if (senderId === userIdRef.current) return
    if (peerConnectionRef.current) {
      await peerConnectionRef.current.setRemoteDescription(new RTCSessionDescription(answer))
    }
  }, [])

  const handleIceCandidate = useCallback(async (candidate, senderId) => {
    if (senderId === userIdRef.current) return
    if (peerConnectionRef.current) {
      try {
        await peerConnectionRef.current.addIceCandidate(new RTCIceCandidate(candidate))
      } catch (error) {
        console.error('Error adding ICE candidate:', error)
      }
    }
  }, [])

  const startCall = useCallback(async () => {
    peerConnectionRef.current = createPeerConnection()
    const offer = await peerConnectionRef.current.createOffer()
    await peerConnectionRef.current.setLocalDescription(offer)
    triggerEvent('offer', { offer })
  }, [createPeerConnection, triggerEvent])

  // Compute motion score between current frame and previous (like Python script)
  const computeMotionScore = useCallback(() => {
    if (!localVideoRef.current || !motionCanvasRef.current) return 0
    
    const video = localVideoRef.current
    const canvas = motionCanvasRef.current
    const ctx = canvas.getContext('2d')
    
    // Downscale to 64x64 for motion detection (like Python)
    canvas.width = 64
    canvas.height = 64
    ctx.drawImage(video, 0, 0, 64, 64)
    
    const imageData = ctx.getImageData(0, 0, 64, 64)
    const data = imageData.data
    
    // Convert to grayscale
    const gray = new Uint8Array(64 * 64)
    for (let i = 0; i < gray.length; i++) {
      const idx = i * 4
      gray[i] = Math.round(0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2])
    }
    
    // Compute motion as mean absolute difference
    let motion = 0
    if (prevGrayRef.current) {
      let sum = 0
      for (let i = 0; i < gray.length; i++) {
        sum += Math.abs(gray[i] - prevGrayRef.current[i])
      }
      motion = sum / gray.length
    }
    
    prevGrayRef.current = gray
    return motion
  }, [])

  // Capture frame for sign language recognition
  const captureFrame = useCallback(() => {
    if (!localVideoRef.current || !canvasRef.current) return null
    
    const video = localVideoRef.current
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    
    canvas.width = video.videoWidth || 640
    canvas.height = video.videoHeight || 480
    
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    return canvas.toDataURL('image/jpeg', 0.8)
  }, [])

  // Create video from frames and send for prediction
  const sendForPrediction = useCallback(async (frames) => {
    if (frames.length < 10) {
      console.log('Not enough frames:', frames.length)
      return
    }
    
    setSignStatus('processing')
    console.log('Creating video from', frames.length, 'frames')
    
    try {
      // Create video using canvas and MediaRecorder
      const canvas = canvasRef.current
      if (!canvas) return
      
      const stream = canvas.captureStream(25)
      
      // Check supported mime types - prefer mp4 if available
      let mimeType = 'video/webm'
      if (MediaRecorder.isTypeSupported('video/mp4')) {
        mimeType = 'video/mp4'
      } else if (MediaRecorder.isTypeSupported('video/webm;codecs=vp8')) {
        mimeType = 'video/webm;codecs=vp8'
      }
      
      const recorder = new MediaRecorder(stream, { mimeType })
      const chunks = []
      
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data)
      }
      
      // Create video by drawing frames
      await new Promise((resolve, reject) => {
        recorder.onstop = resolve
        recorder.onerror = reject
        recorder.start()
        
        let frameIndex = 0
        const ctx = canvas.getContext('2d', { willReadFrequently: true })
        
        const drawNextFrame = () => {
          if (frameIndex < frames.length) {
            const img = new Image()
            img.onload = () => {
              ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
              frameIndex++
              setTimeout(drawNextFrame, 40) // ~25fps
            }
            img.onerror = () => {
              frameIndex++
              setTimeout(drawNextFrame, 40)
            }
            img.src = frames[frameIndex]
          } else {
            recorder.stop()
          }
        }
        drawNextFrame()
      })
      
      // Create blob and send as FormData (like Python does)
      const videoBlob = new Blob(chunks, { type: mimeType })
      console.log('Video blob size:', videoBlob.size, 'type:', mimeType)
      
      // Send as FormData to match Python's multipart/form-data
      const formData = new FormData()
      const extension = mimeType.includes('mp4') ? 'mp4' : 'webm'
      formData.append('video', videoBlob, `clip.${extension}`)
      formData.append('top_k', '5')
      
      // Send directly to HuggingFace API (bypass our API route)
      const response = await fetch('https://khalood619-signbridge-api.hf.space/sign/predict', {
        method: 'POST',
        body: formData,
      })
      
      const result = await response.json()
      console.log('Prediction result:', result)
      
      if (response.ok && result.gloss) {
        console.log('Sign detected:', result.gloss, 'prob:', result.probability, 'topk:', result.topk)
        
        // Always show the current prediction
        setCurrentSign(result.gloss)
        setSignConfidence(result.probability)
        
        // Only add to transcript if confidence is high enough
        if (result.probability >= MIN_CONFIDENCE) {
          setTranscript(prev => {
            if (prev[prev.length - 1] !== result.gloss) {
              return [...prev, result.gloss]
            }
            return prev
          })
          // Send translation to peer
          triggerEvent('sign-translation', { 
            gloss: result.gloss, 
            probability: result.probability 
          })
        }
      } else if (result.error || result.detail) {
        console.error('API error:', result.error || result.detail)
      }
    } catch (error) {
      console.error('Sign prediction error:', error)
    }
  }, [triggerEvent])

  // Refine transcript into proper sentence using LLM
  const refineTranscript = useCallback(async () => {
    if (transcript.length < 2 || isRefining) return
    
    setIsRefining(true)
    try {
      const response = await fetch('/api/sign/refine', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ glosses: transcript })
      })
      
      const result = await response.json()
      if (result.refined) {
        setRefinedSentence(result.refined)
        // Send refined sentence to peer
        triggerEvent('refined-sentence', { sentence: result.refined })
      }
    } catch (error) {
      console.error('Refine error:', error)
    } finally {
      setIsRefining(false)
    }
  }, [transcript, isRefining, triggerEvent])

  // Speech-to-text for hearing users
  const startSpeechRecognition = useCallback(() => {
    if (role !== 'hearing') return
    
    // Check browser support
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SpeechRecognition) {
      alert('Speech recognition is not supported in this browser. Please use Chrome.')
      return
    }
    
    // Stop any existing recognition
    if (recognitionRef.current) {
      recognitionRef.current.stop()
    }
    
    const recognition = new SpeechRecognition()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = 'en-US'
    
    recognition.onstart = () => {
      console.log('Speech recognition started')
      isListeningRef.current = true
      setIsListening(true)
    }
    
    recognition.onresult = (event) => {
      let interimTranscript = ''
      let finalTranscript = ''
      
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const text = event.results[i][0].transcript
        if (event.results[i].isFinal) {
          finalTranscript += text
        } else {
          interimTranscript += text
        }
      }
      
      console.log('Speech result:', { interim: interimTranscript, final: finalTranscript })
      
      // Show interim results
      setSpeechText(interimTranscript || finalTranscript)
      
      // When we have final results, add to transcript and send to peer
      if (finalTranscript) {
        const trimmed = finalTranscript.trim()
        if (trimmed) {
          setSpeechTranscript(prev => [...prev, trimmed])
          setSpeechText('')
          // Send to deaf peer
          triggerEvent('speech-text', { text: trimmed })
        }
      }
    }
    
    recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error)
      // Don't stop on no-speech or aborted errors
      if (event.error !== 'no-speech' && event.error !== 'aborted') {
        isListeningRef.current = false
        setIsListening(false)
      }
    }
    
    recognition.onend = () => {
      console.log('Speech recognition ended, isListening:', isListeningRef.current)
      // Restart if still supposed to be listening
      if (isListeningRef.current) {
        try {
          recognition.start()
        } catch (e) {
          console.error('Failed to restart recognition:', e)
        }
      } else {
        setIsListening(false)
      }
    }
    
    recognitionRef.current = recognition
    
    try {
      recognition.start()
    } catch (e) {
      console.error('Failed to start recognition:', e)
      alert('Failed to start speech recognition. Make sure microphone is allowed.')
    }
  }, [role, triggerEvent])

  const stopSpeechRecognition = useCallback(() => {
    console.log('Stopping speech recognition')
    isListeningRef.current = false
    setIsListening(false)
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop()
      } catch (e) {
        console.error('Error stopping recognition:', e)
      }
      recognitionRef.current = null
    }
  }, [])

  // Motion-based sign language capture (like Python script exactly)
  const startMotionDetection = useCallback(() => {
    if (role !== 'deaf' || captureIntervalRef.current) return
    
    setSignStatus('ready')
    isCapturingRef.current = false
    isProcessingRef.current = false
    frameBufferRef.current = []
    prevGrayRef.current = null
    lastRequestTimeRef.current = 0
    
    captureIntervalRef.current = setInterval(() => {
      const motion = computeMotionScore()
      setMotionScore(motion)
      
      const now = Date.now()
      const timeSinceLast = (now - lastRequestTimeRef.current) / 1000
      const timeToNext = Math.max(0, REQUEST_INTERVAL - timeSinceLast)
      setCooldownTime(timeToNext)
      
      // Update frame count display
      setCaptureFrameCount(frameBufferRef.current.length)
      
      // Can we start capturing? (like Python: not capturing, not inflight, cooldown done, motion detected)
      const canStartCapture = (
        !isCapturingRef.current &&
        !isProcessingRef.current &&
        timeToNext <= 0 &&
        motion >= MIN_MOTION_SCORE
      )
      
      if (canStartCapture) {
        // Motion detected! Start capturing
        console.log('Motion detected! Starting capture. Motion:', motion.toFixed(1))
        isCapturingRef.current = true
        setIsCapturing(true)
        setSignStatus('capturing')
        frameBufferRef.current = []
        lowMotionCountRef.current = 0
        captureStartTimeRef.current = now
      }
      
      if (isCapturingRef.current) {
        // CAPTURING state - record frames (keep only last CLIP_FRAMES like Python deque)
        const frame = captureFrame()
        if (frame) {
          frameBufferRef.current.push(frame)
          if (frameBufferRef.current.length > CLIP_FRAMES) {
            frameBufferRef.current.shift()
          }
        }
        
        // Track low motion to detect end of sign
        if (motion < MIN_MOTION_SCORE) {
          lowMotionCountRef.current++
        } else {
          lowMotionCountRef.current = 0
        }
        
        // Check if should finish capturing (like Python)
        const enoughFrames = frameBufferRef.current.length >= CLIP_FRAMES
        const stableEnough = lowMotionCountRef.current >= LOW_MOTION_FRAMES
        const timeLong = (now - captureStartTimeRef.current) >= MAX_CAPTURE_SECONDS * 1000
        const shouldFinish = (enoughFrames && stableEnough) || timeLong
        
        if (shouldFinish && frameBufferRef.current.length > 0) {
          console.log('Finishing capture. Frames:', frameBufferRef.current.length, 'LowMotion:', lowMotionCountRef.current)
          
          const frames = [...frameBufferRef.current]
          isCapturingRef.current = false
          setIsCapturing(false)
          lowMotionCountRef.current = 0
          
          if (frames.length >= 10 && !isProcessingRef.current) {
            isProcessingRef.current = true
            lastRequestTimeRef.current = now
            setSignStatus('processing')
            frameBufferRef.current = []
            
            sendForPrediction(frames).finally(() => {
              isProcessingRef.current = false
              setSignStatus('ready')
            })
          } else {
            setSignStatus('ready')
          }
        }
      }
    }, 100) // ~10fps motion detection
  }, [role, computeMotionScore, captureFrame, sendForPrediction])

  const stopMotionDetection = useCallback(() => {
    isCapturingRef.current = false
    isProcessingRef.current = false
    setIsCapturing(false)
    setSignStatus('idle')
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current)
      captureIntervalRef.current = null
    }
    frameBufferRef.current = []
    prevGrayRef.current = null
    lowMotionCountRef.current = 0
  }, [])

  useEffect(() => {
    if (!roomId) return

    userIdRef.current = Math.random().toString(36).substring(7)

    const initMedia = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: true,
        })
        localStreamRef.current = stream
        if (localVideoRef.current) {
          localVideoRef.current.srcObject = stream
        }
        setConnectionStatus('Waiting for peer...')
      } catch (error) {
        console.error('Error accessing media devices:', error)
        setConnectionStatus('Camera/mic access denied')
      }
    }

    const initPusher = () => {
      pusherRef.current = new Pusher(process.env.NEXT_PUBLIC_PUSHER_KEY, {
        cluster: process.env.NEXT_PUBLIC_PUSHER_CLUSTER,
        authEndpoint: '/api/pusher/auth',
      })

      channelRef.current = pusherRef.current.subscribe(`private-room-${roomId}`)

      channelRef.current.bind('pusher:subscription_succeeded', () => {
        triggerEvent('user-joined', { role })
      })

      channelRef.current.bind('user-joined', (data) => {
        if (data.senderId !== userIdRef.current) {
          setPeerRole(data.role)
          startCall()
        }
      })

      channelRef.current.bind('offer', (data) => {
        handleOffer(data.offer, data.senderId)
      })

      channelRef.current.bind('answer', (data) => {
        handleAnswer(data.answer, data.senderId)
      })

      channelRef.current.bind('ice-candidate', (data) => {
        handleIceCandidate(data.candidate, data.senderId)
      })

      // Handle sign language translations from deaf user
      channelRef.current.bind('sign-translation', (data) => {
        if (data.senderId !== userIdRef.current) {
          setCurrentSign(data.gloss)
          setSignConfidence(data.probability)
          setTranscript(prev => {
            if (prev[prev.length - 1] !== data.gloss) {
              return [...prev, data.gloss]
            }
            return prev
          })
        }
      })

      // Handle refined sentences from deaf user
      channelRef.current.bind('refined-sentence', (data) => {
        if (data.senderId !== userIdRef.current) {
          setRefinedSentence(data.sentence)
        }
      })

      // Handle speech text from hearing user
      channelRef.current.bind('speech-text', (data) => {
        if (data.senderId !== userIdRef.current) {
          setSpeechTranscript(prev => [...prev, data.text])
        }
      })
    }

    initMedia().then(initPusher)

    return () => {
      if (localStreamRef.current) {
        localStreamRef.current.getTracks().forEach((track) => track.stop())
      }
      if (peerConnectionRef.current) {
        peerConnectionRef.current.close()
      }
      if (channelRef.current) {
        channelRef.current.unbind_all()
        channelRef.current.unsubscribe()
      }
      if (pusherRef.current) {
        pusherRef.current.disconnect()
      }
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current)
      }
      if (recognitionRef.current) {
        recognitionRef.current.stop()
      }
    }
  }, [roomId, role, handleOffer, handleAnswer, handleIceCandidate, startCall, triggerEvent])

  const toggleMute = () => {
    if (localStreamRef.current) {
      const audioTrack = localStreamRef.current.getAudioTracks()[0]
      if (audioTrack) {
        audioTrack.enabled = !audioTrack.enabled
        setIsMuted(!audioTrack.enabled)
      }
    }
  }

  const toggleVideo = () => {
    if (localStreamRef.current) {
      const videoTrack = localStreamRef.current.getVideoTracks()[0]
      if (videoTrack) {
        videoTrack.enabled = !videoTrack.enabled
        setIsVideoOff(!videoTrack.enabled)
      }
    }
  }

  const endCall = () => {
    router.push('/')
  }

  const copyRoomLink = () => {
    const url = window.location.href
    navigator.clipboard.writeText(url)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <>
      <Head>
        <title>Room {roomId} | SignBridge</title>
      </Head>

      {/* Hidden canvases for frame capture and motion detection */}
      <canvas ref={canvasRef} className="hidden" />
      <canvas ref={motionCanvasRef} className="hidden" />

      <main className="min-h-screen bg-slate-900 flex flex-col">
        {/* Header */}
        <header className="bg-slate-800/50 backdrop-blur border-b border-white/10 px-4 py-3">
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center">
                <span className="text-xl">🤟</span>
              </div>
              <div>
                <h1 className="text-white font-semibold">SignBridge - Room: {roomId}</h1>
                <div className="flex items-center gap-2">
                  <p className={`text-sm ${isConnected ? 'text-green-400' : 'text-yellow-400'}`}>
                    {connectionStatus}
                  </p>
                  <span className="text-slate-500">•</span>
                  <span className="text-sm text-purple-400">
                    {role === 'deaf' ? '🤟 Deaf' : '👂 Hearing'}
                  </span>
                </div>
              </div>
            </div>
            <button
              onClick={copyRoomLink}
              className="flex items-center gap-2 bg-white/10 hover:bg-white/20 text-white px-4 py-2 rounded-lg transition-all text-sm"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              {copied ? 'Copied!' : 'Copy Link'}
            </button>
          </div>
        </header>

        {/* Video Grid */}
        <div className="flex-1 p-4 md:p-6">
          <div className="max-w-7xl mx-auto h-full grid grid-cols-1 lg:grid-cols-3 gap-4 md:gap-6">
            {/* Local Video */}
            <div className="relative bg-slate-800 rounded-2xl overflow-hidden aspect-video">
              <video
                ref={localVideoRef}
                autoPlay
                playsInline
                muted
                className={`w-full h-full object-cover ${isVideoOff ? 'hidden' : ''}`}
              />
              {isVideoOff && (
                <div className="absolute inset-0 flex items-center justify-center bg-slate-700">
                  <div className="w-20 h-20 bg-slate-600 rounded-full flex items-center justify-center">
                    <svg className="w-10 h-10 text-slate-400" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" />
                    </svg>
                  </div>
                </div>
              )}
              <div className="absolute bottom-4 left-4 bg-black/50 backdrop-blur px-3 py-1.5 rounded-lg">
                <span className="text-white text-sm font-medium">
                  You {role === 'deaf' ? '🤟' : '👂'}
                </span>
              </div>
              {/* Sign capture status for deaf users */}
              {role === 'deaf' && signStatus !== 'idle' && (
                <div className={`absolute top-4 right-4 backdrop-blur px-3 py-1.5 rounded-lg ${
                  signStatus === 'capturing' ? 'bg-red-500/80 animate-pulse' :
                  signStatus === 'processing' ? 'bg-yellow-500/80' :
                  'bg-green-500/80'
                }`}>
                  <span className="text-white text-sm font-medium">
                    {signStatus === 'capturing' ? '● Recording...' :
                     signStatus === 'processing' ? '⏳ Processing...' :
                     '✓ Ready - perform sign'}
                  </span>
                </div>
              )}
              {/* Motion indicator */}
              {role === 'deaf' && signStatus !== 'idle' && (
                <div className="absolute top-4 left-4 bg-black/50 backdrop-blur px-2 py-1 rounded text-xs text-white">
                  Motion: {motionScore.toFixed(1)}
                </div>
              )}
            </div>

            {/* Remote Video */}
            <div className="relative bg-slate-800 rounded-2xl overflow-hidden aspect-video">
              <video
                ref={remoteVideoRef}
                autoPlay
                playsInline
                className="w-full h-full object-cover"
              />
              {!isConnected && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center">
                    <div className="w-20 h-20 bg-slate-700 rounded-full flex items-center justify-center mx-auto mb-4">
                      <svg className="w-10 h-10 text-slate-500" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" />
                      </svg>
                    </div>
                    <p className="text-slate-400">Waiting for peer to join...</p>
                    <p className="text-slate-500 text-sm mt-2">Share the room link to connect</p>
                  </div>
                </div>
              )}
              {isConnected && (
                <div className="absolute bottom-4 left-4 bg-black/50 backdrop-blur px-3 py-1.5 rounded-lg">
                  <span className="text-white text-sm font-medium">
                    Peer {peerRole === 'deaf' ? '🤟' : '👂'}
                  </span>
                </div>
              )}
            </div>

            {/* Speech Panel (for deaf users to read what hearing user says) */}
            {(role === 'deaf' || peerRole === 'hearing') && (
              <div className="bg-slate-800 rounded-2xl p-4 flex flex-col mb-4">
                <h3 className="text-white font-semibold mb-3 flex items-center gap-2">
                  <span>🎤</span> Speech to Text
                  {isListening && <span className="text-xs text-red-400 animate-pulse">● LIVE</span>}
                </h3>
                
                {/* Current speech (interim) */}
                {speechText && (
                  <div className="bg-blue-900/30 border border-blue-500/30 rounded-xl p-3 mb-3">
                    <p className="text-blue-300 italic">{speechText}...</p>
                  </div>
                )}
                
                {/* Speech transcript */}
                <div className="flex-1 bg-slate-700/50 rounded-xl p-4 overflow-hidden">
                  <div className="h-24 overflow-y-auto">
                    {speechTranscript.length > 0 ? (
                      <p className="text-white leading-relaxed">
                        {speechTranscript.join(' ')}
                      </p>
                    ) : (
                      <p className="text-slate-500">
                        {role === 'hearing' ? 'Click 🎤 to start speaking...' : 'Waiting for hearing user to speak...'}
                      </p>
                    )}
                  </div>
                </div>
                
                {speechTranscript.length > 0 && (
                  <button
                    onClick={() => setSpeechTranscript([])}
                    className="mt-2 text-slate-400 hover:text-white text-sm transition-colors"
                  >
                    Clear
                  </button>
                )}
              </div>
            )}

            {/* Translation Panel */}
            <div className="bg-slate-800 rounded-2xl p-4 flex flex-col">
              <h3 className="text-white font-semibold mb-3 flex items-center gap-2">
                <span>🔤</span> Sign Translation
              </h3>
              
              {/* Status (like Python overlay) */}
              <div className={`rounded-xl p-3 mb-3 ${
                signStatus === 'idle' ? 'bg-slate-700/50' :
                signStatus === 'ready' ? 'bg-green-900/30 border border-green-500/30' :
                signStatus === 'capturing' ? 'bg-red-900/30 border border-red-500/30' :
                'bg-yellow-900/30 border border-yellow-500/30'
              }`}>
                <p className="text-xs uppercase tracking-wide mb-1 text-slate-400">Status</p>
                <p className={`font-medium ${
                  signStatus === 'idle' ? 'text-slate-400' :
                  signStatus === 'ready' ? 'text-green-400' :
                  signStatus === 'capturing' ? 'text-red-400' :
                  'text-yellow-400'
                }`}>
                  {signStatus === 'idle' ? '⏸ Click 🤟 to start' :
                   signStatus === 'ready' ? (cooldownTime > 0 ? `waiting (${cooldownTime.toFixed(1)}s)` : '✓ READY - perform sign') :
                   signStatus === 'capturing' ? '● CAPTURING - keep signing' :
                   '⏳ SENDING clip to API...'}
                </p>
                {signStatus !== 'idle' && (
                  <div className="text-slate-500 text-xs mt-1 space-y-0.5">
                    <p>Motion: {motionScore.toFixed(1)} {motionScore >= MIN_MOTION_SCORE ? '🟢' : '⚪'}</p>
                    {signStatus === 'capturing' && (
                      <p>Capturing: {captureFrameCount}/{CLIP_FRAMES} frames</p>
                    )}
                  </div>
                )}
              </div>

              {/* Current Sign */}
              <div className="bg-slate-700/50 rounded-xl p-4 mb-4">
                <p className="text-slate-400 text-xs uppercase tracking-wide mb-1">Current Sign</p>
                {currentSign ? (
                  <div>
                    <p className="text-3xl font-bold text-white">{currentSign}</p>
                    <p className="text-green-400 text-sm mt-1">
                      {(signConfidence * 100).toFixed(0)}% confidence
                    </p>
                  </div>
                ) : (
                  <p className="text-slate-500 text-lg">Waiting for signs...</p>
                )}
              </div>

              {/* Refined Sentence */}
              {refinedSentence && (
                <div className="bg-green-900/30 border border-green-500/30 rounded-xl p-4 mb-3">
                  <p className="text-green-400 text-xs uppercase tracking-wide mb-1">✨ Refined Sentence</p>
                  <p className="text-white text-lg font-medium">{refinedSentence}</p>
                </div>
              )}

              {/* Transcript */}
              <div className="flex-1 bg-slate-700/50 rounded-xl p-4 overflow-hidden">
                <p className="text-slate-400 text-xs uppercase tracking-wide mb-2">Raw Glosses</p>
                <div className="h-24 overflow-y-auto">
                  {transcript.length > 0 ? (
                    <p className="text-white leading-relaxed">
                      {transcript.join(' ')}
                    </p>
                  ) : (
                    <p className="text-slate-500">Signs will appear here...</p>
                  )}
                </div>
              </div>

              {/* Action buttons */}
              {transcript.length >= 2 && (
                <div className="mt-3 flex gap-2">
                  <button
                    onClick={refineTranscript}
                    disabled={isRefining}
                    className="flex-1 bg-purple-600 hover:bg-purple-700 disabled:bg-purple-800 disabled:cursor-wait text-white text-sm py-2 px-3 rounded-lg transition-colors"
                  >
                    {isRefining ? '✨ Refining...' : '✨ Make Sentence'}
                  </button>
                  <button
                    onClick={() => { setTranscript([]); setRefinedSentence('') }}
                    className="text-slate-400 hover:text-white text-sm py-2 px-3 transition-colors"
                  >
                    Clear
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="bg-slate-800/50 backdrop-blur border-t border-white/10 px-4 py-4">
          <div className="max-w-7xl mx-auto flex items-center justify-center gap-4">
            {/* Sign detection toggle for deaf users */}
            {role === 'deaf' && (
              <button
                onClick={signStatus === 'idle' ? startMotionDetection : stopMotionDetection}
                className={`w-14 h-14 rounded-full flex items-center justify-center transition-all ${
                  signStatus === 'idle' ? 'bg-purple-600 hover:bg-purple-700' :
                  signStatus === 'capturing' ? 'bg-red-500 hover:bg-red-600 animate-pulse' :
                  'bg-green-500 hover:bg-green-600'
                }`}
                title={signStatus === 'idle' ? 'Start sign detection' : 'Stop sign detection'}
              >
                <span className="text-2xl">🤟</span>
              </button>
            )}

            {/* Speech-to-text toggle for hearing users */}
            {role === 'hearing' && (
              <button
                onClick={isListening ? stopSpeechRecognition : startSpeechRecognition}
                className={`w-14 h-14 rounded-full flex items-center justify-center transition-all ${
                  isListening ? 'bg-red-500 hover:bg-red-600 animate-pulse' : 'bg-blue-600 hover:bg-blue-700'
                }`}
                title={isListening ? 'Stop speech-to-text' : 'Start speech-to-text'}
              >
                <span className="text-2xl">{isListening ? '🎙️' : '🎤'}</span>
              </button>
            )}

            <button
              onClick={toggleMute}
              className={`w-14 h-14 rounded-full flex items-center justify-center transition-all ${
                isMuted ? 'bg-red-500 hover:bg-red-600' : 'bg-slate-700 hover:bg-slate-600'
              }`}
            >
              {isMuted ? (
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2" />
                </svg>
              ) : (
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                </svg>
              )}
            </button>

            <button
              onClick={toggleVideo}
              className={`w-14 h-14 rounded-full flex items-center justify-center transition-all ${
                isVideoOff ? 'bg-red-500 hover:bg-red-600' : 'bg-slate-700 hover:bg-slate-600'
              }`}
            >
              {isVideoOff ? (
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 3l18 18" />
                </svg>
              ) : (
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
              )}
            </button>

            <button
              onClick={endCall}
              className="w-14 h-14 bg-red-500 hover:bg-red-600 rounded-full flex items-center justify-center transition-all"
            >
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 8l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2M5 3a2 2 0 00-2 2v1c0 8.284 6.716 15 15 15h1a2 2 0 002-2v-3.28a1 1 0 00-.684-.948l-4.493-1.498a1 1 0 00-1.21.502l-1.13 2.257a11.042 11.042 0 01-5.516-5.517l2.257-1.128a1 1 0 00.502-1.21L9.228 3.683A1 1 0 008.279 3H5z" />
              </svg>
            </button>
          </div>
        </div>
      </main>
    </>
  )
}
