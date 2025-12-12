import { useState } from 'react'
import { useRouter } from 'next/router'
import { v4 as uuidv4 } from 'uuid'
import Head from 'next/head'

export default function Home() {
  const [roomId, setRoomId] = useState('')
  const [role, setRole] = useState('') // 'deaf' or 'hearing' - must be selected
  const router = useRouter()

  const createRoom = () => {
    const newRoomId = uuidv4().slice(0, 8)
    router.push(`/room/${newRoomId}?role=${role}`)
  }

  const joinRoom = (e) => {
    e.preventDefault()
    if (roomId.trim()) {
      router.push(`/room/${roomId.trim()}?role=${role}`)
    }
  }

  return (
    <>
      <Head>
        <title>SignBridge - Sign Language Video Call</title>
        <meta name="description" content="Video calling with real-time sign language translation" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <main className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center p-4">
        <div className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 md:p-12 max-w-md w-full shadow-2xl border border-white/20">
          <div className="text-center mb-8">
            <div className="w-20 h-20 bg-gradient-to-br from-purple-500 to-pink-500 rounded-2xl mx-auto mb-6 flex items-center justify-center shadow-lg">
              <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
              </svg>
            </div>
            <h1 className="text-3xl font-bold text-white mb-2">SignBridge</h1>
            <p className="text-gray-300">Video calls with sign language translation</p>
          </div>

          {/* Role Selection */}
          <div className="mb-6">
            <label className="block text-gray-300 text-sm font-medium mb-3">I am:</label>
            <div className="grid grid-cols-2 gap-3">
              <button
                type="button"
                onClick={() => setRole('deaf')}
                className={`p-4 rounded-xl border-2 transition-all ${
                  role === 'deaf'
                    ? 'border-purple-500 bg-purple-500/20 text-white'
                    : 'border-white/20 bg-white/5 text-gray-400 hover:bg-white/10'
                }`}
              >
                <div className="text-2xl mb-1">🤟</div>
                <div className="font-medium">Deaf / HoH</div>
                <div className="text-xs opacity-70">Sign language user</div>
              </button>
              <button
                type="button"
                onClick={() => setRole('hearing')}
                className={`p-4 rounded-xl border-2 transition-all ${
                  role === 'hearing'
                    ? 'border-purple-500 bg-purple-500/20 text-white'
                    : 'border-white/20 bg-white/5 text-gray-400 hover:bg-white/10'
                }`}
              >
                <div className="text-2xl mb-1">👂</div>
                <div className="font-medium">Hearing</div>
                <div className="text-xs opacity-70">Voice user</div>
              </button>
            </div>
          </div>

          <button
            onClick={createRoom}
            disabled={!role}
            className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed text-white font-semibold py-4 px-6 rounded-xl transition-all duration-200 transform hover:scale-[1.02] disabled:hover:scale-100 shadow-lg mb-6"
          >
            {role ? 'Create New Room' : 'Select your role first'}
          </button>

          <div className="relative mb-6">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-white/20"></div>
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-4 bg-transparent text-gray-400">or join existing</span>
            </div>
          </div>

          <form onSubmit={joinRoom} className="space-y-4">
            <input
              type="text"
              value={roomId}
              onChange={(e) => setRoomId(e.target.value)}
              placeholder="Enter room code"
              className="w-full bg-white/10 border border-white/20 rounded-xl py-4 px-5 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all"
            />
            <button
              type="submit"
              disabled={!roomId.trim() || !role}
              className="w-full bg-white/10 hover:bg-white/20 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-4 px-6 rounded-xl transition-all duration-200 border border-white/20"
            >
              {role ? 'Join Room' : 'Select your role first'}
            </button>
          </form>

          <p className="text-center text-gray-500 text-xs mt-6">
            Deaf users&apos; signs are translated to text in real-time
          </p>
        </div>
      </main>
    </>
  )
}
