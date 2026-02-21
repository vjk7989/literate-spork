import React, { useState } from 'react';
import { useLiveAPI } from '../hooks/useLiveAPI';
import { Mic, MicOff, Volume2, Shield, Info, AlertCircle, Send } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { VoiceVisualizer } from './VoiceVisualizer';

export const LucaBot: React.FC = () => {
  const { isConnected, isAITalking, isUserTalking, volume, messages, error, connect, disconnect, sendTextMessage } = useLiveAPI();
  const [isHovering, setIsHovering] = useState(false);
  const [inputText, setInputText] = useState('');
  const scrollRef = React.useRef<HTMLDivElement>(null);

  const isTalking = isAITalking || isUserTalking;

  React.useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleToggle = () => {
    if (isConnected) {
      disconnect();
    } else {
      connect();
    }
  };

  const handleSendText = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (inputText.trim() && isConnected) {
      sendTextMessage(inputText.trim());
      setInputText('');
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white flex flex-col items-center justify-center p-6 font-sans">
      {/* Background Atmosphere */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div 
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full opacity-20 blur-[120px] transition-colors duration-1000"
          style={{ 
            background: isConnected 
              ? (isAITalking ? 'radial-gradient(circle, #ffffff 0%, transparent 70%)' : (isUserTalking ? 'radial-gradient(circle, #bf00ff 0%, transparent 70%)' : 'radial-gradient(circle, #3b82f6 0%, transparent 70%)'))
              : 'radial-gradient(circle, #4b5563 0%, transparent 70%)'
          }}
        />
      </div>

      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative z-10 w-full max-w-lg flex flex-col items-center"
      >
        {/* Header */}
        <div className="mb-8 text-center">
          <motion.h1 
            className="text-5xl font-bold tracking-tighter mb-2 bg-clip-text text-transparent bg-gradient-to-b from-white to-white/40"
            animate={{ scale: isTalking ? [1, 1.02, 1] : 1 }}
            transition={{ repeat: Infinity, duration: 2 }}
          >
            LUCA
          </motion.h1>
          <p className="text-white/40 uppercase tracking-[0.2em] text-xs font-semibold">
            Voice Assistant
          </p>
        </div>

        {/* Visualizer Area */}
        <div className="w-full mb-8">
          <VoiceVisualizer isUserTalking={isUserTalking} isAITalking={isAITalking} volume={volume} />
        </div>

        {/* Main Visualizer / Button */}
        <div className="relative group mb-12">
          <AnimatePresence>
            {isConnected && (
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ 
                  scale: isTalking ? [1, 1.2, 1] : 1,
                  opacity: 1,
                  rotate: 360
                }}
                exit={{ scale: 0.8, opacity: 0 }}
                transition={{ 
                  scale: { repeat: Infinity, duration: 1.5 },
                  rotate: { repeat: Infinity, duration: 10, ease: "linear" }
                }}
                className="absolute -inset-8 rounded-full border border-white/5 border-dashed"
              />
            )}
          </AnimatePresence>

          <motion.button
            onClick={handleToggle}
            onMouseEnter={() => setIsHovering(true)}
            onMouseLeave={() => setIsHovering(false)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className={`
              relative z-20 w-32 h-32 rounded-full flex items-center justify-center transition-all duration-500
              ${isConnected 
                ? 'bg-white text-black shadow-[0_0_50px_rgba(255,255,255,0.2)]' 
                : 'bg-white/5 border border-white/10 hover:border-white/20'}
            `}
          >
            <div className="flex flex-col items-center gap-3">
              {isConnected ? (
                <>
                  {isAITalking ? (
                    <Volume2 className="w-6 h-6 animate-pulse" />
                  ) : (
                    <Mic className={`w-6 h-6 ${isUserTalking ? 'text-purple-600' : ''}`} />
                  )}
                  <span className="text-[9px] uppercase tracking-widest font-bold">
                    {isAITalking ? 'Speaking' : (isUserTalking ? 'User' : 'Listen')}
                  </span>
                </>
              ) : (
                <>
                  <MicOff className="w-6 h-6 text-white/40" />
                  <span className="text-[9px] uppercase tracking-widest font-bold text-white/40">
                    Connect
                  </span>
                </>
              )}
            </div>
          </motion.button>
        </div>

        {/* Transcript Area */}
        <div className="w-full h-48 mb-6 overflow-hidden relative rounded-2xl bg-white/5 border border-white/10">
          <div 
            ref={scrollRef}
            className="absolute inset-0 p-4 overflow-y-auto scroll-smooth space-y-4 mask-fade-edges"
          >
            {messages.length === 0 ? (
              <div className="h-full flex items-center justify-center text-white/20 text-xs uppercase tracking-widest italic">
                No conversation yet
              </div>
            ) : (
              messages.map((msg, i) => (
                <motion.div 
                  key={i}
                  initial={{ opacity: 0, x: msg.role === 'user' ? 10 : -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}
                >
                  <span className="text-[9px] uppercase tracking-wider text-white/30 mb-1 font-bold">
                    {msg.role === 'user' ? 'You' : 'LUCA'}
                  </span>
                  <div className={`
                    max-w-[85%] p-3 rounded-2xl text-sm leading-relaxed
                    ${msg.role === 'user' 
                      ? 'bg-white/10 text-white rounded-tr-none' 
                      : 'bg-white text-black rounded-tl-none font-medium'}
                  `}>
                    {msg.text}
                  </div>
                </motion.div>
              ))
            )}
          </div>
        </div>

        {/* Text Input Area */}
        <AnimatePresence>
          {isConnected && (
            <motion.form 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              onSubmit={handleSendText}
              className="w-full mb-8 flex gap-2"
            >
              <input 
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Type a message..."
                className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-2 text-sm focus:outline-none focus:border-white/30 transition-colors placeholder:text-white/20"
              />
              <button 
                type="submit"
                disabled={!inputText.trim()}
                className="p-2 bg-white text-black rounded-xl disabled:opacity-50 disabled:cursor-not-allowed hover:scale-105 active:scale-95 transition-all"
              >
                <Send className="w-4 h-4" />
              </button>
            </motion.form>
          )}
        </AnimatePresence>

        {/* Status & Info */}
        <div className="w-full space-y-4">
          <AnimatePresence mode="wait">
            {error ? (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="flex items-center gap-3 p-4 rounded-2xl bg-red-500/10 border border-red-500/20 text-red-400 text-sm"
              >
                <AlertCircle className="w-4 h-4 shrink-0" />
                <p>{error}</p>
              </motion.div>
            ) : (
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="grid grid-cols-2 gap-4"
              >
                <div className="p-4 rounded-2xl bg-white/5 border border-white/10 flex flex-col gap-1">
                  <div className="flex items-center gap-2 text-white/40 mb-1">
                    <Shield className="w-3 h-3" />
                    <span className="text-[10px] uppercase tracking-wider font-bold">Identity</span>
                  </div>
                  <p className="text-sm font-medium">LUCA Assistant</p>
                </div>
                <div className="p-4 rounded-2xl bg-white/5 border border-white/10 flex flex-col gap-1">
                  <div className="flex items-center gap-2 text-white/40 mb-1">
                    <Info className="w-3 h-3" />
                    <span className="text-[10px] uppercase tracking-wider font-bold">Origin</span>
                  </div>
                  <p className="text-sm font-medium">10x Technologies</p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Footer Hint */}
        <p className="mt-8 text-white/20 text-[10px] uppercase tracking-[0.3em] font-medium">
          Powered by Buckleson
        </p>
      </motion.div>
    </div>
  );
};
