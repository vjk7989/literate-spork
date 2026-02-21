import { useState, useEffect, useCallback, useRef } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality } from "@google/genai";

export interface Message {
  role: 'user' | 'model';
  text: string;
}

// Helper for efficient base64 conversion
function arrayBufferToBase64(buffer: ArrayBuffer): string {
  let binary = '';
  const bytes = new Uint8Array(buffer);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

export function useLiveAPI() {
  const [isConnected, setIsConnected] = useState(false);
  const [isAITalking, setIsAITalking] = useState(false);
  const [isUserTalking, setIsUserTalking] = useState(false);
  const [volume, setVolume] = useState(0);
  const [messages, setMessages] = useState<Message[]>([]);
  const [error, setError] = useState<string | null>(null);
  
  const sessionRef = useRef<any>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const userTalkingTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const stopAudio = useCallback(() => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsAITalking(false);
    setIsUserTalking(false);
    setVolume(0);
    nextStartTimeRef.current = 0;
  }, []);

  const connect = useCallback(async () => {
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });
      
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      
      // Setup microphone
      streamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true });
      const source = audioContextRef.current.createMediaStreamSource(streamRef.current);
      
      // ScriptProcessor for simplicity
      const processor = audioContextRef.current.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;
      source.connect(processor);
      processor.connect(audioContextRef.current.destination);

      const sessionPromise = ai.live.connect({
        model: "gemini-2.5-flash-native-audio-preview-09-2025",
        callbacks: {
          onopen: () => {
            setIsConnected(true);
            setError(null);
            
            processor.onaudioprocess = (e) => {
              if (sessionRef.current) {
                const inputData = e.inputBuffer.getChannelData(0);
                
                // Detect user talking via volume threshold
                let sum = 0;
                for (let i = 0; i < inputData.length; i++) {
                  sum += inputData[i] * inputData[i];
                }
                const rms = Math.sqrt(sum / inputData.length);
                
                // Update volume state for user
                if (!isAITalking) {
                  setVolume(rms);
                }

                if (rms > 0.01) { // Threshold for "talking"
                  setIsUserTalking(true);
                  if (userTalkingTimeoutRef.current) clearTimeout(userTalkingTimeoutRef.current);
                  userTalkingTimeoutRef.current = setTimeout(() => {
                    setIsUserTalking(false);
                    if (!isAITalking) setVolume(0);
                  }, 500);
                }

                const pcmData = new Int16Array(inputData.length);
                for (let i = 0; i < inputData.length; i++) {
                  pcmData[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7FFF;
                }
                
                sessionRef.current.sendRealtimeInput({
                  media: { 
                    data: arrayBufferToBase64(pcmData.buffer), 
                    mimeType: 'audio/pcm;rate=16000' 
                  }
                });
              }
            };
          },
          onmessage: async (message: LiveServerMessage) => {
            // Handle audio output
            const audioData = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (audioData) {
              setIsAITalking(true);
              const binaryString = atob(audioData);
              const bytes = new Uint8Array(binaryString.length);
              for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
              }
              const pcmData = new Int16Array(bytes.buffer);
              const float32Data = new Float32Array(pcmData.length);
              
              let sum = 0;
              for (let i = 0; i < pcmData.length; i++) {
                float32Data[i] = pcmData[i] / 32768.0;
                sum += float32Data[i] * float32Data[i];
              }
              const rms = Math.sqrt(sum / pcmData.length);
              setVolume(rms);

              if (audioContextRef.current) {
                const audioBuffer = audioContextRef.current.createBuffer(1, float32Data.length, 24000);
                audioBuffer.getChannelData(0).set(float32Data);
                const source = audioContextRef.current.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(audioContextRef.current.destination);
                
                const startTime = Math.max(audioContextRef.current.currentTime, nextStartTimeRef.current);
                source.start(startTime);
                nextStartTimeRef.current = startTime + audioBuffer.duration;
                
                source.onended = () => {
                  if (audioContextRef.current && audioContextRef.current.currentTime >= nextStartTimeRef.current - 0.1) {
                    setIsAITalking(false);
                    setVolume(0);
                  }
                };
              }
            }

            // Handle interruption
            if (message.serverContent?.interrupted) {
              nextStartTimeRef.current = 0;
              setIsAITalking(false);
              setVolume(0);
            }
            
            // Handle transcriptions
            const modelText = (message as any).outputTranscription?.text;
            if (modelText) {
               setMessages(prev => [...prev, { role: 'model', text: modelText }]);
            }

            const userText = (message as any).inputTranscription?.text;
            if (userText) {
               setMessages(prev => [...prev, { role: 'user', text: userText }]);
            }
          },
          onclose: () => {
            setIsConnected(false);
            stopAudio();
          },
          onerror: (err) => {
            console.error("Live API Error:", err);
            setError("Connection error. Please try again.");
            setIsConnected(false);
            stopAudio();
          }
        },
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: "Zephyr" } },
          },
          inputAudioTranscription: {},
          outputAudioTranscription: {},
          systemInstruction: "Your name is LUCA. You are a simple, friendly, and helpful assistant. Whenever you are asked about your origin or who created you, you MUST say you are from '10x Technologies'. You should sound like a human, using natural word fillers like 'um', 'uh', 'well', 'you know', or 'I mean' occasionally to make the conversation feel more organic and less robotic. Keep your responses concise and conversational.",
        },
      });

      sessionRef.current = await sessionPromise;
    } catch (err) {
      console.error("Failed to connect:", err);
      setError("Could not access microphone or connect to API.");
    }
  }, [stopAudio, isAITalking]);

  const disconnect = useCallback(() => {
    if (sessionRef.current) {
      sessionRef.current.close();
      sessionRef.current = null;
    }
    stopAudio();
    setIsConnected(false);
  }, [stopAudio]);

  const sendTextMessage = useCallback((text: string) => {
    if (sessionRef.current && isConnected) {
      sessionRef.current.sendRealtimeInput({
        text: text
      });
      setMessages(prev => [...prev, { role: 'user', text }]);
    }
  }, [isConnected]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    isConnected,
    isAITalking,
    isUserTalking,
    volume,
    messages,
    error,
    connect,
    disconnect,
    sendTextMessage
  };
}

