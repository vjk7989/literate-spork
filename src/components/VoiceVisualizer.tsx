import React from 'react';
import { motion } from 'motion/react';

interface VoiceVisualizerProps {
  isUserTalking: boolean;
  isAITalking: boolean;
  volume: number;
}

export const VoiceVisualizer: React.FC<VoiceVisualizerProps> = ({ isUserTalking, isAITalking, volume }) => {
  const bars = [0, 1, 2, 3, 4, 5, 6];
  const dots = Array.from({ length: 48 });

  const activeColor = isUserTalking ? '#bf00ff' : (isAITalking ? '#ffffff' : '#333333');
  const isActive = isUserTalking || isAITalking;

  // Normalize volume for visualization (0 to 1 range)
  // RMS values are typically small, so we'll scale it up
  const normalizedVolume = Math.min(1, volume * 5);

  return (
    <div className="relative w-full h-32 flex items-center justify-center overflow-hidden">
      {/* Dot Grid Background */}
      <div className="absolute inset-0 grid grid-cols-12 gap-4 p-4 opacity-20">
        {dots.map((_, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0.1 }}
            animate={{ 
              opacity: isActive ? [0.1, 0.2 + normalizedVolume * 0.3, 0.1] : 0.1,
              scale: isActive ? [1, 1 + normalizedVolume * 0.5, 1] : 1
            }}
            transition={{ 
              duration: 1.5, 
              repeat: Infinity, 
              delay: i * 0.02,
              ease: "easeInOut"
            }}
            className="w-1 h-1 bg-white rounded-full"
          />
        ))}
      </div>

      {/* Central Waveform Bars */}
      <div className="relative z-10 flex items-center gap-1.5">
        {bars.map((i) => {
          // Calculate height based on index for a diamond/wave shape
          const baseHeight = i === 3 ? 64 : (i === 2 || i === 4 ? 48 : (i === 1 || i === 5 ? 32 : 16));
          
          // Add randomness/dynamic scaling based on volume
          const dynamicScale = isActive ? (0.3 + normalizedVolume * 0.7) : 0.1;
          const targetHeight = isActive ? baseHeight * dynamicScale : 4;

          return (
            <motion.div
              key={i}
              initial={{ height: 4 }}
              animate={{ 
                height: targetHeight,
                backgroundColor: activeColor,
                boxShadow: isActive ? `0 0 ${10 + normalizedVolume * 20}px ${activeColor}66` : 'none'
              }}
              transition={{ 
                type: "spring",
                stiffness: 300,
                damping: 20,
                mass: 0.5
              }}
              className="w-2 rounded-full transition-colors duration-300"
            />
          );
        })}
      </div>

      {/* Glow Effect */}
      <motion.div 
        animate={{ 
          opacity: isActive ? [0.1, 0.15 + normalizedVolume * 0.15, 0.1] : 0,
          scale: isActive ? [1, 1 + normalizedVolume * 0.4, 1] : 1
        }}
        className="absolute w-48 h-48 rounded-full blur-3xl pointer-events-none transition-all duration-300"
        style={{ backgroundColor: activeColor }}
      />
    </div>
  );
};
