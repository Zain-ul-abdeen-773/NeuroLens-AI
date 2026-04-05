import { motion } from 'framer-motion';

/**
 * Animated waveform visualization for active analysis state.
 */
export default function WaveformEffect({ active = false, barCount = 20, color = '#00d4ff' }) {
  return (
    <div className="flex items-center justify-center gap-[3px] h-8">
      {Array.from({ length: barCount }).map((_, i) => (
        <motion.div
          key={i}
          className="waveform-bar"
          style={{
            width: 3,
            background: `linear-gradient(180deg, ${color}, ${color}40)`,
            boxShadow: active ? `0 0 4px ${color}60` : 'none',
          }}
          animate={
            active
              ? {
                  height: [4, 16 + Math.random() * 16, 4],
                }
              : { height: 4 }
          }
          transition={
            active
              ? {
                  duration: 0.6 + Math.random() * 0.4,
                  repeat: Infinity,
                  repeatType: 'reverse',
                  delay: i * 0.05,
                  ease: 'easeInOut',
                }
              : { duration: 0.3 }
          }
        />
      ))}
    </div>
  );
}
