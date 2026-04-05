import { motion } from 'framer-motion';

/**
 * Animated confidence meter with gradient fill and glow effect.
 */
export default function ConfidenceMeter({
  value = 0,
  label = 'Confidence',
  color = '#00d4ff',
  showLabel = true,
}) {
  const getGradient = (val) => {
    if (val > 0.7) return ['#ff2d95', '#ff6b35'];
    if (val > 0.4) return ['#ffea00', '#ff6b35'];
    return ['#00ff88', '#00d4ff'];
  };
  const [gradStart, gradEnd] = getGradient(value);

  return (
    <div className="w-full">
      {showLabel && (
        <div className="flex justify-between items-center mb-2">
          <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">
            {label}
          </span>
          <span
            className="text-sm font-mono font-bold"
            style={{ color }}
          >
            {(value * 100).toFixed(1)}%
          </span>
        </div>
      )}
      <div className="h-2 rounded-full bg-white/5 overflow-hidden relative">
        <motion.div
          className="h-full rounded-full relative"
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 1.2, ease: 'easeOut' }}
          style={{
            background: `linear-gradient(90deg, ${gradStart}, ${gradEnd})`,
            boxShadow: `0 0 10px ${gradStart}60`,
          }}
        >
          {/* Glowing tip */}
          <div
            className="absolute right-0 top-1/2 -translate-y-1/2 w-3 h-3 rounded-full"
            style={{
              background: gradEnd,
              boxShadow: `0 0 12px ${gradEnd}, 0 0 24px ${gradEnd}80`,
            }}
          />
        </motion.div>
      </div>
    </div>
  );
}
