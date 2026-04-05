import { motion } from 'framer-motion';

/**
 * Animated radial progress bar with neon glow.
 */
export default function RadialProgress({
  value = 0,
  size = 120,
  strokeWidth = 8,
  color = '#00d4ff',
  label = '',
  sublabel = '',
}) {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (value / 100) * circumference;
  const center = size / 2;

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="radial-progress-ring">
          {/* Background ring */}
          <circle
            cx={center}
            cy={center}
            r={radius}
            stroke="rgba(255,255,255,0.05)"
            strokeWidth={strokeWidth}
            fill="none"
          />
          {/* Progress ring */}
          <motion.circle
            cx={center}
            cy={center}
            r={radius}
            stroke={color}
            strokeWidth={strokeWidth}
            fill="none"
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: offset }}
            transition={{ duration: 1.5, ease: 'easeOut' }}
            style={{
              filter: `drop-shadow(0 0 6px ${color}80)`,
            }}
          />
        </svg>
        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.span
            className="text-2xl font-bold font-display"
            style={{ color }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            {Math.round(value)}%
          </motion.span>
        </div>
      </div>
      {label && (
        <span className="text-sm font-medium text-gray-300">{label}</span>
      )}
      {sublabel && (
        <span className="text-xs text-gray-500">{sublabel}</span>
      )}
    </div>
  );
}
