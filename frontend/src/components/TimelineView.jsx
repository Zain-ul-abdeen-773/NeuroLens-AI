import { motion } from 'framer-motion';
import { EMOTION_COLORS, RISK_COLORS } from '../utils/constants';

/**
 * Timeline view showing behavioral evolution over a conversation session.
 */
export default function TimelineView({ history = [] }) {
  if (history.length === 0) return <EmptyTimeline />;

  // Extract data series from history
  const timelineData = history.map((entry, i) => {
    const result = entry.result || {};
    return {
      index: i + 1,
      timestamp: entry.timestamp,
      text: entry.text?.slice(0, 50) + (entry.text?.length > 50 ? '...' : ''),
      deception: result.deception?.probability || 0,
      confidence: result.confidence_score || 0,
      emotion: result.dominant_emotions?.[0]?.emotion || 'neutral',
      emotionProb: result.dominant_emotions?.[0]?.probability || 0,
      manipulation: result.manipulation?.detected,
      risk: result.manipulation?.risk_level || 'low',
    };
  });

  const maxDeception = Math.max(...timelineData.map(d => d.deception));

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-4"
    >
      <div className="flex items-center gap-2 mb-4">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-neon-purple/20 to-neon-pink/20 flex items-center justify-center border border-white/10">
          <span className="text-lg">📊</span>
        </div>
        <div>
          <h2 className="text-sm font-display text-white">Behavioral Timeline</h2>
          <p className="text-[10px] text-gray-500">Track behavioral patterns over time</p>
        </div>
      </div>

      {/* ── Deception Trend Chart ──────────────────────────────────── */}
      <div className="glass-card p-5">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-4">
          Deception Probability Trend
        </h3>
        <div className="h-32 flex items-end gap-1">
          {timelineData.map((d, i) => (
            <motion.div
              key={i}
              className="flex-1 flex flex-col items-center gap-1"
              initial={{ height: 0 }}
              animate={{ height: 'auto' }}
              transition={{ delay: i * 0.1 }}
            >
              <span className="text-[9px] font-mono text-gray-500">
                {(d.deception * 100).toFixed(0)}%
              </span>
              <motion.div
                className="w-full rounded-t-md relative overflow-hidden"
                initial={{ height: 0 }}
                animate={{ height: `${Math.max(d.deception * 100, 4)}%` }}
                transition={{ delay: i * 0.1, duration: 0.5 }}
                style={{
                  background: d.deception > 0.6
                    ? 'linear-gradient(180deg, #ff2d95, #ff2d9540)'
                    : d.deception > 0.3
                    ? 'linear-gradient(180deg, #ffea00, #ffea0040)'
                    : 'linear-gradient(180deg, #00ff88, #00ff8840)',
                  minHeight: '4px',
                  boxShadow: d.deception > 0.5 ? '0 0 10px rgba(255,45,149,0.3)' : 'none',
                }}
              />
              <span className="text-[9px] text-gray-600">#{d.index}</span>
            </motion.div>
          ))}
        </div>
      </div>

      {/* ── Timeline Entries ───────────────────────────────────────── */}
      <div className="relative">
        {/* Vertical line */}
        <div className="absolute left-4 top-0 bottom-0 w-px bg-gradient-to-b from-neon-blue/30 via-neon-purple/30 to-transparent" />

        <div className="space-y-3">
          {timelineData.map((d, i) => (
            <motion.div
              key={i}
              className="relative pl-10"
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.05 }}
            >
              {/* Timeline dot */}
              <div
                className="absolute left-2.5 top-3 w-3 h-3 rounded-full border-2"
                style={{
                  borderColor: d.deception > 0.5 ? '#ff2d95' : '#00d4ff',
                  background: d.deception > 0.5 ? '#ff2d9540' : '#00d4ff40',
                  boxShadow: `0 0 8px ${d.deception > 0.5 ? '#ff2d9560' : '#00d4ff60'}`,
                }}
              />

              <div className="glass-card p-3 hover:border-white/10 transition-colors">
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-mono text-gray-500">
                      Message #{d.index}
                    </span>
                    {d.manipulation && (
                      <span
                        className="text-[9px] px-1.5 py-0.5 rounded-full border"
                        style={{
                          borderColor: RISK_COLORS[d.risk],
                          color: RISK_COLORS[d.risk],
                        }}
                      >
                        ⚡ {d.manipulation.replace(/_/g, ' ')}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <span
                      className="text-[10px] px-2 py-0.5 rounded-full"
                      style={{
                        background: `${EMOTION_COLORS[d.emotion] || '#90a4ae'}20`,
                        color: EMOTION_COLORS[d.emotion] || '#90a4ae',
                      }}
                    >
                      {d.emotion}
                    </span>
                    <span
                      className="text-[10px] font-mono font-bold"
                      style={{
                        color: d.deception > 0.5 ? '#ff2d95' : '#00ff88',
                      }}
                    >
                      {(d.deception * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
                <p className="text-xs text-gray-400 truncate">{d.text}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </motion.div>
  );
}

function EmptyTimeline() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center py-20">
      <div className="w-16 h-16 mb-4 rounded-xl bg-white/5 flex items-center justify-center opacity-40">
        <span className="text-2xl">📊</span>
      </div>
      <p className="text-gray-600 text-sm">No conversation history yet</p>
      <p className="text-gray-700 text-xs mt-1">Analyze multiple messages to see behavioral trends</p>
    </div>
  );
}
