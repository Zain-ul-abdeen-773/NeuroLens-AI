import { motion } from 'framer-motion';
import TextHeatmap from './TextHeatmap';

/**
 * Explainability panel showing WHY predictions were made.
 * Includes deception reasoning, top influential tokens, and feature analysis.
 */
export default function ExplainabilityPanel({ result }) {
  if (!result) return null;

  const dec = result.deception || {};
  const reasons = dec.reasons || [];
  const topTokens = result.top_tokens || [];
  const insights = result.linguistic_insights || [];

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4 }}
      className="space-y-5"
    >
      {/* ── Section Header ────────────────────────────────────────── */}
      <div className="flex items-center gap-2">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-neon-blue/20 to-neon-purple/20 flex items-center justify-center border border-white/10">
          <span className="text-lg">🔍</span>
        </div>
        <div>
          <h2 className="text-sm font-display text-white">Explainability Layer</h2>
          <p className="text-[10px] text-gray-500">Why this prediction was made</p>
        </div>
      </div>

      {/* ── Deception Reasoning ────────────────────────────────────── */}
      {reasons.length > 0 && (
        <motion.div
          className="glass-card p-5"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.1 }}
        >
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full" style={{ background: dec.verdict === 'deceptive' ? '#ff2d95' : '#00ff88' }} />
            Deception Analysis Reasoning
          </h3>
          <div className="space-y-2.5">
            {reasons.map((reason, i) => (
              <motion.div
                key={i}
                className="flex items-start gap-2.5 text-xs"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 + i * 0.1 }}
              >
                <span className="text-neon-purple mt-0.5 flex-shrink-0">▸</span>
                <p className="text-gray-300 leading-relaxed">{reason}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}

      {/* ── Token Importance Heatmap ───────────────────────────────── */}
      {result.text && topTokens.length > 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          <TextHeatmap text={result.text} tokenImportance={topTokens} />
        </motion.div>
      )}

      {/* ── Top Influential Tokens ─────────────────────────────────── */}
      {topTokens.length > 0 && (
        <motion.div
          className="glass-card p-5"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-neon-blue animate-pulse" />
            Most Influential Tokens
          </h3>
          <div className="flex flex-wrap gap-2">
            {topTokens.slice(0, 8).map((token, i) => {
              const alpha = 0.2 + token.importance * 0.6;
              return (
                <motion.span
                  key={i}
                  className="px-3 py-1.5 rounded-lg text-xs font-mono"
                  style={{
                    background: `rgba(0, 212, 255, ${alpha * 0.3})`,
                    border: `1px solid rgba(0, 212, 255, ${alpha})`,
                    color: `rgba(0, 212, 255, ${0.5 + token.importance * 0.5})`,
                  }}
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.3 + i * 0.05, type: 'spring' }}
                >
                  {token.token}
                  <span className="ml-1.5 text-[10px] opacity-60">
                    {(token.importance * 100).toFixed(0)}%
                  </span>
                </motion.span>
              );
            })}
          </div>
        </motion.div>
      )}

      {/* ── Meta Feature Summary ───────────────────────────────────── */}
      {result.meta_features && (
        <motion.div
          className="glass-card p-5"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
        >
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-neon-green" />
            Feature Analysis
          </h3>
          <div className="grid grid-cols-2 gap-2 text-xs">
            {Object.entries(result.meta_features)
              .slice(0, 8)
              .map(([key, val]) => (
                <div key={key} className="flex justify-between items-center py-1 border-b border-white/5">
                  <span className="text-gray-400 capitalize">
                    {key.replace(/_/g, ' ')}
                  </span>
                  <span className="font-mono text-neon-blue">
                    {typeof val === 'number' ? val.toFixed(3) : val}
                  </span>
                </div>
              ))}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
