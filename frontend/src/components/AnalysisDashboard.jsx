import { motion } from 'framer-motion';
import RadialProgress from './RadialProgress';
import ConfidenceMeter from './ConfidenceMeter';
import { EMOTION_COLORS, RISK_COLORS } from '../utils/constants';

/**
 * Full analysis dashboard with radial progress, emotion bars, and manipulation display.
 */
export default function AnalysisDashboard({ result }) {
  if (!result) return <EmptyDashboard />;

  const dec = result.deception || {};
  const emotions = (result.emotions || []).slice(0, 12);
  const manip = result.manipulation || {};
  const insights = result.linguistic_insights || [];

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="space-y-6"
    >
      {/* ── Top Row: Key Metrics ─────────────────────────────────── */}
      <div className="grid grid-cols-3 gap-4">
        <motion.div
          className="glass-card p-5 flex flex-col items-center"
          initial={{ scale: 0.9 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.1 }}
        >
          <RadialProgress
            value={dec.probability * 100}
            color={dec.verdict === 'deceptive' ? '#ff2d95' : '#00ff88'}
            label="Deception"
            sublabel={dec.verdict?.toUpperCase()}
            size={110}
          />
        </motion.div>

        <motion.div
          className="glass-card p-5 flex flex-col items-center"
          initial={{ scale: 0.9 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.2 }}
        >
          <RadialProgress
            value={result.confidence_score * 100}
            color="#00d4ff"
            label="Confidence"
            size={110}
          />
        </motion.div>

        <motion.div
          className="glass-card p-5 flex flex-col items-center"
          initial={{ scale: 0.9 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.3 }}
        >
          <RadialProgress
            value={
              manip.probabilities
                ? Math.max(...manip.probabilities.filter(p => p.type !== 'none').map(p => p.probability)) * 100
                : 0
            }
            color={RISK_COLORS[manip.risk_level] || '#00ff88'}
            label="Manipulation"
            sublabel={manip.risk_level?.toUpperCase() + ' RISK'}
            size={110}
          />
        </motion.div>
      </div>

      {/* ── Emotion Spectrum ─────────────────────────────────────── */}
      <motion.div
        className="glass-card p-5"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-neon-purple animate-pulse" />
          Emotional Spectrum
        </h3>
        <div className="space-y-3">
          {emotions.map((emo, i) => (
            <div key={i} className="flex items-center gap-3">
              <span className="text-xs text-gray-400 w-24 truncate capitalize">
                {emo.emotion}
              </span>
              <div className="flex-1">
                <ConfidenceMeter
                  value={emo.probability}
                  color={EMOTION_COLORS[emo.emotion] || '#00d4ff'}
                  showLabel={false}
                />
              </div>
              <span className="text-xs font-mono text-gray-500 w-12 text-right">
                {(emo.probability * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      </motion.div>

      {/* ── Manipulation Breakdown ───────────────────────────────── */}
      {manip.probabilities && (
        <motion.div
          className="glass-card p-5"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-neon-pink animate-pulse" />
            Manipulation Analysis
          </h3>
          <div className="grid grid-cols-2 gap-3">
            {manip.probabilities
              .filter(p => p.type !== 'none')
              .map((p, i) => (
                <div key={i} className="flex items-center gap-2">
                  <div
                    className="w-1.5 h-6 rounded-full"
                    style={{
                      background: p.probability > 0.3
                        ? `linear-gradient(180deg, #ff2d95, #ff6b35)`
                        : 'rgba(255,255,255,0.1)',
                    }}
                  />
                  <div className="flex-1">
                    <div className="text-xs capitalize text-gray-300">
                      {p.type.replace(/_/g, ' ')}
                    </div>
                    <div className="text-[10px] font-mono text-gray-500">
                      {(p.probability * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              ))}
          </div>
        </motion.div>
      )}

      {/* ── Linguistic Insights ──────────────────────────────────── */}
      {insights.length > 0 && (
        <motion.div
          className="glass-card p-5"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
        >
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-neon-green animate-pulse" />
            Linguistic Insights
          </h3>
          <div className="space-y-2">
            {insights.map((ins, i) => {
              const severityColors = { low: '#00ff88', medium: '#ffea00', high: '#ff6b35', critical: '#ff2d95' };
              return (
                <div key={i} className="flex items-start gap-2 text-xs">
                  <span
                    className="w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0"
                    style={{ background: severityColors[ins.severity] || '#00d4ff' }}
                  />
                  <div>
                    <span className="text-gray-400">{ins.category}: </span>
                    <span className="text-gray-300">{ins.insight}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </motion.div>
      )}

      {/* ── Processing Info ──────────────────────────────────────── */}
      <div className="flex justify-between text-[10px] text-gray-600 px-1">
        <span>Processed in {result.processing_time_ms?.toFixed(0)}ms</span>
        <span>Meta features: {Object.keys(result.meta_features || {}).length}</span>
      </div>
    </motion.div>
  );
}

function EmptyDashboard() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center py-20">
      <div className="w-16 h-16 mb-4 rounded-xl bg-white/5 flex items-center justify-center opacity-40">
        <svg className="w-8 h-8 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      </div>
      <p className="text-gray-600 text-sm">Analysis results will appear here</p>
      <p className="text-gray-700 text-xs mt-1">Enter text in the chat to begin</p>
    </div>
  );
}
