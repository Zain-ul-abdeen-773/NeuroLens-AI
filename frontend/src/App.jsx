import { useState, Suspense } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ParticleBackground from './components/ParticleBackground';
import ChatInterface from './components/ChatInterface';
import AnalysisDashboard from './components/AnalysisDashboard';
import ExplainabilityPanel from './components/ExplainabilityPanel';
import TimelineView from './components/TimelineView';
import { useAnalysis } from './hooks/useAnalysis';

// Lazy load 3D component for performance
import NeuralNetworkViz from './components/NeuralNetworkViz';

const tabs = [
  { id: 'dashboard', label: 'Dashboard', icon: '📊' },
  { id: 'explainability', label: 'Explain', icon: '🔍' },
  { id: 'timeline', label: 'Timeline', icon: '📈' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const { result, loading, error, history, analyze } = useAnalysis();

  return (
    <div className="min-h-screen relative cyber-grid">
      {/* Particle Background */}
      <ParticleBackground />

      {/* ── Header ──────────────────────────────────────────────── */}
      <header className="relative z-10 border-b border-white/5">
        <div className="max-w-[1600px] mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <motion.div
              className="w-10 h-10 rounded-xl bg-gradient-to-br from-neon-blue/30 to-neon-purple/30 flex items-center justify-center border border-white/10"
              animate={{ rotate: [0, 5, -5, 0] }}
              transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
            >
              <span className="text-xl">🧠</span>
            </motion.div>
            <div>
              <h1 className="text-lg font-display font-bold gradient-text tracking-wider">
                NEUROLENS
              </h1>
              <p className="text-[10px] text-gray-500 font-mono tracking-widest uppercase">
                Behavioral Intelligence Engine
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* Status indicator */}
            <div className="flex items-center gap-2">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-neon-green opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-neon-green"></span>
              </span>
              <span className="text-[10px] text-gray-400 font-mono">ACTIVE</span>
            </div>

            {/* Version badge */}
            <span className="text-[10px] px-2 py-1 rounded-full border border-white/10 text-gray-500 font-mono">
              v1.0.0
            </span>
          </div>
        </div>
      </header>

      {/* ── Main Layout ─────────────────────────────────────────── */}
      <main className="relative z-10 max-w-[1600px] mx-auto px-6 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6" style={{ height: 'calc(100vh - 120px)' }}>

          {/* ── Left Panel: Chat ─────────────────────────────────── */}
          <div className="lg:col-span-5 glass-card flex flex-col overflow-hidden">
            <ChatInterface
              onAnalyze={analyze}
              loading={loading}
              result={result}
            />
          </div>

          {/* ── Right Panel: Analysis ────────────────────────────── */}
          <div className="lg:col-span-7 flex flex-col gap-4 overflow-hidden">

            {/* 3D Neural Network */}
            <Suspense fallback={
              <div className="glass-card h-[200px] flex items-center justify-center">
                <span className="text-gray-500 text-sm">Loading 3D visualization...</span>
              </div>
            }>
              <NeuralNetworkViz />
            </Suspense>

            {/* Tab Navigation */}
            <div className="flex gap-1 p-1 rounded-xl bg-white/[0.03] border border-white/5">
              {tabs.map((tab) => (
                <motion.button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex-1 flex items-center justify-center gap-2 py-2.5 px-4 rounded-lg text-xs font-medium transition-all ${
                    activeTab === tab.id
                      ? 'bg-gradient-to-r from-neon-blue/15 to-neon-purple/10 text-neon-blue border border-neon-blue/20'
                      : 'text-gray-400 hover:text-gray-300 hover:bg-white/[0.03]'
                  }`}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  id={`tab-${tab.id}`}
                >
                  <span>{tab.icon}</span>
                  <span className="hidden sm:inline">{tab.label}</span>
                </motion.button>
              ))}
            </div>

            {/* Tab Content */}
            <div className="flex-1 overflow-y-auto min-h-0">
              <AnimatePresence mode="wait">
                {activeTab === 'dashboard' && (
                  <motion.div
                    key="dashboard"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ duration: 0.2 }}
                  >
                    <AnalysisDashboard result={result} />
                  </motion.div>
                )}

                {activeTab === 'explainability' && (
                  <motion.div
                    key="explainability"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ duration: 0.2 }}
                  >
                    <ExplainabilityPanel result={result} />
                  </motion.div>
                )}

                {activeTab === 'timeline' && (
                  <motion.div
                    key="timeline"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ duration: 0.2 }}
                  >
                    <TimelineView history={history} />
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
