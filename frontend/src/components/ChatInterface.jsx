import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import WaveformEffect from './WaveformEffect';

/**
 * ChatGPT-style chat interface with live typing analysis.
 */
export default function ChatInterface({ onAnalyze, loading, result }) {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const chatEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (result && messages.length > 0) {
      const lastIdx = messages.length - 1;
      setMessages(prev => {
        const updated = [...prev];
        if (updated[lastIdx] && updated[lastIdx].type === 'user') {
          updated.push({
            type: 'analysis',
            data: result,
            timestamp: new Date().toISOString(),
          });
        }
        return updated;
      });
    }
  }, [result]);

  const handleSubmit = (e) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    setMessages(prev => [...prev, {
      type: 'user',
      text,
      timestamp: new Date().toISOString(),
    }]);

    setInput('');
    onAnalyze(text);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const quickExamples = [
    "I honestly had nothing to do with it, trust me.",
    "You should be ashamed. After everything I've done for you, this is how you repay me?",
    "I'm feeling really grateful for your support today. It means everything to me!",
    "I don't know, maybe it happened, I'm not sure. I think possibly someone else was there.",
  ];

  return (
    <div className="flex flex-col h-full">
      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
        {messages.length === 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex flex-col items-center justify-center h-full text-center"
          >
            <div className="w-20 h-20 mb-6 rounded-2xl bg-gradient-to-br from-neon-blue/20 to-neon-purple/20 flex items-center justify-center border border-white/10">
              <span className="text-4xl">🧠</span>
            </div>
            <h2 className="text-2xl font-display gradient-text mb-3">
              NeuroLens AI
            </h2>
            <p className="text-gray-400 text-sm max-w-md mb-8">
              Paste any text to analyze for deception signals, emotional patterns,
              and manipulation indicators in real-time.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-lg w-full">
              {quickExamples.map((example, i) => (
                <motion.button
                  key={i}
                  onClick={() => { setInput(example); inputRef.current?.focus(); }}
                  className="glass-card-hover p-3 text-left text-xs text-gray-300 hover:text-white"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <span className="text-neon-blue/60 mr-1">→</span>
                  {example.length > 70 ? example.slice(0, 70) + '...' : example}
                </motion.button>
              ))}
            </div>
          </motion.div>
        )}

        <AnimatePresence>
          {messages.map((msg, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
            >
              {msg.type === 'user' ? (
                <div className="flex justify-end">
                  <div className="max-w-[80%] glass-card p-4 rounded-2xl rounded-br-sm border-neon-blue/20">
                    <p className="text-sm text-gray-200 whitespace-pre-wrap">{msg.text}</p>
                    <span className="text-[10px] text-gray-500 mt-2 block text-right">
                      {new Date(msg.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                </div>
              ) : msg.type === 'analysis' ? (
                <div className="flex justify-start">
                  <div className="max-w-[85%] glass-card p-4 rounded-2xl rounded-bl-sm border-neon-purple/20">
                    <AnalysisBubble data={msg.data} />
                  </div>
                </div>
              ) : null}
            </motion.div>
          ))}
        </AnimatePresence>

        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex justify-start"
          >
            <div className="glass-card p-4 rounded-2xl rounded-bl-sm">
              <div className="flex items-center gap-3">
                <WaveformEffect active barCount={12} />
                <span className="text-sm text-neon-blue animate-pulse">Analyzing...</span>
              </div>
            </div>
          </motion.div>
        )}

        <div ref={chatEndRef} />
      </div>

      {/* Input Area */}
      <div className="p-4 border-t border-white/5">
        <form onSubmit={handleSubmit} className="flex gap-3">
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Paste text to analyze for behavioral signals..."
              className="neuro-input resize-none h-12 max-h-32 pr-4"
              rows={1}
              style={{ minHeight: '48px' }}
              onInput={(e) => {
                e.target.style.height = '48px';
                e.target.style.height = Math.min(e.target.scrollHeight, 128) + 'px';
              }}
              id="chat-input"
            />
          </div>
          <motion.button
            type="submit"
            disabled={!input.trim() || loading}
            className="neuro-btn-primary px-6 py-3 rounded-xl flex items-center gap-2"
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
            <span className="hidden sm:inline">Analyze</span>
          </motion.button>
        </form>
      </div>
    </div>
  );
}

/**
 * Compact analysis result bubble displayed in chat.
 */
function AnalysisBubble({ data }) {
  if (!data) return null;

  const dec = data.deception || {};
  const emotions = (data.dominant_emotions || data.emotions || []).slice(0, 5);
  const manip = data.manipulation || {};

  const verdictColor = dec.verdict === 'deceptive' ? '#ff2d95' : '#00ff88';
  const riskColors = { low: '#00ff88', moderate: '#ffea00', high: '#ff6b35', critical: '#ff2d95' };

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-xs text-gray-500">
        <span className="w-1.5 h-1.5 rounded-full bg-neon-purple animate-pulse" />
        NeuroLens Analysis
      </div>

      {/* Deception */}
      <div className="flex items-center gap-3">
        <span className="text-xl" role="img">
          {dec.verdict === 'deceptive' ? '⚠️' : '✅'}
        </span>
        <div>
          <span className="text-sm font-semibold" style={{ color: verdictColor }}>
            {dec.verdict === 'deceptive' ? 'DECEPTION DETECTED' : 'TRUTHFUL'}
          </span>
          <span className="text-xs text-gray-400 ml-2">
            ({(dec.probability * 100).toFixed(1)}% probability)
          </span>
        </div>
      </div>

      {/* Emotions */}
      {emotions.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {emotions.map((e, i) => (
            <span
              key={i}
              className="px-2 py-0.5 rounded-full text-[10px] font-medium border"
              style={{
                borderColor: `hsl(${(i * 60) % 360}, 70%, 60%)`,
                color: `hsl(${(i * 60) % 360}, 70%, 70%)`,
                background: `hsla(${(i * 60) % 360}, 70%, 60%, 0.1)`,
              }}
            >
              {e.emotion} {(e.probability * 100).toFixed(0)}%
            </span>
          ))}
        </div>
      )}

      {/* Manipulation */}
      {manip.detected && (
        <div className="text-xs flex items-center gap-2">
          <span className="text-neon-pink">⚡ Manipulation:</span>
          <span className="capitalize" style={{ color: riskColors[manip.risk_level] || '#ffea00' }}>
            {manip.detected.replace(/_/g, ' ')}
          </span>
        </div>
      )}

      {/* Processing time */}
      <div className="text-[10px] text-gray-600">
        Processed in {data.processing_time_ms?.toFixed(0)}ms
      </div>
    </div>
  );
}
