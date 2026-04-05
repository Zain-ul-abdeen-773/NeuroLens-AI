/**
 * Interactive text heatmap with hover tooltips.
 * Highlights words based on their importance/influence on the prediction.
 */
export default function TextHeatmap({ text, tokenImportance = [] }) {
  if (!text) return null;

  const words = text.split(/\s+/);
  const importanceMap = {};

  // Build importance lookup from token data
  tokenImportance.forEach(({ token, importance }) => {
    const cleanToken = token.replace(/^##/, '').toLowerCase();
    importanceMap[cleanToken] = Math.max(
      importanceMap[cleanToken] || 0,
      importance
    );
  });

  const getColor = (importance) => {
    if (importance > 0.7) return 'rgba(255, 45, 149, 0.5)';
    if (importance > 0.5) return 'rgba(255, 107, 53, 0.4)';
    if (importance > 0.3) return 'rgba(255, 234, 0, 0.3)';
    if (importance > 0.15) return 'rgba(0, 212, 255, 0.2)';
    return 'transparent';
  };

  const getBorderColor = (importance) => {
    if (importance > 0.7) return 'rgba(255, 45, 149, 0.6)';
    if (importance > 0.5) return 'rgba(255, 107, 53, 0.5)';
    if (importance > 0.3) return 'rgba(255, 234, 0, 0.4)';
    return 'transparent';
  };

  return (
    <div className="glass-card p-5">
      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3 flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-neon-blue animate-pulse" />
        Token Importance Heatmap
      </h3>
      <div className="leading-8 text-base">
        {words.map((word, i) => {
          const importance = importanceMap[word.toLowerCase().replace(/[.,!?;:'"]/g, '')] || 0;
          return (
            <span
              key={i}
              className="heatmap-word"
              style={{
                backgroundColor: getColor(importance),
                borderBottom: `2px solid ${getBorderColor(importance)}`,
              }}
            >
              {word}
              {importance > 0.1 && (
                <span className="tooltip">
                  Importance: {(importance * 100).toFixed(0)}%
                </span>
              )}
            </span>
          );
        })}
      </div>
      {/* Legend */}
      <div className="flex items-center gap-4 mt-4 text-xs text-gray-500">
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded" style={{ background: 'rgba(0, 212, 255, 0.2)' }} />
          Low
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded" style={{ background: 'rgba(255, 234, 0, 0.3)' }} />
          Medium
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded" style={{ background: 'rgba(255, 107, 53, 0.4)' }} />
          High
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded" style={{ background: 'rgba(255, 45, 149, 0.5)' }} />
          Critical
        </span>
      </div>
    </div>
  );
}
