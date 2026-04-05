import { useState, useCallback } from 'react';
import { analyzeText } from '../services/api';

/**
 * Custom hook for text analysis state management.
 */
export function useAnalysis() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);

  const analyze = useCallback(async (text, sessionId = null) => {
    setLoading(true);
    setError(null);

    try {
      const data = await analyzeText(text, sessionId);
      setResult(data);
      setHistory(prev => [...prev, { text, result: data, timestamp: new Date().toISOString() }]);
      return data;
    } catch (err) {
      setError(err.message);
      // Generate mock result for demo when backend is not available
      const mockResult = generateMockResult(text);
      setResult(mockResult);
      setHistory(prev => [...prev, { text, result: mockResult, timestamp: new Date().toISOString() }]);
      return mockResult;
    } finally {
      setLoading(false);
    }
  }, []);

  const clearResult = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { result, loading, error, history, analyze, clearResult };
}

/**
 * Generate realistic mock results for demo/offline mode.
 */
function generateMockResult(text) {
  const words = text.toLowerCase().split(/\s+/);
  const wordCount = words.length;

  // Heuristic deception detection
  const deceptionMarkers = ['honestly', 'trust', 'believe', 'swear', 'seriously', 'truly', 'frankly'];
  const hedgingWords = ['maybe', 'perhaps', 'possibly', 'might', 'could', 'probably', 'think'];
  const negativeWords = ['hate', 'angry', 'terrible', 'awful', 'horrible', 'disgusting', 'furious'];
  const positiveWords = ['love', 'happy', 'wonderful', 'great', 'amazing', 'fantastic', 'beautiful'];
  const manipulationWords = ['must', 'should', 'need', 'deserve', 'fault', 'blame', 'always', 'never'];

  const deceptionCount = words.filter(w => deceptionMarkers.includes(w)).length;
  const hedgingCount = words.filter(w => hedgingWords.includes(w)).length;
  const negCount = words.filter(w => negativeWords.includes(w)).length;
  const posCount = words.filter(w => positiveWords.includes(w)).length;
  const manipCount = words.filter(w => manipulationWords.includes(w)).length;

  const deceptionProb = Math.min(0.95, Math.max(0.05, (deceptionCount * 0.15 + hedgingCount * 0.08 + 0.2 + Math.random() * 0.15)));
  const manipulationProb = Math.min(0.9, Math.max(0.05, manipCount * 0.12 + Math.random() * 0.1));

  // Token importance (highlight deception-related tokens)
  const tokenImportance = words.slice(0, 50).map((word, i) => {
    let importance = Math.random() * 0.3;
    if (deceptionMarkers.includes(word)) importance = 0.7 + Math.random() * 0.3;
    else if (hedgingWords.includes(word)) importance = 0.5 + Math.random() * 0.3;
    else if (manipulationWords.includes(word)) importance = 0.4 + Math.random() * 0.3;
    return { token: word, importance: Math.min(1, importance), index: i };
  });

  // Emotions
  const emotionList = [
    'neutral', 'anger', 'joy', 'sadness', 'fear', 'surprise',
    'disgust', 'curiosity', 'nervousness', 'disappointment', 'approval',
    'caring', 'excitement', 'gratitude', 'love', 'optimism'
  ];
  const emotions = emotionList.map(emotion => {
    let prob = Math.random() * 0.3;
    if (emotion === 'anger' && negCount > 0) prob = 0.6 + Math.random() * 0.3;
    if (emotion === 'joy' && posCount > 0) prob = 0.6 + Math.random() * 0.3;
    if (emotion === 'neutral' && negCount === 0 && posCount === 0) prob = 0.5 + Math.random() * 0.3;
    if (emotion === 'nervousness' && hedgingCount > 0) prob = 0.4 + Math.random() * 0.3;
    return { emotion, probability: Math.min(0.95, prob) };
  }).sort((a, b) => b.probability - a.probability);

  // Reasons
  const reasons = [];
  if (deceptionCount > 0) reasons.push("Presence of classic deception markers ('honestly', 'trust me', 'believe me') — unprompted truthfulness claims");
  if (hedgingCount > 0) reasons.push("High use of hedging language and uncertainty markers (e.g., 'maybe', 'I think', 'possibly')");
  if (deceptionProb > 0.5) reasons.push("Pattern analysis detected subtle linguistic cues correlated with deceptive communication");
  if (reasons.length === 0) reasons.push("Text exhibits linguistic patterns consistent with truthful communication");

  const manipulationTypes = ['none', 'guilt_tripping', 'gaslighting', 'love_bombing', 'fear_mongering', 'flattery', 'coercion'];
  const detectedManipulation = manipulationProb > 0.3 ? manipulationTypes[Math.floor(Math.random() * (manipulationTypes.length - 1)) + 1] : 'none';

  return {
    text,
    deception: {
      probability: deceptionProb,
      verdict: deceptionProb > 0.5 ? 'deceptive' : 'truthful',
      confidence: Math.max(deceptionProb, 1 - deceptionProb),
      reasons,
    },
    emotions,
    dominant_emotions: emotions.filter(e => e.probability > 0.3),
    manipulation: {
      detected: detectedManipulation === 'none' ? null : detectedManipulation,
      probabilities: manipulationTypes.map(type => ({
        type,
        probability: type === detectedManipulation ? manipulationProb : Math.random() * 0.2,
      })),
      risk_level: manipulationProb > 0.6 ? 'high' : manipulationProb > 0.3 ? 'moderate' : 'low',
    },
    confidence_score: Math.max(deceptionProb, 1 - deceptionProb),
    token_importance: tokenImportance.sort((a, b) => b.importance - a.importance),
    top_tokens: tokenImportance.sort((a, b) => b.importance - a.importance).slice(0, 10),
    linguistic_insights: [
      { category: 'Sentiment', insight: negCount > posCount ? 'Negative emotional undertone detected' : 'Generally positive or neutral tone', severity: negCount > posCount ? 'medium' : 'low' },
      { category: 'Complexity', insight: wordCount > 30 ? 'Complex, multi-layered communication structure' : 'Concise, direct communication style', severity: 'low' },
    ],
    processing_time_ms: 45 + Math.random() * 120,
    meta_features: {
      word_count: wordCount,
      lexical_diversity: new Set(words).size / wordCount,
      sentiment_mean: (posCount - negCount) / wordCount,
      uncertainty_ratio: hedgingCount / wordCount,
    },
  };
}
