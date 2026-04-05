// NeuroLens API Constants
export const API_BASE = '/api/v1';
export const WS_URL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/analyze`;

export const EMOTION_COLORS = {
  admiration: '#FFD700',
  amusement: '#FF69B4',
  anger: '#FF4444',
  annoyance: '#FF6B6B',
  approval: '#00E676',
  caring: '#E040FB',
  confusion: '#FFB74D',
  curiosity: '#40C4FF',
  desire: '#FF1744',
  disappointment: '#78909C',
  disapproval: '#EF5350',
  disgust: '#8BC34A',
  embarrassment: '#FFAB91',
  excitement: '#FFEA00',
  fear: '#9C27B0',
  gratitude: '#26A69A',
  grief: '#546E7A',
  joy: '#FFC107',
  love: '#E91E63',
  nervousness: '#B39DDB',
  optimism: '#66BB6A',
  pride: '#AB47BC',
  realization: '#29B6F6',
  relief: '#80CBC4',
  remorse: '#8D6E63',
  sadness: '#5C6BC0',
  surprise: '#FF7043',
  neutral: '#90A4AE',
};

export const MANIPULATION_COLORS = {
  none: '#00ff88',
  guilt_tripping: '#ff6b35',
  gaslighting: '#ff2d95',
  love_bombing: '#e040fb',
  fear_mongering: '#9c27b0',
  flattery: '#ffd700',
  coercion: '#ff4444',
};

export const RISK_COLORS = {
  low: '#00ff88',
  moderate: '#ffea00',
  high: '#ff6b35',
  critical: '#ff2d95',
};
