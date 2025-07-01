export const calculateSMA = (data, period) => {
  if (!Array.isArray(data) || data.length < period) return [];
  const result = new Array(period - 1).fill(undefined);
  let sum = data.slice(0, period).reduce((acc, val) => acc + val, 0);
  result.push(sum / period);
  for (let i = period; i < data.length; i++) {
    sum = sum - data[i - period] + data[i];
    result.push(sum / period);
  }
  return result;
};

export const calculateEMA = (data, period) => {
  if (!Array.isArray(data) || data.length < period) return [];
  const result = new Array(period - 1).fill(undefined);
  const k = 2 / (period + 1);
  let ema = data.slice(0, period).reduce((sum, val) => sum + val, 0) / period;
  result.push(ema);
  for (let i = period; i < data.length; i++) {
    ema = data[i] * k + ema * (1 - k);
    result.push(ema);
  }
  return result;
};
