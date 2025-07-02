import React, { useRef, useLayoutEffect, useState, useCallback } from 'react';
import {
  Box,
  Typography,
  Switch,
  FormControlLabel,
  Paper,
  Stack,
  useTheme,
  alpha
} from '@mui/material';
import { createChart, ColorType, CrosshairMode } from 'lightweight-charts';
import { calculateEMA, calculateSMA } from '../utils/indicators.js'; // ✅ 更新 import 路徑

const StockChart = ({ stockData = [], stockCode }) => {
  const theme = useTheme();
  const chartContainerRef = useRef(null);

  const [indicators, setIndicators] = useState({
    ema6: true, ema12: true, ema24: true,
    sma50: true, sma150: true, sma200: true,
  });

  useLayoutEffect(() => {
    const chartContainer = chartContainerRef.current;
    if (!chartContainer || !stockData || stockData.length === 0) {
      if(chartContainer) chartContainer.innerHTML = '';
      return;
    }

    const sortedData = [...stockData].sort((a, b) => a.time - b.time);
    const candleData = sortedData.map(item => ({
      time: item.time, open: item.open, high: item.high, low: item.low, close: item.close,
    }));
    const volumeData = sortedData.map(item => ({
      time: item.time, value: item.volume,
      color: item.close >= item.open ? alpha(theme.palette.success.main, 0.6) : alpha(theme.palette.error.main, 0.6),
    }));

    const chart = createChart(chartContainer, {
      width: chartContainer.clientWidth,
      height: chartContainer.clientHeight,
      layout: {
        background: { type: ColorType.Solid, color: theme.palette.background.paper },
        textColor: theme.palette.text.primary,
        fontFamily: theme.typography.fontFamily,
      },
      grid: { vertLines: { color: theme.palette.divider }, horzLines: { color: theme.palette.divider } },
      crosshair: { mode: CrosshairMode.Normal },
      timeScale: { borderColor: theme.palette.divider, timeVisible: true },
    });

    const candlestickSeries = chart.addCandlestickSeries({
      upColor: theme.palette.success.main, downColor: theme.palette.error.main,
      borderVisible: false,
      wickUpColor: theme.palette.success.light, wickDownColor: theme.palette.error.light,
    });
    candlestickSeries.setData(candleData);

    const volumeSeries = chart.addHistogramSeries({
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    });
    volumeSeries.setData(volumeData);
    chart.priceScale('volume').applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });

    const closes = sortedData.map(d => d.close);
    const indicatorConfigs = [
      { key: 'ema6',   data: calculateEMA(closes, 6),   color: '#2196f3', width: 1 },
      { key: 'ema12',  data: calculateEMA(closes, 12),  color: '#ff9800', width: 1 },
      { key: 'ema24',  data: calculateEMA(closes, 24),  color: '#9c27b0', width: 1 },
      { key: 'sma50',  data: calculateSMA(closes, 50),  color: '#4caf50', width: 2 },
      { key: 'sma150', data: calculateSMA(closes, 150), color: '#f44336', width: 2 },
      { key: 'sma200', data: calculateSMA(closes, 200), color: theme.palette.mode === 'dark' ? '#ffffff' : '#000000', width: 2 },
    ];
    
    indicatorConfigs.forEach(config => {
      if (indicators[config.key]) {
        const indicatorData = config.data
          .map((value, index) => ({ time: sortedData[index].time, value }))
          .filter(item => item.value !== undefined && !isNaN(item.value));
        
        const lineSeries = chart.addLineSeries({
          color: config.color, lineWidth: config.width,
          crosshairMarkerVisible: false, lastValueVisible: false,
        });
        lineSeries.setData(indicatorData);
      }
    });

    chart.timeScale().fitContent();
    
    const handleResize = () => chart.applyOptions({ width: chartContainer.clientWidth });
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [stockData, indicators, theme, stockCode]);

  const handleIndicatorToggle = useCallback((indicatorKey) => {
    setIndicators(prev => ({ ...prev, [indicatorKey]: !prev[indicatorKey] }));
  }, []);

  return (
    <Box height="100%" display="flex" flexDirection="column">
      <Paper elevation={0} sx={{ p: 1, mb: 1, flexShrink: 0 }}>
        <Typography variant="subtitle2" gutterBottom>技術指標</Typography>
        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
          {Object.keys(indicators).map(key => (
            <FormControlLabel
              key={key}
              control={<Switch checked={indicators[key]} onChange={() => handleIndicatorToggle(key)} size="small" />}
              label={key.toUpperCase()}
              sx={{ mr: 1 }}
            />
          ))}
        </Stack>
      </Paper>
      <Box ref={chartContainerRef} sx={{ flexGrow: 1, width: '100%', minHeight: 300 }} />
      {(!stockData || stockData.length === 0) && (
          <Box sx={{ p: 4, textAlign: 'center', position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }}>
              <Typography variant="h6" color="text.secondary">
                  {stockCode ? `沒有 ${stockCode} 的圖表資料` : "請選擇股票"}
              </Typography>
          </Box>
      )}
    </Box>
  );
};

export default React.memo(StockChart);
