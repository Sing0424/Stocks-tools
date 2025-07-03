import React, { useRef, useLayoutEffect, useState, useCallback, useMemo } from 'react';
import {
  Box, Typography, Switch, FormControlLabel, Paper, Stack, ButtonGroup, Button, useTheme, alpha
} from '@mui/material';
import { createChart, ColorType, CrosshairMode } from 'lightweight-charts';
import { calculateEMA, calculateSMA } from '../utils/indicators.js';

// ✅ **核心修復**: 使用更穩健的、基於 Map 的聚合演算法
function aggregateToWeekly(dailyData) {
  if (!dailyData || dailyData.length === 0) return [];

  const weeklyDataMap = new Map();

  dailyData.forEach(day => {
    const date = new Date(day.time * 1000);
    
    // 計算該日期所在週的週一零點時間戳 (UTC)
    const dayOfWeek = date.getUTCDay(); // Sunday = 0, Monday = 1, ...
    const diff = (dayOfWeek + 6) % 7; // Monday = 0, Tuesday = 1, ..., Sunday = 6
    
    const weekStart = new Date(date);
    weekStart.setUTCDate(date.getUTCDate() - diff);
    weekStart.setUTCHours(0, 0, 0, 0);
    
    const weekTimestamp = Math.floor(weekStart.getTime() / 1000);

    if (!weeklyDataMap.has(weekTimestamp)) {
      // 這是本週的第一筆資料，創建新的週K線
      weeklyDataMap.set(weekTimestamp, {
        time: weekTimestamp,
        open: day.open,
        high: day.high,
        low: day.low,
        close: day.close,
        volume: day.volume,
      });
    } else {
      // 更新已有的週K線
      const week = weeklyDataMap.get(weekTimestamp);
      week.high = Math.max(week.high, day.high);
      week.low = Math.min(week.low, day.low);
      week.close = day.close; // 不斷更新收盤價為本週最新一日的收盤價
      week.volume += day.volume; // 累加成交量
      weeklyDataMap.set(weekTimestamp, week);
    }
  });

  // 從 Map 中取出所有值，並按時間排序以確保順序正確
  return Array.from(weeklyDataMap.values()).sort((a, b) => a.time - b.time);
}

const StockChart = ({ stockData = [], stockCode, height = 480 }) => {
  const theme = useTheme();
  const chartContainerRef = useRef(null);

  // 日/週 K線模式狀態
  const [candleMode, setCandleMode] = useState('day'); // 'day' 或 'week'
  const [indicators, setIndicators] = useState({
    ema6: true, ema12: true, ema24: true,
    sma50: true, sma150: true, sma200: true,
  });

  const handleModeChange = useCallback((mode) => {
    if (mode) {
      setCandleMode(mode);
    }
  }, []);

  // 根據模式選擇要顯示的資料
  const displayedData = useMemo(() => {
    if (candleMode === 'week') {
        return aggregateToWeekly(stockData);
    }
    return stockData;
  }, [stockData, candleMode]);

  useLayoutEffect(() => {
    const chartContainer = chartContainerRef.current;
    if (!chartContainer || !displayedData || displayedData.length === 0) {
      if(chartContainer) chartContainer.innerHTML = '';
      return;
    }

    const sortedData = [...displayedData].sort((a, b) => a.time - b.time);
    const candleData = sortedData.map(item => ({
      time: item.time, open: item.open, high: item.high, low: item.low, close: item.close,
    }));
    const volumeData = sortedData.map(item => ({
      time: item.time, value: item.volume,
      color: item.close >= item.open ? alpha(theme.palette.success.main, 0.6) : alpha(theme.palette.error.main, 0.6),
    }));

    const chart = createChart(chartContainer, {
      width: chartContainer.clientWidth,
      height,
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
  }, [displayedData, indicators, theme, stockCode, height]);

  const handleIndicatorToggle = useCallback((indicatorKey) => {
    setIndicators(prev => ({ ...prev, [indicatorKey]: !prev[indicatorKey] }));
  }, []);

  return (
    <Box height="100%" display="flex" flexDirection="column">
      <Paper elevation={0} sx={{ p: 1, mb: 1, flexShrink: 0, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap' }}>
        <ButtonGroup size="small" variant="outlined">
          <Button
            variant={candleMode === 'day' ? 'contained' : 'outlined'}
            onClick={() => handleModeChange('day')}
          >
            日K
          </Button>
          <Button
            variant={candleMode === 'week' ? 'contained' : 'outlined'}
            onClick={() => handleModeChange('week')}
          >
            週K
          </Button>
        </ButtonGroup>
        <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
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
      <Box
        ref={chartContainerRef}
        sx={{
          width: '100%',
          height,
          flexGrow: 1,
        }}
      />
      {(!displayedData || displayedData.length === 0) && (
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
