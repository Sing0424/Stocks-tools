import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  Container, Typography, Paper, Alert, Snackbar, CircularProgress, Fab,
  Tooltip, AppBar, Toolbar, IconButton, useMediaQuery, Stack, LinearProgress, Box
} from '@mui/material'; // <--- Box 已加進來
import {
  Refresh as RefreshIcon, ArrowBack as ArrowBackIcon, ArrowForward as ArrowForwardIcon,
  DarkMode as DarkModeIcon, LightMode as LightModeIcon, CheckCircle as CheckCircleIcon, Error as ErrorIcon
} from '@mui/icons-material';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import { CssBaseline } from '@mui/material';
import Papa from 'papaparse';
import StockTable from './components/StockTable.jsx';
import StockChart from './components/StockChart.jsx';
import './App.css';

const createAppTheme = (mode) => createTheme({
  palette: {
    mode,
    primary: { main: mode === 'dark' ? '#90caf9' : '#1976d2' },
    background: {
      default: mode === 'dark' ? '#121212' : '#f5f5f5',
      paper: mode === 'dark' ? '#1e1e1e' : '#fff',
    },
  },
  typography: { fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif' },
});

function App() {
  const [listData, setListData] = useState([]);
  const [priceData, setPriceData] = useState([]);
  const [selectedSymbol, setSelectedSymbol] = useState(null);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [loadingStatus, setLoadingStatus] = useState({
    screenResults: 'pending',
    priceData: 'pending'
  });
  const [darkMode, setDarkMode] = useState(() => localStorage.getItem('darkMode') === 'true');
  const theme = useMemo(() => createAppTheme(darkMode ? 'dark' : 'light'), [darkMode]);
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  const toggleDarkMode = useCallback(() => {
    setDarkMode(prev => {
      const next = !prev;
      localStorage.setItem('darkMode', JSON.stringify(next));
      return next;
    });
  }, []);

  useEffect(() => {
    const loadAllData = async () => {
      try {
        setLoadingStatus(s => ({ ...s, screenResults: 'loading' }));
        const response = await fetch('/data/screenResults.csv');
        if (!response.ok) throw new Error(`HTTP ${response.status} for screenResults.csv`);
        const csvText = await response.text();
        Papa.parse(csvText, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          transformHeader: h => h.trim(),
          complete: (results) => {
            const mappedData = results.data
              .filter(r => r && r.symbol)
              .map(r => ({
                id: r.symbol,
                symbol: r.symbol,
                currentPrice: parseFloat(r.price) || 0,
                rsRank: parseInt(r.rs_rank) || 0,
                high52Week: parseFloat(r.high_52w) || 0,
                low52Week: parseFloat(r.low_52w) || 0,
              }));
            setListData(mappedData);
            setLoadingStatus(s => ({ ...s, screenResults: 'success' }));
            if (mappedData.length > 0) {
              setSelectedSymbol(mappedData[0].symbol);
              setSelectedIndex(0);
            }
          },
          error: (err) => { throw err; }
        });
      } catch(e) {
        setError(`載入 screenResults.csv 失敗: ${e.message}`);
        setLoadingStatus(s => ({ ...s, screenResults: 'error' }));
      }

      try {
        setLoadingStatus(s => ({ ...s, priceData: 'loading' }));
        const response = await fetch('/data/consolidated_price_data.csv');
        if (!response.ok) throw new Error(`HTTP ${response.status} for consolidated_price_data.csv`);
        const csvText = await response.text();
        Papa.parse(csvText, {
          worker: true,
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          complete: (results) => {
            const cleanData = results.data.map(row =>
              Object.entries(row).reduce((acc, [key, value]) => {
                acc[key.trim()] = value;
                return acc;
              }, {})
            );
            const validData = cleanData
              .filter(r => r && r.Symbol && r.Date && r.Close)
              .map(r => ({
                Symbol: r.Symbol,
                time: Math.floor(new Date(r.Date).getTime() / 1000),
                open: +r.Open, high: +r.High, low: +r.Low, close: +r.Close, volume: +r.Volume,
              }))
              .filter(r => !isNaN(r.time) && r.close > 0)
              .sort((a, b) => a.time - b.time);
            setPriceData(validData);
            setLoadingStatus(s => ({ ...s, priceData: 'success' }));
          },
          error: (err) => { throw err; }
        });
      } catch (e) {
         setError(`載入 consolidated_price_data.csv 失敗: ${e.message}`);
         setLoadingStatus(s => ({ ...s, priceData: 'error' }));
      }
    };

    loadAllData();
  }, []);

  useEffect(() => {
    if (loadingStatus.screenResults !== 'pending' && loadingStatus.priceData !== 'pending') {
      setLoading(false);
    }
  }, [loadingStatus]);

  const priceIndex = useMemo(() => {
    const m = new Map();
    priceData.forEach(r => {
      if (!m.has(r.Symbol)) m.set(r.Symbol, []);
      m.get(r.Symbol).push(r);
    });
    return m;
  }, [priceData]);

  const handleSelect = useCallback(sym => {
    const idx = listData.findIndex(r => r.symbol === sym);
    setSelectedSymbol(sym);
    setSelectedIndex(idx);
  }, [listData]);

  const navigate = useCallback(dir => {
    if (!listData.length) return;
    let idx = selectedIndex;
    if (dir === 'next') idx = (idx + 1) % listData.length;
    else idx = idx <= 0 ? listData.length - 1 : idx - 1;
    const sym = listData[idx].symbol;
    setSelectedSymbol(sym);
    setSelectedIndex(idx);
  }, [listData, selectedIndex]);

  const handleRefresh = useCallback(() => window.location.reload(), []);
  const handleCloseErr = useCallback(() => setError(null), []);

  const LoadingIndicator = ({ status, label }) => {
    let icon;
    if (status === 'success') icon = <CheckCircleIcon color="success" fontSize="small" />;
    else if (status === 'error') icon = <ErrorIcon color="error" fontSize="small" />;
    else icon = <CircularProgress size={16} />;
    return (
      <span className="loading-indicator">
        {icon}
        <span className="loading-label">{label}</span>
      </span>
    );
  };

  if (loading) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline/>
        <div className="app-loading-root">
          <CircularProgress size={60}/>
          <Typography variant="h6" color="textSecondary">載入資料中…</Typography>
          <Paper elevation={2} className="app-loading-paper">
            <Typography variant="subtitle2" gutterBottom>載入進度</Typography>
            <Stack spacing={2}>
              <Box>
                <LoadingIndicator status={loadingStatus.screenResults} label="篩選結果"/>
                {loadingStatus.screenResults === 'loading' && <LinearProgress sx={{ mt: 1 }}/>}
              </Box>
              <Box>
                <LoadingIndicator status={loadingStatus.priceData} label="價格資料"/>
                {loadingStatus.priceData === 'loading' && <Typography variant="caption" color="text.secondary">正在背景解析，請稍候...</Typography>}
                <LinearProgress variant="determinate" value={loadingStatus.priceData === 'success' ? 100 : (loadingStatus.priceData === 'pending' ? 0 : 50)} sx={{ mt: 1 }}/>
              </Box>
            </Stack>
          </Paper>
        </div>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline/>
      <AppBar position="sticky" elevation={1}>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>股票分析檢視器</Typography>
          {selectedSymbol && (
            <div className="toolbar-arrows">
              <Tooltip title="上一支"><IconButton color="inherit" onClick={()=>navigate('prev')} disabled={listData.length === 0}><ArrowBackIcon/></IconButton></Tooltip>
              <Tooltip title="下一支"><IconButton color="inherit" onClick={()=>navigate('next')} disabled={listData.length === 0}><ArrowForwardIcon/></IconButton></Tooltip>
            </div>
          )}
          <Tooltip title={darkMode?'淺色模式':'深色模式'}><IconButton color="inherit" onClick={toggleDarkMode}>{darkMode?<LightModeIcon/>:<DarkModeIcon/>}</IconButton></Tooltip>
        </Toolbar>
      </AppBar>
      <Container maxWidth="xl" className="main-container">
        <Paper elevation={1} className="status-paper">
          <Typography variant="subtitle2" gutterBottom>資料載入狀態</Typography>
          <div className="status-row">
            <LoadingIndicator status={loadingStatus.screenResults} label={`篩選結果: ${listData.length} 支`}/>
            <LoadingIndicator status={loadingStatus.priceData} label={`價格資料: ${priceData.length} 筆`}/>
          </div>
        </Paper>
        <div className={isMobile ? "main-flex mobile" : "main-flex"}>
          {/* 篩選結果（可捲動） */}
          <Paper elevation={2} className={isMobile ? "left-panel mobile" : "left-panel"}>
            <Typography variant="h6" gutterBottom>
              篩選結果 ({listData.length} 支)
            </Typography>
            <div className="table-scroll">
              <StockTable rows={listData} onSelect={handleSelect} selectedSymbol={selectedSymbol}/>
            </div>
          </Paper>
          {/* 圖表區（自適應寬度） */}
          <div className="chart-panel">
            <Paper elevation={2} className="chart-paper">
              <Typography variant="h6" gutterBottom>
                {selectedSymbol} - 技術分析 ({priceIndex.get(selectedSymbol)?.length||0} 點)
              </Typography>
              <StockChart
                stockCode={selectedSymbol}
                stockData={priceIndex.get(selectedSymbol)||[]}
                height={isMobile ? 260 : 480}
              />
            </Paper>
          </div>
        </div>
        <Tooltip title="重新載入"><Fab color="primary" onClick={handleRefresh} className="fab-refresh"><RefreshIcon/></Fab></Tooltip>
        <Snackbar open={!!error} autoHideDuration={6000} onClose={handleCloseErr} anchorOrigin={{ vertical:'bottom', horizontal:'left' }}>
          <Alert onClose={handleCloseErr} severity="error" variant="filled" sx={{ width:'100%' }}>{error}</Alert>
        </Snackbar>
      </Container>
    </ThemeProvider>
  );
}

export default App;
