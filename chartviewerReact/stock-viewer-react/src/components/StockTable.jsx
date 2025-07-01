import React, { useMemo, useCallback } from 'react';
import { Box, Typography, Chip, useTheme, alpha } from '@mui/material';
import { DataGrid } from '@mui/x-data-grid';

const StockTable = ({ rows = [], onSelect, selectedSymbol }) => {
  const theme = useTheme();

  const columns = useMemo(() => [
    { field: 'symbol', headerName: '股票代碼', width: 100 },
    { field: 'currentPrice', headerName: '現價', width: 100, type: 'number',
      valueFormatter: (value) => value ? `$${value.toFixed(2)}` : ''
    },
    { field: 'rsRank', headerName: 'RS 排名', width: 100, type: 'number',
      renderCell: (params) => {
        const value = params.value || 0;
        const color = value >= 80 ? 'success' : value >= 60 ? 'warning' : 'error';
        return <Chip label={value} size="small" color={color} variant="outlined" />;
      },
    },
  ], []);

  const handleRowClick = useCallback((params) => onSelect(params.row.symbol), [onSelect]);

  return (
    <Box sx={{ height: '100%', width: '100%' }}>
      <DataGrid
        rows={rows}
        columns={columns}
        onRowClick={handleRowClick}
        density="compact"
        hideFooter
        sx={{
          border: 'none',
          '& .MuiDataGrid-row': {
            cursor: 'pointer',
            '&:hover': {
              backgroundColor: alpha(theme.palette.action.hover, 0.5),
            },
            '&.Mui-selected': {
              backgroundColor: alpha(theme.palette.primary.main, 0.2),
              '&:hover': {
                backgroundColor: alpha(theme.palette.primary.main, 0.3),
              },
            },
          },
          '& .MuiDataGrid-cell:focus, & .MuiDataGrid-cell:focus-within': {
            outline: 'none',
          },
        }}
        getRowId={(row) => row.symbol}
        selectionModel={selectedSymbol ? [selectedSymbol] : []}
      />
    </Box>
  );
};

export default React.memo(StockTable);
