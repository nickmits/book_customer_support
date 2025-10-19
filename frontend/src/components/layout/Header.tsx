import React from 'react';
import { AppBar, Toolbar, Typography, Box } from '@mui/material';
import MenuBookIcon from '@mui/icons-material/MenuBook';

export const Header: React.FC = () => {
  return (
    <AppBar
      position="static"
      elevation={2}
      sx={{
        bgcolor: 'primary.main',
        color: 'primary.contrastText',
      }}
    >
      <Toolbar>
        <MenuBookIcon sx={{ mr: 2, fontSize: 32 }} />
        <Box sx={{ textAlign: 'center',  }}>
          <Typography variant="h5" component="h1" sx={{ fontWeight: 600 }}>
            Nick Mits
          </Typography>
          <Typography variant="caption" sx={{ opacity: 0.9 }}>
           Customer Support Chatbot
          </Typography>
        </Box>
      </Toolbar>
    </AppBar>
  );
};
