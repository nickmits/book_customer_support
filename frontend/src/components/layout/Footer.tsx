import React from 'react';
import { Box, Typography, Link } from '@mui/material';

export const Footer: React.FC = () => {
  return (
    <Box
      component="footer"
      sx={{
        py: 2,
        px: 3,
        mt: 'auto',
        bgcolor: 'background.paper',
        borderTop: '1px solid',
        borderColor: 'divider',
        textAlign: 'center',
      }}
    >
      <Typography variant="body2" color="text.secondary">
        Powered by AI |{' '}
        <Link
          href="#"
          color="primary"
          underline="hover"
          sx={{ fontWeight: 500 }}
        >
          Privacy Policy
        </Link>
      </Typography>
    </Box>
  );
};
