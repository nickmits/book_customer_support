import { createTheme } from '@mui/material/styles';

export const bookstoreTheme = createTheme({
  palette: {
    primary: {
      main: '#8B4513', // Saddle brown
      light: '#A0826D',
      dark: '#654321',
      contrastText: '#FFF8DC',
    },
    secondary: {
      main: '#800020', // Burgundy
      light: '#A0522D',
      dark: '#5C001A',
      contrastText: '#FAF0E6',
    },
    background: {
      default: '#FAF0E6', // Linen
      paper: '#FFF8DC', // Cornsilk
    },
    text: {
      primary: '#3E2723', // Dark brown
      secondary: '#4E342E', // Darker brown
    },
    divider: '#D2B48C', // Tan
  },
  typography: {
    fontFamily: '"Open Sans", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontFamily: '"Playfair Display", "Georgia", serif',
      fontWeight: 700,
      color: '#3E2723',
    },
    h2: {
      fontFamily: '"Playfair Display", "Georgia", serif',
      fontWeight: 600,
      color: '#3E2723',
    },
    h3: {
      fontFamily: '"Playfair Display", "Georgia", serif',
      fontWeight: 600,
      color: '#3E2723',
    },
    h4: {
      fontFamily: '"Playfair Display", "Georgia", serif',
      fontWeight: 500,
      color: '#4E342E',
    },
    h5: {
      fontFamily: '"Playfair Display", "Georgia", serif',
      fontWeight: 500,
      color: '#4E342E',
    },
    h6: {
      fontFamily: '"Playfair Display", "Georgia", serif',
      fontWeight: 500,
      color: '#4E342E',
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.5,
    },
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
        elevation1: {
          boxShadow: '0 2px 8px rgba(62, 39, 35, 0.1)',
        },
        elevation2: {
          boxShadow: '0 4px 12px rgba(62, 39, 35, 0.12)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
          borderRadius: 8,
        },
        contained: {
          boxShadow: '0 2px 4px rgba(139, 69, 19, 0.2)',
          '&:hover': {
            boxShadow: '0 4px 8px rgba(139, 69, 19, 0.3)',
          },
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            backgroundColor: '#FFFFFF',
            '&:hover fieldset': {
              borderColor: '#A0826D',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#8B4513',
            },
          },
        },
      },
    },
  },
});
