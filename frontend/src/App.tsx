import { ThemeProvider, CssBaseline } from '@mui/material';
import { MainLayout } from './components/layout/MainLayout';
import { ChatContainer } from './components/chat/ChatContainer';
import { bookstoreTheme } from './theme/theme';

function App() {
  return (
    <ThemeProvider theme={bookstoreTheme}>
      <CssBaseline />
      <MainLayout>
        <ChatContainer />
      </MainLayout>
    </ThemeProvider>
  );
}

export default App;
