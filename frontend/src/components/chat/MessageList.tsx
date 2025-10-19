import React, { useEffect, useRef } from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import { ChatMessage } from './ChatMessage';
import type { Message } from '../../types';

interface MessageListProps {
  messages: Message[];
  isLoading?: boolean;
}

export const MessageList: React.FC<MessageListProps> = ({ messages, isLoading = false }) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <Box
      sx={{
        flexGrow: 1,
        overflow: 'auto',
        py: 2,
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {messages.length === 0 ? (
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            px: 3,
            textAlign: 'center',
          }}
        >
          <Typography variant="h5" color="text.secondary" gutterBottom>
            Welcome to the Bookstore
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Ask me anything about books, authors, or our collection!
          </Typography>
        </Box>
      ) : (
        <>
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}
          {isLoading && (
            <Box
              sx={{
                display: 'flex',
                justifyContent: 'flex-start',
                px: 2,
                mb: 2,
              }}
            >
              <Box
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1,
                  bgcolor: 'background.paper',
                  p: 2,
                  borderRadius: 2,
                  boxShadow: '0 2px 8px rgba(62, 39, 35, 0.1)',
                }}
              >
                <CircularProgress size={16} sx={{ color: 'primary.main' }} />
                <Typography variant="body2" color="text.secondary">
                  Searching our shelves...
                </Typography>
              </Box>
            </Box>
          )}
        </>
      )}
      <div ref={messagesEndRef} />
    </Box>
  );
};
