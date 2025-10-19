import React, { useState, useEffect } from 'react';
import { Paper, Alert, Snackbar } from '@mui/material';
import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';
import { bookstoreApi } from '../../services/api';
import type { Message } from '../../types';

export const ChatContainer: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Add welcome message on component mount
  useEffect(() => {
    const addWelcomeMessage = async () => {
      try {
        const welcome = await bookstoreApi.getWelcome();
        const welcomeMessage: Message = {
          id: Date.now().toString(),
          content: `${welcome.message}\n\n${welcome.description}\n\nI can help you with:\n• Browse our book inventory\n• Get bookstore statistics\n• Find specific books`,
          sender: 'assistant',
          timestamp: new Date(),
        };
        setMessages([welcomeMessage]);
      } catch (error) {
        console.error('Error getting welcome message:', error);
      }
    };

    addWelcomeMessage();
  }, []);

  const handleSendMessage = async (content: string) => {
    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      content,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      // Call the LangGraph AI agent
      // Let the backend's parse_request node determine the request_type intelligently
      const aiResponse = await bookstoreApi.chatWithAI({
        query: content,
      });

      // Check if agent is working
      const isAgentActive = aiResponse.search_results.mode === 'langgraph_agent';

      // Optional: Add routing info for debugging (can be removed in production)
      console.log('🤖 AI Agent Response:', {
        mode: aiResponse.search_results.mode,
        search_target: aiResponse.search_results.search_target,
        agent_available: aiResponse.search_results.agent_available,
      });

      // Create assistant message with the AI response
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: aiResponse.final_response,
        sender: 'assistant',
        timestamp: new Date(),
        metadata: {
          mode: aiResponse.search_results.mode,
          search_target: aiResponse.search_results.search_target,
          routing_reasoning: aiResponse.search_results.routing_reasoning,
        },
      };

      setMessages((prev) => [...prev, assistantMessage]);

      // Show a warning if falling back to non-LangGraph mode (optional)
      if (!isAgentActive) {
        console.warn('⚠️ AI Agent not active, using fallback mode');
      }
    } catch (error) {
      console.error('Error processing request:', error);
      setError('Failed to connect to the AI agent. Please make sure the backend server is running.');

      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: 'I apologize, but I encountered an error while processing your request. Please try again or check if the server is running.',
        sender: 'assistant',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCloseError = () => {
    setError(null);
  };

  return (
    <Paper
      elevation={0}
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        bgcolor: 'background.default',
        overflow: 'hidden',
      }}
    >
      <MessageList messages={messages} isLoading={isLoading} />
      <ChatInput onSendMessage={handleSendMessage} disabled={isLoading} />
      
      <Snackbar
        open={!!error}
        autoHideDuration={6000}
        onClose={handleCloseError}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={handleCloseError} severity="error" sx={{ width: '100%' }}>
          {error}
        </Alert>
      </Snackbar>
    </Paper>
  );
};