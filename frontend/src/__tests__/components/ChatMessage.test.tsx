import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ChatMessage } from '../../components/chat/ChatMessage';
import type { Message } from '../../types';

describe('ChatMessage', () => {
  const mockUserMessage: Message = {
    id: '1',
    content: 'Hello, do you have any books by Jane Austen?',
    sender: 'user',
    timestamp: new Date('2024-01-01T12:00:00'),
  };

  const mockAssistantMessage: Message = {
    id: '2',
    content: 'Yes, we have several books by Jane Austen including Pride and Prejudice.',
    sender: 'assistant',
    timestamp: new Date('2024-01-01T12:00:05'),
  };

  it('renders user message correctly', () => {
    render(<ChatMessage message={mockUserMessage} />);
    expect(screen.getByText(mockUserMessage.content)).toBeInTheDocument();
  });

  it('renders assistant message correctly', () => {
    render(<ChatMessage message={mockAssistantMessage} />);
    expect(screen.getByText(mockAssistantMessage.content)).toBeInTheDocument();
  });

  it('displays timestamp', () => {
    render(<ChatMessage message={mockUserMessage} />);
    expect(screen.getByText(/12:00/)).toBeInTheDocument();
  });
});
