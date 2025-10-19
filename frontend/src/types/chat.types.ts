export type MessageSender = 'user' | 'assistant';

export interface Message {
  id: string;
  content: string;
  sender: MessageSender;
  timestamp: Date;
  metadata?: {
    mode?: string;
    search_target?: string;
    routing_reasoning?: string;
  };
}

export interface ChatState {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
}

export interface SendMessageRequest {
  content: string;
}

export interface SendMessageResponse {
  message: Message;
}
