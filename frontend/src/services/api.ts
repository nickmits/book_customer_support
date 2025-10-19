import axios from 'axios';

// Configure the base URL for the backend API
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 120000, // 2 minutes - increased for AI agent processing
});

// Types for bookstore API
export interface Book {
  id?: number;
  title: string;
  author: string;
  isbn: string;
  genre: string;
  price: number;
  stock: number;
  description?: string;
  add_date?: string;
}

export interface BooksResponse {
  message: string;
  books: Book[];
}
export interface AIAgentRequest {
  query: string;
  request_type?: string;
}

export interface SearchResults {
  mode: 'langgraph_agent' | 'intelligent_response_fallback' | 'fallback';
  query: string;
  agent_available: boolean;
  search_target?: 'csv_metadata' | 'pdf_fulltext' | 'both' | 'write_post' | 'unknown';
  routing_reasoning?: string;
  timestamp: string;
}

export interface AIAgentResponse {
  response: string;
  search_results: SearchResults;
  final_response: string;
}
export interface BookStatsResponse {
  total_books: number;
  total_genres: number;
  genres: string[];
  total_inventory_value: number;
  average_price: number;
}

export interface WelcomeResponse {
  message: string;
  description: string;
  endpoints: any;
}

// API service functions for bookstore
export const bookstoreApi = {
  /**
   * Get welcome message
   */
  getWelcome: async (): Promise<WelcomeResponse> => {
    const response = await apiClient.get<WelcomeResponse>('/');
    return response.data;
  },

    // ... existing functions ...

  /**
   * Chat with AI agent
   */
  chatWithAI: async (request: AIAgentRequest): Promise<AIAgentResponse> => {
    const response = await apiClient.post<AIAgentResponse>('/ai-agent', request);
    return response.data;
  },
  
  /**
   * Get all books with optional filtering
   */
  getBooks: async (genre?: string, author?: string): Promise<BooksResponse> => {
    const params = new URLSearchParams();
    if (genre) params.append('genre', genre);
    if (author) params.append('author', author);
    
    const response = await apiClient.get<BooksResponse>(`/books?${params.toString()}`);
    return response.data;
  },

  /**
   * Get a specific book by ID
   */
  getBook: async (id: number): Promise<Book> => {
    const response = await apiClient.get<Book>(`/books/${id}`);
    return response.data;
  },

  /**
   * Add a new book
   */
  addBook: async (book: Omit<Book, 'id' | 'add_date'>): Promise<{ message: string; book: Book }> => {
    const response = await apiClient.post<{ message: string; book: Book }>('/books', book);
    return response.data;
  },

  /**
   * Update a book
   */
  updateBook: async (id: number, bookUpdate: Partial<Book>): Promise<{ message: string; book: Book }> => {
    const response = await apiClient.put<{ message: string; book: Book }>(`/books/${id}`, bookUpdate);
    return response.data;
  },



  /**
   * Get bookstore statistics
   */
  getStats: async (): Promise<BookStatsResponse> => {
    const response = await apiClient.get<BookStatsResponse>('/stats');
    return response.data;
  },
};