# Bookstore Customer Support Frontend

A simple, elegant chat application for bookstore customer support with a warm, literary aesthetic.

## Technology Stack

- **React 18+** with TypeScript
- **Material UI v7** (latest) - UI component library
- **Vite** - Build tool and dev server
- **Vitest** - Unit testing framework
- **Axios** - HTTP client

## Design Theme

The application features a cozy bookstore aesthetic with:
- Warm brown and sepia color palette
- Serif fonts (Playfair Display) for headers
- Sans-serif fonts (Open Sans) for readability
- Book-page-like message styling
- Gentle shadows and rounded corners

## Getting Started

### Prerequisites

- Node.js 18+ and npm

### Installation

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:5173`

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run test` - Run unit tests
- `npm run test:ui` - Run tests with UI
- `npm run test:coverage` - Run tests with coverage report
- `npm run lint` - Run ESLint
- `npm run format` - Format code with Prettier
- `npm run format:check` - Check code formatting

## Project Structure

```
src/
├── components/
│   ├── chat/              # Chat-related components
│   │   ├── ChatContainer.tsx
│   │   ├── ChatMessage.tsx
│   │   ├── ChatInput.tsx
│   │   └── MessageList.tsx
│   ├── common/            # Reusable components
│   └── layout/            # Layout components
│       ├── MainLayout.tsx
│       ├── Header.tsx
│       └── Footer.tsx
├── hooks/                 # Custom React hooks
├── theme/                 # MUI theme configuration
├── types/                 # TypeScript type definitions
├── services/              # API services
├── utils/                 # Utility functions
└── __tests__/             # Test files
```

## Features

- Real-time chat interface
- User and assistant message differentiation
- Auto-scroll to latest messages
- Loading indicators
- Responsive design (mobile-first)
- Accessibility (WCAG AA)
- Full TypeScript support

## Backend Integration

To connect to the Python backend:

1. Update the API endpoint in `src/services/api.ts`
2. The backend should be running on the configured port
3. Messages will be sent to the backend for processing

## Development Guidelines

Please refer to `.cursor/rules/frontend-rule.mdc` for comprehensive development guidelines including:
- Component structure
- TypeScript best practices
- Material UI usage
- Testing practices
- Accessibility requirements
- Code quality standards

## Testing

Run tests with:
```bash
npm run test
```

For UI-based test running:
```bash
npm run test:ui
```

## Building for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

## License

Private
