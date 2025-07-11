# YouTube RAG Chatbot Frontend

A Vue.js frontend application for the YouTube RAG system chatbot interface.

## Features

- 💬 **Chat Interface**: Clean, modern chat UI with message bubbles
- 🔄 **Loading Indicators**: Spinning animations during API calls
- 📊 **Source References**: Expandable search results from RAG queries
- 🎛️ **RAG Toggle**: Option to use RAG or direct LLM queries
- 📱 **Responsive Design**: Works on desktop and mobile devices
- 🔗 **API Integration**: Connects to FastAPI backend
- ⚡ **Real-time Status**: Shows API connection and processed video count

## Setup

1. **Install dependencies:**
   ```bash
   cd frontend
   npm install
   ```

2. **Start development server:**
   ```bash
   npm run dev
   ```

3. **Build for production:**
   ```bash
   npm run build
   ```

## Configuration

The frontend is configured to proxy API requests to `http://localhost:8000` (your FastAPI backend).

Make sure your FastAPI backend is running before starting the frontend.

## Usage

1. Start the FastAPI backend: `python app.py`
2. Start the frontend: `npm run dev`
3. Open your browser to `http://localhost:3000`
4. Start chatting with your YouTube RAG system!

## API Endpoints Used

- `GET /status` - Check API connection and system status
- `POST /query` - Send chat messages and receive responses

## Project Structure

```
frontend/
├── src/
│   ├── App.vue          # Main chat component
│   └── main.js          # Vue app initialization
├── index.html           # HTML template
├── package.json         # Dependencies and scripts
├── vite.config.js       # Vite configuration
└── README.md           # This file
```