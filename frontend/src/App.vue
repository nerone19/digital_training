<template>
  <div id="app">
    <div class="chat-container">
      <div class="header">
        <h1>🎬 YouTube RAG Chatbot</h1>
        <p>Ask questions about your processed YouTube videos</p>
      </div>
      
      <div class="chat-messages" ref="messagesContainer">
        <div v-if="messages.length === 0" class="welcome-message">
          <div class="welcome-icon">💬</div>
          <h3>Welcome to YouTube RAG Chat!</h3>
          <p>Start by asking a question about your processed videos.</p>
        </div>
        
        <div
          v-for="(message, index) in messages"
          :key="index"
          :class="['message', message.type]"
        >
          <div class="message-content">
            <div class="message-header">
              <span class="sender">{{ message.type === 'user' ? 'You' : 'AI Assistant' }}</span>
              <span class="timestamp">{{ formatTime(message.timestamp) }}</span>
            </div>
            <div class="message-text" v-html="formatMessage(message.text)"></div>
            <div v-if="message.searchResults && message.searchResults.length > 0" class="search-results">
              <details>
                <summary>📊 Source References ({{ message.searchResults.length }})</summary>
                <div class="results-list">
                  <div
                    v-for="(result, idx) in message.searchResults"
                    :key="idx"
                    class="result-item"
                  >
                    <div class="result-score">Score: {{ result.score.toFixed(3) }}</div>
                    <div class="result-text">{{ result.text }}</div>
                    <div v-if="result.url" class="result-url">
                      <a :href="result.url" target="_blank">🔗 Source Video</a>
                    </div>
                  </div>
                </div>
              </details>
            </div>
          </div>
        </div>
        
        <!-- Loading indicator -->
        <div v-if="isLoading" class="message ai loading">
          <div class="message-content">
            <div class="message-header">
              <span class="sender">AI Assistant</span>
              <span class="timestamp">{{ formatTime(new Date()) }}</span>
            </div>
            <div class="loading-indicator">
              <div class="spinner"></div>
              <span>Thinking...</span>
            </div>
          </div>
        </div>
      </div>
      
      <div class="chat-input">
        <form @submit.prevent="sendMessage" class="input-form">
          <div class="input-group">
            <input
              v-model="currentMessage"
              type="text"
              placeholder="Ask a question about your videos..."
              :disabled="isLoading"
              class="message-input"
              autofocus
            />
            <button
              type="submit"
              :disabled="!currentMessage.trim() || isLoading"
              class="send-button"
            >
              <span v-if="!isLoading">Send</span>
              <div v-else class="button-spinner"></div>
            </button>
          </div>
          <div class="input-options">
            <label class="toggle-option">
              <input
                v-model="useRAG"
                type="checkbox"
              />
              <span>Use RAG (recommended)</span>
            </label>
          </div>
        </form>
      </div>
      
      <!-- Status indicator -->
      <div class="status-bar">
        <div class="status-item">
          <span class="status-dot" :class="{ 'online': apiStatus.connected }"></span>
          API: {{ apiStatus.connected ? 'Connected' : 'Disconnected' }}
        </div>
        <div v-if="apiStatus.processedVideos" class="status-item">
          📹 {{ apiStatus.processedVideos }} videos processed
        </div>
        <div v-if="sessionId" class="status-item">
          💬 Session: {{ sessionId.substring(0, 8) }}...
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'App',
  data() {
    return {
      messages: [],
      currentMessage: '',
      isLoading: false,
      useRAG: true,
      sessionId: null, // Track the current session ID
      apiStatus: {
        connected: false,
        processedVideos: 0
      }
    }
  },
  
  mounted() {
    this.checkApiStatus()
  },
  
  methods: {
    async checkApiStatus() {
      try {
        const response = await axios.get('/api/status')
        this.apiStatus.connected = true
        this.apiStatus.processedVideos = response.data.data?.processed_videos || 0
      } catch (error) {
        this.apiStatus.connected = false
        console.error('API connection failed:', error)
      }
    },
    
    async sendMessage() {
      if (!this.currentMessage.trim() || this.isLoading) return
      
      const userMessage = {
        type: 'user',
        text: this.currentMessage,
        timestamp: new Date()
      }
      
      this.messages.push(userMessage)
      const question = this.currentMessage
      this.currentMessage = ''
      this.isLoading = true
      
      this.scrollToBottom()
      
      try {
        let response;
        
        // Use follow-up-query endpoint if we have a session, otherwise use query
        if (this.sessionId) {
          response = await axios.post('/api/follow-up-query', {
            question: question,
            use_rag: this.useRAG,
            session_id: this.sessionId
          })
        } else {
          response = await axios.post('/api/query', {
            question: question,
            use_rag: this.useRAG
          })
          // Store session ID from first query
          this.sessionId = response.data.session_id
        }
        
        const aiMessage = {
          type: 'ai',
          text: response.data.answer,
          timestamp: new Date(),
          searchResults: response.data.search_results || []
        }
        
        this.messages.push(aiMessage)
        
      } catch (error) {
        const errorMessage = {
          type: 'ai',
          text: `Sorry, I encountered an error: ${error.response?.data?.detail || error.message}`,
          timestamp: new Date(),
          isError: true
        }
        
        this.messages.push(errorMessage)
      } finally {
        this.isLoading = false
        this.scrollToBottom()
      }
    },
    
    formatMessage(text) {
      // Simple formatting for better readability
      return text
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
    },
    
    formatTime(date) {
      return new Intl.DateTimeFormat('en-US', {
        hour: '2-digit',
        minute: '2-digit'
      }).format(date)
    },
    
    scrollToBottom() {
      this.$nextTick(() => {
        const container = this.$refs.messagesContainer
        if (container) {
          container.scrollTop = container.scrollHeight
        }
      })
    }
  }
}
</script>

<style scoped>
#app {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
}

.chat-container {
  width: 100%;
  max-width: 800px;
  height: 80vh;
  background: white;
  border-radius: 16px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 24px;
  text-align: center;
}

.header h1 {
  margin-bottom: 8px;
  font-size: 24px;
  font-weight: 600;
}

.header p {
  opacity: 0.9;
  font-size: 14px;
}

.chat-messages {
  flex: 1;
  padding: 20px;
  overflow-y: auto;
  background: #f8fafc;
}

.welcome-message {
  text-align: center;
  padding: 60px 20px;
  color: #64748b;
}

.welcome-icon {
  font-size: 48px;
  margin-bottom: 16px;
}

.welcome-message h3 {
  margin-bottom: 8px;
  color: #334155;
}

.message {
  margin-bottom: 16px;
  animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.message.user .message-content {
  margin-left: auto;
  margin-right: 0;
  background: #667eea;
  color: white;
}

.message.ai .message-content {
  margin-left: 0;
  margin-right: auto;
  background: white;
  border: 1px solid #e2e8f0;
}

.message-content {
  max-width: 70%;
  padding: 16px;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.message-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  font-size: 12px;
  opacity: 0.7;
}

.sender {
  font-weight: 600;
}

.message-text {
  line-height: 1.5;
}

.search-results {
  margin-top: 12px;
  font-size: 12px;
}

.search-results details {
  background: rgba(0, 0, 0, 0.05);
  border-radius: 6px;
  padding: 8px;
}

.search-results summary {
  cursor: pointer;
  font-weight: 600;
  margin-bottom: 8px;
}

.result-item {
  background: white;
  padding: 8px;
  margin: 4px 0;
  border-radius: 4px;
  border-left: 3px solid #667eea;
}

.result-score {
  font-weight: 600;
  color: #667eea;
  margin-bottom: 4px;
}

.result-text {
  margin-bottom: 4px;
}

.result-url a {
  color: #667eea;
  text-decoration: none;
}

.loading-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #64748b;
}

.spinner {
  width: 16px;
  height: 16px;
  border: 2px solid #e2e8f0;
  border-top: 2px solid #667eea;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.chat-input {
  padding: 20px;
  background: white;
  border-top: 1px solid #e2e8f0;
}

.input-form {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.input-group {
  display: flex;
  gap: 12px;
}

.message-input {
  flex: 1;
  padding: 12px 16px;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  font-size: 14px;
  transition: border-color 0.2s;
}

.message-input:focus {
  outline: none;
  border-color: #667eea;
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.send-button {
  padding: 12px 24px;
  background: #667eea;
  color: white;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  cursor: pointer;
  transition: background-color 0.2s;
  min-width: 80px;
}

.send-button:hover:not(:disabled) {
  background: #5a67d8;
}

.send-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.button-spinner {
  width: 16px;
  height: 16px;
  border: 2px solid transparent;
  border-top: 2px solid white;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto;
}

.input-options {
  display: flex;
  align-items: center;
  gap: 16px;
  font-size: 14px;
}

.toggle-option {
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
}

.toggle-option input[type="checkbox"] {
  margin: 0;
}

.status-bar {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 8px 20px;
  background: #f1f5f9;
  border-top: 1px solid #e2e8f0;
  font-size: 12px;
  color: #64748b;
}

.status-item {
  display: flex;
  align-items: center;
  gap: 6px;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #ef4444;
}

.status-dot.online {
  background: #10b981;
}

/* Mobile responsiveness */
@media (max-width: 768px) {
  #app {
    padding: 10px;
  }
  
  .chat-container {
    height: 90vh;
  }
  
  .message-content {
    max-width: 85%;
  }
  
  .header {
    padding: 16px;
  }
  
  .header h1 {
    font-size: 20px;
  }
}
</style>