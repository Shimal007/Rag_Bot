import React, { useState, useEffect, useRef } from 'react';
import { Upload, Send, FileText, MessageCircle, Settings, Loader2, Download, Trash2, Plus, History } from 'lucide-react';

const RAGBot = () => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const [availableModels, setAvailableModels] = useState(['gemma3:1b', 'mistral:7b-instruct-q2_K']);
  const [selectedModel, setSelectedModel] = useState('gemma3:1b');
  const [documents, setDocuments] = useState([]);
  const [uploadProgress, setUploadProgress] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [conversations, setConversations] = useState([]);
  const [selectedConversationId, setSelectedConversationId] = useState(null);
  const [isLoadingConversations, setIsLoadingConversations] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    fetchModels();
    fetchDocuments();
    fetchConversations();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/models');
      const data = await response.json();
      const models = data.models || ['gemma3:1b', 'mistral:7b-instruct-q2_K'];
      setAvailableModels(models);
      
      // If the current selected model is not in the available models, switch to the first available one
      if (models.length > 0 && !models.includes(selectedModel)) {
        setSelectedModel(models[0]);
      }
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  const fetchDocuments = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/documents');
      const data = await response.json();
      setDocuments(data.documents || []);
    } catch (error) {
      console.error('Error fetching documents:', error);
    }
  };

  const fetchConversations = async () => {
    setIsLoadingConversations(true);
    try {
      const response = await fetch('http://localhost:5000/api/conversations');
      const data = await response.json();
      setConversations(data.conversations || []);
    } catch (error) {
      console.error('Error fetching conversations:', error);
    }
    setIsLoadingConversations(false);
  };

  const loadConversationHistory = async (conversationId) => {
    try {
      const response = await fetch(`http://localhost:5000/api/conversations/${conversationId}`);
      const data = await response.json();
      
      const conversationMessages = data.conversations.map(conv => ({
        type: 'user',
        content: conv.query,
        timestamp: new Date(conv.timestamp),
        model: conv.model
      }));
      
      // Add bot responses
      const botMessages = data.conversations.map(conv => ({
        type: 'bot',
        content: conv.response,
        sources: conv.sources || [],
        model: conv.model,
        timestamp: new Date(conv.timestamp)
      }));
      
      // Interleave user and bot messages
      const allMessages = [];
      for (let i = 0; i < conversationMessages.length; i++) {
        allMessages.push(conversationMessages[i]);
        if (botMessages[i]) {
          allMessages.push(botMessages[i]);
        }
      }
      
      setMessages(allMessages);
      setConversationId(conversationId);
      setSelectedConversationId(conversationId);
    } catch (error) {
      console.error('Error loading conversation history:', error);
      setMessages(prev => [...prev, {
        type: 'error',
        content: `Error loading conversation history: ${error.message}`,
        timestamp: new Date()
      }]);
    }
  };

  const startNewChat = () => {
    setMessages([]);
    setConversationId(null);
    setSelectedConversationId(null);
    setInputMessage('');
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setUploadProgress(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5000/api/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (response.ok) {
        setMessages(prev => [...prev, {
          type: 'system',
          content: `✅ Successfully uploaded and processed: ${file.name}`,
          timestamp: new Date()
        }]);
        fetchDocuments();
      } else {
        setMessages(prev => [...prev, {
          type: 'error',
          content: `❌ Upload failed: ${data.error}`,
          timestamp: new Date()
        }]);
      }
    } catch (error) {
      setMessages(prev => [...prev, {
        type: 'error',
        content: `❌ Upload error: ${error.message}`,
        timestamp: new Date()
      }]);
    }

    setUploadProgress(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      type: 'user',
      content: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:5000/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: inputMessage,
          model: selectedModel,
          conversation_id: conversationId
        }),
      });

      const data = await response.json();
      
      if (response.ok) {
        setConversationId(data.conversation_id);
        setSelectedConversationId(data.conversation_id);
        setMessages(prev => [...prev, {
          type: 'bot',
          content: data.response,
          sources: data.sources,
          model: selectedModel,
          timestamp: new Date()
        }]);
        // Refresh conversations list after sending a message
        fetchConversations();
      } else {
        setMessages(prev => [...prev, {
          type: 'error',
          content: `Error: ${data.error}`,
          timestamp: new Date()
        }]);
      }
    } catch (error) {
      setMessages(prev => [...prev, {
        type: 'error',
        content: `Network error: ${error.message}`,
        timestamp: new Date()
      }]);
    }

    setIsLoading(false);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const clearConversation = () => {
    setMessages([]);
    setConversationId(null);
    setSelectedConversationId(null);
    fetchConversations();
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const MessageComponent = ({ message }) => {
    const baseClasses = "max-w-[80%] p-4 rounded-lg shadow-sm";
    
    switch (message.type) {
      case 'user':
        return (
          <div className="flex justify-end mb-4">
            <div className={`${baseClasses} bg-blue-500 text-white ml-auto`}>
              <p className="whitespace-pre-wrap">{message.content}</p>
              <span className="text-xs opacity-75 mt-2 block">
                {formatTimestamp(message.timestamp)}
              </span>
            </div>
          </div>
        );
      
      case 'bot':
        return (
          <div className="flex justify-start mb-4">
            <div className={`${baseClasses} bg-gray-100 text-gray-800`}>
              <p className="whitespace-pre-wrap mb-2">{message.content}</p>
              {message.sources && message.sources.length > 0 && (
                <div className="border-t pt-2 mt-2">
                  <p className="text-xs font-medium text-gray-600 mb-1">Sources:</p>
                  <div className="flex flex-wrap gap-1">
                    {message.sources.map((source, idx) => (
                      <span key={idx} className="inline-block bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded">
                        {source}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              <div className="flex justify-between items-center mt-2 pt-2 border-t">
                <span className="text-xs text-gray-500">
                  Model: {message.model}
                </span>
                <span className="text-xs text-gray-500">
                  {formatTimestamp(message.timestamp)}
                </span>
              </div>
            </div>
          </div>
        );
      
      case 'system':
        return (
          <div className="flex justify-center mb-4">
            <div className="bg-green-100 text-green-800 px-4 py-2 rounded-lg text-sm">
              {message.content}
            </div>
          </div>
        );
      
      case 'error':
        return (
          <div className="flex justify-center mb-4">
            <div className="bg-red-100 text-red-800 px-4 py-2 rounded-lg text-sm">
              {message.content}
            </div>
          </div>
        );
      
      default:
        return null;
    }
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-gray-200">
          <h1 className="text-xl font-bold text-gray-800 flex items-center gap-2">
            <MessageCircle className="w-6 h-6 text-blue-500" />
            RAG Bot
          </h1>
          <p className="text-sm text-gray-600 mt-1">Local AI Q&A Assistant</p>
        </div>

        {/* Upload Section */}
        <div className="p-4 border-b border-gray-200">
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploadProgress}
            className="w-full flex items-center justify-center gap-2 bg-blue-500 hover:bg-blue-600 disabled:bg-blue-300 text-white px-4 py-2 rounded-lg transition-colors"
          >
            {uploadProgress ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Upload className="w-4 h-4" />
            )}
            {uploadProgress ? 'Processing...' : 'Upload Document'}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".txt,.pdf,.docx"
            onChange={handleFileUpload}
            className="hidden"
          />
        </div>

        {/* Documents List */}
        <div className="flex-1 overflow-y-auto p-4">
          <h3 className="font-medium text-gray-700 mb-2 flex items-center gap-2">
            <FileText className="w-4 h-4" />
            Documents ({documents.length})
          </h3>
          <div className="space-y-2">
            {documents.map((doc) => (
              <div key={doc.id} className="bg-gray-50 p-3 rounded-lg">
                <p className="text-sm font-medium text-gray-700 truncate">
                  {doc.filename}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  {new Date(doc.created_at).toLocaleDateString()}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Conversations List */}
        <div className="flex-1 overflow-y-auto p-4 border-t border-gray-200">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-medium text-gray-700 flex items-center gap-2">
              <History className="w-4 h-4" />
              Conversations
            </h3>
            <button
              onClick={startNewChat}
              className="p-1 hover:bg-gray-100 rounded-md transition-colors"
              title="New Chat"
            >
              <Plus className="w-4 h-4 text-gray-600" />
            </button>
          </div>
          
          {isLoadingConversations ? (
            <div className="flex items-center justify-center py-4">
              <Loader2 className="w-4 h-4 animate-spin text-gray-500" />
            </div>
          ) : conversations.length === 0 ? (
            <div className="text-center py-4 text-gray-500 text-sm">
              No conversations yet
            </div>
          ) : (
            <div className="space-y-2">
              {conversations.map((conv) => (
                <div
                  key={conv.conversation_id}
                  onClick={() => loadConversationHistory(conv.conversation_id)}
                  className={`p-3 rounded-lg cursor-pointer transition-colors ${
                    selectedConversationId === conv.conversation_id
                      ? 'bg-blue-100 border border-blue-200'
                      : 'bg-gray-50 hover:bg-gray-100'
                  }`}
                >
                  <p className="text-sm font-medium text-gray-700 truncate">
                    {conv.first_query || 'New conversation'}
                  </p>
                  <div className="flex items-center justify-between mt-1">
                    <p className="text-xs text-gray-500">
                      {new Date(conv.first_timestamp).toLocaleDateString()}
                    </p>
                    <span className="text-xs text-gray-400">
                      {conv.model}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Settings */}
        <div className="p-4 border-t border-gray-200">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="flex items-center gap-2 text-gray-600 hover:text-gray-800 mb-3"
          >
            <Settings className="w-4 h-4" />
            Model Settings
          </button>
          
          {showSettings && (
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  AI Model
                </label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full p-2 border border-gray-300 rounded-md text-sm"
                >
                  {availableModels.map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                </select>
              </div>
              
              <button
                onClick={clearConversation}
                className="w-full flex items-center justify-center gap-2 bg-red-500 hover:bg-red-600 text-white px-3 py-2 rounded-md text-sm transition-colors"
              >
                <Trash2 className="w-4 h-4" />
                Clear Chat
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-6">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-500">
              <MessageCircle className="w-16 h-16 mb-4 text-gray-300" />
              <h2 className="text-xl font-medium mb-2">Welcome to RAG Bot!</h2>
              <p className="text-center max-w-md">
                Upload documents and start asking questions. I'll search through your documents and provide contextual answers using local AI models.
              </p>
              <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h3 className="font-medium text-blue-800 mb-2">📄 Supported Files</h3>
                  <p className="text-blue-700">PDF, DOCX, TXT files</p>
                </div>
                <div className="bg-green-50 p-4 rounded-lg">
                  <h3 className="font-medium text-green-800 mb-2">🤖 Local AI</h3>
                  <p className="text-green-700">Powered by Ollama models</p>
                </div>
              </div>
            </div>
          ) : (
            <>
              {messages.map((message, index) => (
                <MessageComponent key={index} message={message} />
              ))}
              {isLoading && (
                <div className="flex justify-start mb-4">
                  <div className="bg-gray-100 p-4 rounded-lg">
                    <div className="flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span className="text-gray-600">Thinking...</span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-200 p-4">
          <div className="flex gap-2">
            <div className="flex-1 relative">
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask a question about your documents..."
                className="w-full p-3 pr-12 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                rows="2"
                disabled={isLoading}
              />
              <button
                onClick={handleSendMessage}
                disabled={!inputMessage.trim() || isLoading}
                className="absolute right-2 bottom-2 p-2 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 text-white rounded-md transition-colors"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Press Enter to send, Shift+Enter for new line • Model: {selectedModel}
          </p>
        </div>
      </div>
    </div>
  );
};

export default RAGBot;