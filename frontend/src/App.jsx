import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import { Bot, User, CornerDownLeft, Loader, Book } from 'lucide-react';
import RAGBot from './pages/Ragbot';

function App() {
  
  return (
    <>
    <RAGBot />
    </>
  )
  }

export default App;
