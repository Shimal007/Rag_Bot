// src/App.js

import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import { Bot, User, CornerDownLeft, Loader, Book } from 'lucide-react';
import RAGBot from './pages/Ragbot';
// You might need to add highlight.js CSS in your public/index.html or main CSS file
// e.g., <link rel="stylesheet" href="//cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/atom-one-dark.min.css">

function App() {
  
  return (
    <>
    <RAGBot />
    </>
  )
  }

export default App;