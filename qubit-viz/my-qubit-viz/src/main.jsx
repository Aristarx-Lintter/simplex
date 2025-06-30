// Inside src/main.jsx (or similar entry point)
import React from 'react'
import ReactDOM from 'react-dom/client'
import QubitBlochOriginViz from './App.jsx' // Make sure this points to your App.jsx
import './index.css' // Default styling if present

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <QubitBlochOriginViz />
  </React.StrictMode>,
)