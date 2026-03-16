import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import * as echarts from 'echarts'
import { darkTheme } from './utils/echartTheme'
import App from './App'
import './index.css'

echarts.registerTheme('dark-premium', darkTheme)

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
)
