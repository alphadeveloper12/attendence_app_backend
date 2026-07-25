import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ConfigProvider } from 'antd'
import App from './App'
import { antdTheme } from './lib/antdTheme'
import './index.css'

/**
 * One shared React Query client. Server state (lists, stats, etc.) is cached
 * here; components read it via hooks. Defaults tuned for a dashboard: no refetch
 * on window focus (avoids surprise reloads), one retry, 30s default staleness.
 */
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
      staleTime: 30_000,
    },
  },
})

// NOTE: intentionally NOT wrapped in <React.StrictMode>. The Google Maps loader
// (@react-google-maps/api `useJsApiLoader`) is incompatible with StrictMode's
// mount→unmount→remount probe: the unmount cancels the script's onload callback
// so `isLoaded` never flips true and maps hang on "Loading map…". StrictMode's
// checks are dev-only, so dropping it has no effect on the shipped app.
ReactDOM.createRoot(document.getElementById('root')!).render(
  <ConfigProvider theme={antdTheme}>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </ConfigProvider>,
)
