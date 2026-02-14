import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { AppShell } from './components/layout/AppShell';
import { ChatPage } from './pages/ChatPage';
import { HealthPage } from './pages/HealthPage';
import { PaperDetailPage } from './pages/PaperDetailPage';
import { PapersPage } from './pages/PapersPage';
import { StatsPage } from './pages/StatsPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<ChatPage />} />
          <Route path="papers" element={<PapersPage />} />
          <Route path="papers/:arxivId" element={<PaperDetailPage />} />
          <Route path="stats" element={<StatsPage />} />
          <Route path="health" element={<HealthPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
