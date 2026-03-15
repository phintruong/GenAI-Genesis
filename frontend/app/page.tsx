'use client';

import { type PointerEvent as ReactPointerEvent, useCallback, useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import Sidebar from './components/Sidebar';
import { type GraphData, loadGraphFromCSV } from './lib/api';
import { Moon, Sun, Search, Loader2 } from 'lucide-react';

const NetworkGraph = dynamic(() => import('./components/NetworkGraph'), { ssr: false });
const SIDEBAR_MIN_WIDTH = 320;
const SIDEBAR_MAX_WIDTH = 640;

export default function Dashboard() {
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
  const [selectedAccount, setSelectedAccount] = useState<any>(null);
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [globalSearch, setGlobalSearch] = useState('');
  const [sidebarWidth, setSidebarWidth] = useState(384);
  const [isResizingSidebar, setIsResizingSidebar] = useState(false);
  const [viewMode, setViewMode] = useState<'3d' | '2d'>('3d');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load data from CSV files in public/data/
  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await loadGraphFromCSV();
      if (data.nodes.length > 0) {
        setGraphData(data);
      } else {
        setError('No data found. Run the backend pipeline first to generate CSV files.');
      }
    } catch {
      setError('Could not load CSV data. Run the backend pipeline first to generate CSV files.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleNodeClick = (node: any) => {
    setSelectedAccount((prev: any) => (prev?.id === node.id ? null : node));
  };

  const executeSearch = (term: string) => {
    if (!term.trim()) return;
    const foundNode = graphData.nodes.find(n => n.id.toLowerCase() === term.toLowerCase());

    if (foundNode) {
      setSelectedAccount(foundNode);
      setGlobalSearch('');
    } else {
      alert(`Account "${term}" not found in current network.`);
    }
  };

  useEffect(() => {
    if (!isResizingSidebar) return;

    const handlePointerMove = (event: globalThis.PointerEvent) => {
      const nextWidth = window.innerWidth - event.clientX;
      const clampedWidth = Math.min(SIDEBAR_MAX_WIDTH, Math.max(SIDEBAR_MIN_WIDTH, nextWidth));
      setSidebarWidth(clampedWidth);
    };

    const stopResizing = () => {
      setIsResizingSidebar(false);
    };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', stopResizing);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';

    return () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', stopResizing);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isResizingSidebar]);

  return (
    <div
      className={`flex h-screen w-screen flex-col overflow-hidden ${isDarkMode ? 'dark bg-slate-950' : ''}`}
      style={isDarkMode ? undefined : { backgroundColor: 'rgba(252, 246, 177, 0.28)' }}
    >
      {/* Top Bar */}
      <div
        className={`relative z-10 flex shrink-0 items-center justify-between border-b px-6 py-3 backdrop-blur-md ${isDarkMode ? 'border-[#2D1E2F]' : 'border-[#e7da7d] shadow-sm'}`}
        style={isDarkMode ? { backgroundColor: '#120c13' } : { backgroundColor: '#e3dac9' }}
      >
        <h1 className={`text-lg font-bold tracking-tight ${isDarkMode ? 'text-[#fff7cc]' : 'text-[#2D1E2F]'}`}>
          GenAI Genesis — AML Network
        </h1>

        <div className="flex items-center gap-3">
          {/* Search */}
          <div className="relative">
            <Search className={`absolute left-3 top-1/2 -translate-y-1/2 ${isDarkMode ? 'text-[#f4e7a1]' : 'text-[#6a5a35]'}`} size={16} />
            <input
              type="text"
              value={globalSearch}
              onChange={(e) => setGlobalSearch(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && executeSearch(globalSearch)}
              placeholder="Search account..."
              className={`rounded-lg border pl-9 pr-3 py-1.5 text-sm outline-none transition-colors ${
                isDarkMode
                  ? 'border-[#2D1E2F] bg-[#1b121d] text-[#fff7cc] placeholder-[#9f8f58] focus:border-[#f4e7a1]'
                  : 'border-[#e7da7d] bg-[#fffbe0] text-[#2D1E2F] placeholder-[#978850] focus:border-[#cdbf5e]'
              }`}
            />
          </div>

          {/* Dark mode toggle */}
          <button
            onClick={() => setIsDarkMode(!isDarkMode)}
            className={`rounded-lg p-2 transition-colors ${isDarkMode ? 'text-[#f4e7a1] hover:text-[#fff7cc] hover:bg-[#1b121d]' : 'text-[#6a5a35] hover:text-[#2D1E2F] hover:bg-[#f3ea98]'}`}
          >
            {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
          </button>
        </div>
      </div>

      <div className="relative min-h-0 flex-1">
        {/* 2D Graph */}
        <div className="h-full w-full" style={{ paddingRight: selectedAccount ? sidebarWidth : 0 }}>
          {loading ? (
            <div className="flex h-full items-center justify-center">
              <div className="flex flex-col items-center gap-3">
                <Loader2 size={32} className={`animate-spin ${isDarkMode ? 'text-[#f4e7a1]' : 'text-[#8f804d]'}`} />
                <p className={`text-sm ${isDarkMode ? 'text-[#f4e7a1]' : 'text-[#8f804d]'}`}>
                  Loading graph data...
                </p>
              </div>
            </div>
          ) : error ? (
            <div className="flex h-full items-center justify-center">
              <div className="flex flex-col items-center gap-3">
                <p className="text-sm text-red-500">{error}</p>
                <button onClick={loadData} className="rounded-lg border border-red-500 px-4 py-1.5 text-sm text-red-500 hover:bg-red-500/10">
                  Retry
                </button>
              </div>
            </div>
          ) : (
            <NetworkGraph
              data={graphData}
              selectedNode={selectedAccount}
              onNodeClick={handleNodeClick}
              isDarkMode={isDarkMode}
              viewMode={viewMode}
            />
          )}
        </div>

        {/* 2D/3D Toggle */}
        <div className="absolute bottom-14 left-5 z-20">
          <div
            className={`flex rounded-lg overflow-hidden border backdrop-blur-md text-sm font-semibold ${
              isDarkMode
                ? 'border-[#2D1E2F]'
                : 'border-[#e7da7d] shadow-sm'
            }`}
            style={isDarkMode ? { backgroundColor: '#120c13' } : { backgroundColor: '#e3dac9' }}
          >
            <button
              onClick={() => setViewMode('2d')}
              className={`px-3 py-1.5 transition-colors ${
                viewMode === '2d'
                  ? isDarkMode
                    ? 'bg-[#2D1E2F] text-[#fff7cc]'
                    : 'bg-[#f3ea98] text-[#2D1E2F]'
                  : isDarkMode
                    ? 'text-[#9f8f58] hover:text-[#fff7cc]'
                    : 'text-[#978850] hover:text-[#2D1E2F]'
              }`}
            >
              2D
            </button>
            <button
              onClick={() => setViewMode('3d')}
              className={`px-3 py-1.5 transition-colors ${
                viewMode === '3d'
                  ? isDarkMode
                    ? 'bg-[#2D1E2F] text-[#fff7cc]'
                    : 'bg-[#f3ea98] text-[#2D1E2F]'
                  : isDarkMode
                    ? 'text-[#9f8f58] hover:text-[#fff7cc]'
                    : 'text-[#978850] hover:text-[#2D1E2F]'
              }`}
            >
              3D
            </button>
          </div>
        </div>

        {/* Sidebar */}
        {selectedAccount && (
          <div className="absolute top-0 right-0 h-full" style={{ width: sidebarWidth }}>
            <Sidebar
              selectedAccount={selectedAccount}
              onClose={() => setSelectedAccount(null)}
              allLinks={graphData.links}
              onResizeStart={(e: ReactPointerEvent<HTMLDivElement>) => {
                e.preventDefault();
                setIsResizingSidebar(true);
              }}
              isDarkMode={isDarkMode}
            />
          </div>
        )}
      </div>
    </div>
  );
}
