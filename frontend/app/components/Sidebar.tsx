import { type PointerEvent, useEffect, useState } from 'react';
import { AlertTriangle, ShieldCheck, Activity, ArrowRight, ArrowLeft } from 'lucide-react';

export default function Sidebar({ 
  selectedAccount, 
  onClose, 
  allLinks = [],
  onResizeStart,
  isDarkMode
}: { 
  selectedAccount: any, 
  onClose: () => void,
  allLinks?: any[],
  onResizeStart: (event: PointerEvent<HTMLDivElement>) => void,
  isDarkMode: boolean
}) {
  const [counterpartyFilter, setCounterpartyFilter] = useState('');
  const isHighRisk = selectedAccount?.risk === 'laundering';
  const isMediumRisk = selectedAccount?.risk === 'suspicious';
  const panelClasses = isDarkMode
    ? 'border-[#5c4a11] bg-[#21181f] text-[#fdf4c6]'
    : 'border-[#e7da7d] bg-[#fffbe0] text-[#2D1E2F]';
  const titleClasses = isDarkMode ? 'text-[#fff7cc]' : 'text-[#2D1E2F]';
  const mutedTextClasses = isDarkMode ? 'text-[#d9c874]' : 'text-[#6a5a35]';
  const bodyTextClasses = isDarkMode ? 'text-[#f0df9b]' : 'text-[#4b394d]';
  const surfaceClasses = isDarkMode
    ? 'border-[#5c4a11] bg-[#2c2129]'
    : 'border-[#efe39a] bg-[#fffdf1]';
  const resizeHandleClasses = isDarkMode
    ? 'bg-[#705f19] hover:bg-[#8b7720]'
    : 'bg-[#efe39a] hover:bg-[#e7da7d]';

  const accountTransactions = allLinks.filter(link => {
    const sourceId = link.source.id || link.source;
    const targetId = link.target.id || link.target;
    return sourceId === selectedAccount?.id || targetId === selectedAccount?.id;
  });

  const filteredTransactions = !counterpartyFilter.trim()
    ? accountTransactions
    : accountTransactions.filter(link => {
        const sourceId = link.source.id || link.source;
        const targetId = link.target.id || link.target;
        const counterpartId = sourceId === selectedAccount?.id ? targetId : sourceId;
        return counterpartId.toLowerCase().includes(counterpartyFilter.trim().toLowerCase());
      });

  useEffect(() => {
    setCounterpartyFilter('');
  }, [selectedAccount?.id]);

  return (
    <div className={`relative h-full w-full overflow-hidden border-l p-6 shadow-2xl transition-colors duration-300 ${panelClasses}`}>
      <div
        role="separator"
        aria-label="Resize sidebar"
        aria-orientation="vertical"
        onPointerDown={onResizeStart}
        className="absolute left-0 top-0 h-full w-3 -translate-x-1/2 cursor-col-resize touch-none"
      >
        <div className={`absolute inset-y-0 left-1/2 w-px -translate-x-1/2 transition-colors ${resizeHandleClasses}`} />
      </div>

      <div className="flex justify-between items-center shrink-0">
        <h2 className={`text-xl font-bold tracking-tight ${titleClasses}`}>Account Details</h2>
        <button
          onClick={onClose}
          className={`rounded-md p-2 text-4xl leading-none transition-colors ${isDarkMode ? 'text-[#d9c874] hover:text-[#fff7cc] hover:bg-[#2c2129]' : 'text-[#6a5a35] hover:text-[#2D1E2F] hover:bg-[#fff7cc]'}`}
          aria-label="Close sidebar"
        >
          &times;
        </button>
      </div>

      {selectedAccount && (
        <div className="flex flex-col h-full overflow-hidden space-y-6">
          
          <div className="space-y-6 shrink-0">
            <div>
              <p className={`mb-1 text-sm uppercase tracking-wider ${mutedTextClasses}`}>Account ID</p>
              <p className={`text-xl font-mono ${titleClasses}`}>{selectedAccount.id}</p>
              
              {/* Rest of your risk badge code remains exactly the same */}
              <div className={`mt-3 inline-flex items-center gap-2 rounded-full px-3 py-1 text-sm font-semibold
                ${isHighRisk ? 'border border-[#e3170a]/30 bg-[#e3170a]/12 text-[#e3170a] dark:border-[#8f0e08]/60 dark:bg-[#8f0e08]/22 dark:text-[#e06b62]' : 
                  isMediumRisk ? 'border border-[#F78D2A]/30 bg-[#F78D2A]/14 text-[#b85f14] dark:border-[#B85F14]/60 dark:bg-[#B85F14]/22 dark:text-[#f0a55f]' : 
                  'border border-[#a9e5bb]/45 bg-[#a9e5bb]/30 text-[#4d6f56] dark:border-[#5B8A68]/60 dark:bg-[#5B8A68]/24 dark:text-[#b5d8bf]'}`}>
                {isHighRisk && <AlertTriangle size={16} />}
                {isMediumRisk && <Activity size={16} />}
                {!isHighRisk && !isMediumRisk && <ShieldCheck size={16} />}
                {selectedAccount.risk.toUpperCase()}
              </div>
            </div>

            <div>
              <h3 className={`mb-2 text-sm uppercase tracking-wider ${mutedTextClasses}`}>Pattern Detected</h3>
              <p className={`font-semibold ${isDarkMode ? 'text-slate-100' : 'text-stone-900'}`}>{selectedAccount.pattern}</p>
            </div>

            <div>
              <h3 className={`mb-2 text-sm uppercase tracking-wider ${mutedTextClasses}`}>AI Explanation</h3>
              <p className={`text-sm leading-relaxed ${bodyTextClasses}`}>
                {selectedAccount.aiExplanation}
              </p>
            </div>
          </div>

          {/* Transactions List remains exactly the same */}
          <div className={`flex flex-col flex-1 min-h-0 border-t pt-6 transition-colors ${isDarkMode ? 'border-[#5c4a11]' : 'border-[#efe39a]'}`}>
            <div className="mb-4 shrink-0">
              <label
                htmlFor="counterparty-filter"
                className={`mb-2 block text-sm uppercase tracking-wider ${mutedTextClasses}`}
              >
                Filter User
              </label>
              <input
                id="counterparty-filter"
                type="text"
                value={counterpartyFilter}
                onChange={(e) => setCounterpartyFilter(e.target.value)}
                placeholder="Type an account ID"
                className={`w-full rounded-lg border px-3 py-2 text-sm outline-none transition-colors ${
                  isDarkMode
                    ? 'border-[#5c4a11] bg-[#2c2129] text-[#fff7cc] placeholder-[#b7a867] focus:border-[#d9c874]'
                    : 'border-[#e7da7d] bg-[#fffdf1] text-[#2D1E2F] placeholder-[#978850] focus:border-[#cdbf5e]'
                }`}
              />
            </div>

            <div className="flex justify-between items-end mb-3 shrink-0">
              <h3 className={`text-sm uppercase tracking-wider ${mutedTextClasses}`}>Known Transactions</h3>
              <span className={`text-xs ${mutedTextClasses}`}>{filteredTransactions.length} records</span>
            </div>
             
            <div className="flex-1 overflow-y-auto pr-2 space-y-2 pb-4 scrollbar-thin scrollbar-thumb-[#efe39a] dark:scrollbar-thumb-[#705f19] scrollbar-track-transparent">
              {filteredTransactions.length === 0 ? (
                <p className={`text-sm italic ${mutedTextClasses}`}>
                  {accountTransactions.length === 0 ? 'No connections found.' : 'No transactions match the selected user.'}
                </p>
              ) : (
                filteredTransactions.map((tx, idx) => {
                  const sourceId = tx.source.id || tx.source;
                  const targetId = tx.target.id || tx.target;
                  const isOutgoing = sourceId === selectedAccount.id;
                  const counterpartId = isOutgoing ? targetId : sourceId;

                  return (
                    <div
                      key={idx}
                      className={`flex flex-col gap-1 rounded-lg border p-3 transition-colors ${
                        isDarkMode
                          ? `${surfaceClasses} hover:border-[#8b7720]`
                          : `${surfaceClasses} hover:border-[#e7da7d]`
                      }`}
                    >
                      <div className="flex justify-between items-center">
                        <div className={`flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider ${isOutgoing ? 'text-[#b85f14] dark:text-[#f0a55f]' : 'text-[#5c7c66] dark:text-[#b5d8bf]'}`}>
                          {isOutgoing ? <ArrowRight size={14} /> : <ArrowLeft size={14} />}
                          {isOutgoing ? 'Sent to' : 'Received from'}
                        </div>
                        <div className={`font-mono text-sm font-semibold ${isDarkMode ? 'text-slate-200' : 'text-stone-950'}`}>
                          ${tx.amount.toLocaleString()}
                        </div>
                      </div>
                      <div className={`ml-5 font-mono text-sm ${mutedTextClasses}`}>
                        {counterpartId}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>

        </div>
      )}
    </div>
  );
}
