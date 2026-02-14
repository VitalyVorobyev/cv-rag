import { NavLink } from 'react-router-dom';

const links = [
  { to: '/', label: 'Chat', icon: '💬' },
  { to: '/papers', label: 'Papers', icon: '📄' },
  { to: '/stats', label: 'Stats', icon: '📊' },
  { to: '/health', label: 'Health', icon: '🏥' },
];

export function Sidebar() {
  return (
    <aside className="w-52 bg-gray-900 border-r border-gray-800 flex flex-col shrink-0">
      <div className="px-4 py-5 border-b border-gray-800">
        <h1 className="text-lg font-bold text-white tracking-tight">cv-rag</h1>
        <p className="text-xs text-gray-500 mt-0.5">CS.CV Paper RAG</p>
      </div>
      <nav className="flex-1 px-2 py-3 space-y-0.5">
        {links.map(({ to, label, icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
                isActive
                  ? 'bg-blue-600/20 text-blue-400 font-medium'
                  : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'
              }`
            }
          >
            <span className="text-base">{icon}</span>
            {label}
          </NavLink>
        ))}
      </nav>
      <div className="px-4 py-3 border-t border-gray-800 text-xs text-gray-600">
        v0.1.0
      </div>
    </aside>
  );
}
