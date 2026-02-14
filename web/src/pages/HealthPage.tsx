import { useHealth } from '../hooks/useHealth';

export function HealthPage() {
  const { services, loading, error, refresh } = useHealth();

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-200">Service Health</h2>
        <button
          onClick={refresh}
          disabled={loading}
          className="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-gray-300 text-sm rounded-lg transition-colors disabled:opacity-50"
        >
          {loading ? 'Checking...' : 'Refresh'}
        </button>
      </div>

      {error && (
        <div className="text-red-400 text-sm bg-red-500/10 border border-red-500/20 rounded-lg px-4 py-3 mb-4">
          {error}
        </div>
      )}

      <div className="grid gap-4 max-w-2xl">
        {services.map((svc) => (
          <div
            key={svc.service}
            className="bg-gray-900 border border-gray-800 rounded-xl px-5 py-4 flex items-center gap-4"
          >
            <div
              className={`w-3 h-3 rounded-full shrink-0 ${
                svc.status === 'ok' ? 'bg-green-500' : 'bg-red-500'
              }`}
            />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-200">{svc.service}</p>
              <p className="text-xs text-gray-500 truncate">{svc.detail}</p>
            </div>
            <span
              className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                svc.status === 'ok'
                  ? 'bg-green-500/20 text-green-400'
                  : 'bg-red-500/20 text-red-400'
              }`}
            >
              {svc.status}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
