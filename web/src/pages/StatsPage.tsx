import { useStats } from '../hooks/useStats';

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl px-5 py-4">
      <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">{label}</p>
      <p className="text-2xl font-bold text-gray-100">{value}</p>
    </div>
  );
}

export function StatsPage() {
  const { stats, loading, error } = useStats();

  if (loading) {
    return <div className="h-full flex items-center justify-center text-gray-500">Loading stats...</div>;
  }

  if (error) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-red-400 text-sm bg-red-500/10 border border-red-500/20 rounded-lg px-4 py-3">
          {error}
        </div>
      </div>
    );
  }

  if (!stats) return null;

  const tiers = Object.entries(stats.tier_distribution)
    .sort(([a], [b]) => Number(a) - Number(b));
  const maxTierCount = Math.max(...tiers.map(([, c]) => c), 1);

  return (
    <div className="h-full overflow-y-auto p-6">
      <h2 className="text-xl font-semibold text-gray-200 mb-6">Corpus Statistics</h2>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatCard label="Papers" value={stats.papers_count} />
        <StatCard label="Chunks" value={stats.chunks_count} />
        <StatCard label="With metrics" value={stats.metrics_count} />
        <StatCard label="PDFs on disk" value={stats.pdf_files} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 max-w-4xl">
        {/* Tier distribution */}
        {tiers.length > 0 && (
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <h3 className="text-sm font-semibold text-gray-300 mb-4">Tier Distribution</h3>
            <div className="space-y-3">
              {tiers.map(([tier, count]) => (
                <div key={tier} className="flex items-center gap-3">
                  <span className={`w-16 text-xs font-medium ${
                    tier === '0' ? 'text-yellow-400' : tier === '1' ? 'text-gray-300' : 'text-gray-500'
                  }`}>
                    Tier {tier}
                  </span>
                  <div className="flex-1 bg-gray-800 rounded-full h-5">
                    <div
                      className={`h-5 rounded-full flex items-center px-2 text-xs font-medium ${
                        tier === '0' ? 'bg-yellow-500/30 text-yellow-300'
                          : tier === '1' ? 'bg-blue-500/30 text-blue-300'
                          : 'bg-gray-700 text-gray-400'
                      }`}
                      style={{ width: `${Math.max(15, (count / maxTierCount) * 100)}%` }}
                    >
                      {count}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Top venues */}
        {stats.top_venues.length > 0 && (
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <h3 className="text-sm font-semibold text-gray-300 mb-4">Top Venues</h3>
            <div className="space-y-2">
              {stats.top_venues.map((v) => (
                <div key={v.venue} className="flex items-center justify-between text-sm">
                  <span className="text-gray-300 truncate">{v.venue}</span>
                  <span className="text-gray-500 font-mono ml-3">{v.count}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
