import { useNavigate } from 'react-router-dom';
import { usePapers } from '../hooks/usePapers';

function TierBadge({ tier }: { tier: number | null }) {
  if (tier === null) return <span className="text-xs text-gray-600">-</span>;
  const styles =
    tier === 0 ? 'bg-yellow-500/20 text-yellow-400'
    : tier === 1 ? 'bg-blue-500/20 text-blue-400'
    : 'bg-gray-700/50 text-gray-500';
  return (
    <span className={`text-xs font-medium px-1.5 py-0.5 rounded ${styles}`}>
      T{tier}
    </span>
  );
}

export function PapersPage() {
  const {
    papers, loading, error, search, setSearch,
    currentPage, totalPages, goToPage, total,
  } = usePapers();
  const navigate = useNavigate();

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-200">Papers</h2>
          <p className="text-xs text-gray-500 mt-1">{total} papers in corpus</p>
        </div>
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search by ID or title..."
          className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-300 w-72 focus:outline-none focus:border-blue-500 placeholder-gray-600"
        />
      </div>

      {error && (
        <div className="text-red-400 text-sm bg-red-500/10 border border-red-500/20 rounded-lg px-4 py-3 mb-4">
          {error}
        </div>
      )}

      {loading ? (
        <div className="text-gray-500 text-sm">Loading...</div>
      ) : papers.length === 0 ? (
        <div className="text-center text-gray-500 py-16">
          <p className="text-lg mb-2">No papers found</p>
          <p className="text-sm">Run <code className="bg-gray-800 px-1.5 py-0.5 rounded text-gray-400">cv-rag ingest</code> to add papers</p>
        </div>
      ) : (
        <>
          <div className="space-y-2 mb-6">
            {papers.map((p) => (
              <button
                key={p.arxiv_id}
                onClick={() => navigate(`/papers/${p.arxiv_id}`)}
                className="w-full text-left bg-gray-900 border border-gray-800 rounded-xl px-5 py-4 hover:border-gray-700 transition-colors"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium text-gray-200 truncate">{p.title}</p>
                    <div className="flex items-center gap-3 mt-1.5 text-xs text-gray-500">
                      <span className="font-mono">{p.arxiv_id}</span>
                      {p.published && <span>{new Date(p.published).toLocaleDateString()}</span>}
                      <span>{p.chunk_count} chunks</span>
                      {p.citation_count !== null && <span>{p.citation_count} citations</span>}
                      {p.venue && <span>{p.venue}</span>}
                    </div>
                  </div>
                  <TierBadge tier={p.tier} />
                </div>
              </button>
            ))}
          </div>

          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-2">
              <button
                onClick={() => goToPage(currentPage - 1)}
                disabled={currentPage === 0}
                className="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-gray-300 text-sm rounded-lg transition-colors disabled:opacity-30"
              >
                Prev
              </button>
              <span className="text-xs text-gray-500">
                {currentPage + 1} / {totalPages}
              </span>
              <button
                onClick={() => goToPage(currentPage + 1)}
                disabled={currentPage >= totalPages - 1}
                className="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-gray-300 text-sm rounded-lg transition-colors disabled:opacity-30"
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
