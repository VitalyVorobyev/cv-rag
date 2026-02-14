import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { fetchPaper } from '../api/client';
import type { ChunkResponse, PaperSummary } from '../api/types';

export function PaperDetailPage() {
  const { arxivId } = useParams<{ arxivId: string }>();
  const navigate = useNavigate();
  const [paper, setPaper] = useState<PaperSummary | null>(null);
  const [chunks, setChunks] = useState<ChunkResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!arxivId) return;
    setLoading(true);
    fetchPaper(arxivId)
      .then((data) => {
        setPaper(data.paper);
        setChunks(data.chunks);
      })
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, [arxivId]);

  if (loading) {
    return <div className="h-full flex items-center justify-center text-gray-500">Loading...</div>;
  }

  if (error || !paper) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-red-400 text-sm">{error || 'Paper not found'}</div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-6">
      <button
        onClick={() => navigate('/papers')}
        className="text-sm text-gray-500 hover:text-gray-300 mb-4 flex items-center gap-1"
      >
        &larr; Back to papers
      </button>

      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 mb-6 max-w-4xl">
        <h2 className="text-lg font-semibold text-gray-200 mb-2">{paper.title}</h2>
        <div className="flex flex-wrap items-center gap-3 text-xs text-gray-500 mb-4">
          <span className="font-mono bg-gray-800 px-2 py-0.5 rounded">{paper.arxiv_id}</span>
          {paper.published && <span>{new Date(paper.published).toLocaleDateString()}</span>}
          {paper.venue && <span className="bg-gray-800 px-2 py-0.5 rounded">{paper.venue}</span>}
          {paper.tier !== null && (
            <span className={`px-2 py-0.5 rounded font-medium ${
              paper.tier === 0 ? 'bg-yellow-500/20 text-yellow-400'
              : paper.tier === 1 ? 'bg-blue-500/20 text-blue-400'
              : 'bg-gray-700 text-gray-500'
            }`}>
              Tier {paper.tier}
            </span>
          )}
          {paper.citation_count !== null && <span>{paper.citation_count} citations</span>}
        </div>
        {paper.authors.length > 0 && (
          <p className="text-xs text-gray-400 mb-3">{paper.authors.join(', ')}</p>
        )}
        {paper.summary && (
          <p className="text-sm text-gray-400 leading-relaxed">{paper.summary}</p>
        )}
        <div className="flex gap-3 mt-4">
          {paper.abs_url && (
            <a href={paper.abs_url} target="_blank" rel="noopener noreferrer"
               className="text-xs text-blue-400 hover:text-blue-300">
              Abstract
            </a>
          )}
          {paper.pdf_url && (
            <a href={paper.pdf_url} target="_blank" rel="noopener noreferrer"
               className="text-xs text-blue-400 hover:text-blue-300">
              PDF
            </a>
          )}
        </div>
      </div>

      <h3 className="text-sm font-semibold text-gray-300 mb-3">Chunks ({chunks.length})</h3>
      <div className="space-y-2 max-w-4xl">
        {chunks.map((chunk, i) => (
          <div key={chunk.chunk_id} className="bg-gray-900 border border-gray-800 rounded-lg px-4 py-3">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs font-mono text-gray-500">#{i}</span>
              <span className="text-xs font-medium text-gray-400">{chunk.section_title || 'Untitled'}</span>
            </div>
            <p className="text-xs text-gray-400 leading-relaxed whitespace-pre-wrap">{chunk.text}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
