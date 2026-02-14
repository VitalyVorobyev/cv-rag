import type { ChunkResponse } from '../../api/types';

interface Props {
  index: number;
  chunk: ChunkResponse;
  isActive: boolean;
}

function ScoreBar({ label, score, color }: { label: string; score: number | null; color: string }) {
  if (score === null) return null;
  const width = Math.max(5, Math.min(100, Math.abs(score) * 100));
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="text-gray-500 w-14 shrink-0">{label}</span>
      <div className="flex-1 bg-gray-800 rounded-full h-1.5">
        <div className={`h-1.5 rounded-full ${color}`} style={{ width: `${width}%` }} />
      </div>
      <span className="text-gray-500 w-10 text-right font-mono">{score.toFixed(3)}</span>
    </div>
  );
}

export function SourceCard({ index, chunk, isActive }: Props) {
  return (
    <div
      id={`source-${index}`}
      className={`rounded-lg border p-3 transition-colors ${
        isActive
          ? 'border-blue-500 bg-blue-500/10'
          : 'border-gray-800 bg-gray-900/50 hover:border-gray-700'
      }`}
    >
      <div className="flex items-start justify-between gap-2 mb-2">
        <span className="inline-flex items-center justify-center w-7 h-7 rounded-md bg-blue-500/20 text-blue-400 text-xs font-bold font-mono shrink-0">
          S{index}
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-xs font-medium text-gray-300 truncate">{chunk.title}</p>
          <p className="text-xs text-gray-500">{chunk.arxiv_id} &middot; {chunk.section_title || 'Untitled'}</p>
        </div>
      </div>
      <p className="text-xs text-gray-400 leading-relaxed line-clamp-4 mb-2">{chunk.text}</p>
      <div className="space-y-1">
        <ScoreBar label="fused" score={chunk.fused_score} color="bg-green-500" />
        <ScoreBar label="vector" score={chunk.vector_score} color="bg-blue-500" />
        <ScoreBar label="keyword" score={chunk.keyword_score} color="bg-orange-500" />
      </div>
      {chunk.sources.length > 0 && (
        <div className="flex gap-1 mt-2">
          {chunk.sources.map((s) => (
            <span key={s} className="text-[10px] px-1.5 py-0.5 rounded bg-gray-800 text-gray-500">{s}</span>
          ))}
        </div>
      )}
    </div>
  );
}
