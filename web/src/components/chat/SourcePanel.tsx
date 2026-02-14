import { useEffect, useRef } from 'react';
import type { ChunkResponse } from '../../api/types';
import { SourceCard } from './SourceCard';

interface Props {
  sources: ChunkResponse[];
  activeIndex: number | null;
}

export function SourcePanel({ sources, activeIndex }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (activeIndex !== null && containerRef.current) {
      const el = containerRef.current.querySelector(`#source-${activeIndex}`);
      el?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [activeIndex]);

  if (sources.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-gray-600 text-sm px-4 text-center">
        Sources will appear here when you ask a question
      </div>
    );
  }

  return (
    <div ref={containerRef} className="h-full overflow-y-auto p-3 space-y-2">
      <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider px-1 mb-2">
        Sources ({sources.length})
      </h3>
      {sources.map((chunk, i) => (
        <SourceCard
          key={chunk.chunk_id}
          index={i + 1}
          chunk={chunk}
          isActive={activeIndex === i + 1}
        />
      ))}
    </div>
  );
}
