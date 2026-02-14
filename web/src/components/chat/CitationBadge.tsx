interface Props {
  index: number;
  isActive: boolean;
  onClick: () => void;
}

export function CitationBadge({ index, isActive, onClick }: Props) {
  return (
    <button
      onClick={onClick}
      className={`inline-flex items-center justify-center px-1.5 py-0.5 mx-0.5 text-xs font-mono font-bold rounded transition-colors cursor-pointer ${
        isActive
          ? 'bg-blue-500 text-white'
          : 'bg-blue-500/20 text-blue-400 hover:bg-blue-500/30'
      }`}
    >
      S{index}
    </button>
  );
}
