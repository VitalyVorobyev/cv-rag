const MODE_COLORS: Record<string, string> = {
  single: 'bg-blue-500/20 text-blue-400',
  compare: 'bg-purple-500/20 text-purple-400',
  survey: 'bg-green-500/20 text-green-400',
  implement: 'bg-amber-500/20 text-amber-400',
  evidence: 'bg-red-500/20 text-red-400',
};

interface Props {
  mode: string;
  confidence: number;
}

export function ModeIndicator({ mode, confidence }: Props) {
  const colors = MODE_COLORS[mode] ?? 'bg-gray-500/20 text-gray-400';
  return (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium ${colors}`}>
      {mode}
      <span className="opacity-60">{Math.round(confidence * 100)}%</span>
    </span>
  );
}
