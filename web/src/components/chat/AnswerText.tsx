import { parseAnswerWithCitations } from '../../lib/citations';
import { CitationBadge } from './CitationBadge';

interface Props {
  text: string;
  activeSourceIndex: number | null;
  onCitationClick: (index: number) => void;
}

export function AnswerText({ text, activeSourceIndex, onCitationClick }: Props) {
  const parts = parseAnswerWithCitations(text);

  return (
    <div className="leading-relaxed whitespace-pre-wrap">
      {parts.map((part, i) =>
        part.type === 'text' ? (
          <span key={i}>{part.value}</span>
        ) : (
          <CitationBadge
            key={i}
            index={part.index}
            isActive={activeSourceIndex === part.index}
            onClick={() => onCitationClick(part.index)}
          />
        ),
      )}
    </div>
  );
}
