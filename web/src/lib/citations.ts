export type AnswerPart = { type: 'text'; value: string } | { type: 'citation'; index: number };

const CITATION_RE = /\[S(\d+)\]/g;

export function parseAnswerWithCitations(text: string): AnswerPart[] {
  const parts: AnswerPart[] = [];
  let lastIndex = 0;

  for (const match of text.matchAll(CITATION_RE)) {
    const matchIndex = match.index;
    if (matchIndex > lastIndex) {
      parts.push({ type: 'text', value: text.slice(lastIndex, matchIndex) });
    }
    parts.push({ type: 'citation', index: parseInt(match[1], 10) });
    lastIndex = matchIndex + match[0].length;
  }

  if (lastIndex < text.length) {
    parts.push({ type: 'text', value: text.slice(lastIndex) });
  }

  return parts;
}
