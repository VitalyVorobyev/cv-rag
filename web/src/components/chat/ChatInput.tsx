import { useState, type FormEvent, type KeyboardEvent } from 'react';
import type { AnswerMode } from '../../api/types';

interface Props {
  onSend: (question: string, model: string, mode: AnswerMode) => void;
  isStreaming: boolean;
  onStop: () => void;
}

const MODES: { value: AnswerMode; label: string }[] = [
  { value: 'auto', label: 'Auto' },
  { value: 'explain', label: 'Explain' },
  { value: 'compare', label: 'Compare' },
  { value: 'survey', label: 'Survey' },
  { value: 'implement', label: 'Implement' },
  { value: 'evidence', label: 'Evidence' },
  { value: 'decision', label: 'Decision' },
];

const DEFAULT_MODEL = 'mlx-community/Qwen2.5-7B-Instruct-4bit';

export function ChatInput({ onSend, isStreaming, onStop }: Props) {
  const [question, setQuestion] = useState('');
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [mode, setMode] = useState<AnswerMode>('auto');
  const [showSettings, setShowSettings] = useState(false);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || isStreaming) return;
    onSend(trimmed, model, mode);
    setQuestion('');
  };

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="border-t border-gray-800 bg-gray-950 px-4 py-3">
      {showSettings && (
        <div className="flex items-center gap-3 mb-2 text-xs">
          <label className="flex items-center gap-1.5 text-gray-400">
            Model
            <input
              type="text"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="bg-gray-900 border border-gray-700 rounded px-2 py-1 text-gray-300 w-72 focus:outline-none focus:border-blue-500"
            />
          </label>
          <label className="flex items-center gap-1.5 text-gray-400">
            Mode
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value as AnswerMode)}
              className="bg-gray-900 border border-gray-700 rounded px-2 py-1 text-gray-300 focus:outline-none focus:border-blue-500"
            >
              {MODES.map((m) => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
          </label>
        </div>
      )}
      <form onSubmit={handleSubmit} className="flex items-end gap-2">
        <button
          type="button"
          onClick={() => setShowSettings(!showSettings)}
          className="shrink-0 w-8 h-10 flex items-center justify-center text-gray-500 hover:text-gray-300 transition-colors rounded-lg hover:bg-gray-800"
          title="Toggle settings"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
          </svg>
        </button>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about CS.CV papers..."
          rows={1}
          className="flex-1 bg-gray-900 border border-gray-700 rounded-xl px-4 py-2.5 text-sm text-gray-200 resize-none focus:outline-none focus:border-blue-500 placeholder-gray-600"
        />
        {isStreaming ? (
          <button
            type="button"
            onClick={onStop}
            className="shrink-0 px-4 py-2.5 bg-red-600 hover:bg-red-700 text-white text-sm font-medium rounded-xl transition-colors"
          >
            Stop
          </button>
        ) : (
          <button
            type="submit"
            disabled={!question.trim()}
            className="shrink-0 px-4 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-800 disabled:text-gray-600 text-white text-sm font-medium rounded-xl transition-colors"
          >
            Send
          </button>
        )}
      </form>
    </div>
  );
}
