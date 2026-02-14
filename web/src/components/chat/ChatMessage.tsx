import type { ChatMessage as ChatMessageType } from '../../hooks/useChat';
import { AnswerText } from './AnswerText';
import { ModeIndicator } from './ModeIndicator';

interface Props {
  message: ChatMessageType;
  activeSourceIndex: number | null;
  onCitationClick: (index: number) => void;
}

export function ChatMessage({ message, activeSourceIndex, onCitationClick }: Props) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end mb-4">
        <div className="max-w-2xl bg-blue-600 text-white rounded-2xl rounded-br-md px-4 py-3 text-sm">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="mb-4">
      <div className="max-w-3xl bg-gray-900 border border-gray-800 rounded-2xl rounded-bl-md px-5 py-4">
        {message.route && (
          <div className="mb-3">
            <ModeIndicator mode={message.route.mode} confidence={message.route.confidence} />
          </div>
        )}

        {message.error && (
          <div className="text-red-400 text-sm bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2 mb-3">
            {message.error}
          </div>
        )}

        {message.content ? (
          <div className="text-sm text-gray-200">
            <AnswerText
              text={message.content}
              activeSourceIndex={activeSourceIndex}
              onCitationClick={onCitationClick}
            />
          </div>
        ) : message.isStreaming ? (
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <span className="inline-block w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
            Thinking...
          </div>
        ) : null}

        {message.elapsedMs !== null && (
          <div className="mt-3 text-xs text-gray-600">
            {(message.elapsedMs / 1000).toFixed(1)}s
          </div>
        )}
      </div>
    </div>
  );
}
