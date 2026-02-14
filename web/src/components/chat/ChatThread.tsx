import { useEffect, useRef } from 'react';
import type { ChatMessage as ChatMessageType } from '../../hooks/useChat';
import { ChatMessage } from './ChatMessage';

interface Props {
  messages: ChatMessageType[];
  activeSourceIndex: number | null;
  onCitationClick: (index: number) => void;
}

export function ChatThread({ messages, activeSourceIndex, onCitationClick }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center max-w-md px-4">
          <h2 className="text-xl font-semibold text-gray-300 mb-3">Ask about CS.CV papers</h2>
          <p className="text-sm text-gray-500 mb-6">
            Ask questions about computer vision research papers in your corpus.
            Answers include inline citations linked to source chunks.
          </p>
          <div className="space-y-2">
            {[
              'How does ViT process image patches?',
              'Compare DETR and Faster R-CNN',
              'What are the main approaches to image segmentation?',
            ].map((q) => (
              <div
                key={q}
                className="text-sm text-gray-400 bg-gray-900 border border-gray-800 rounded-lg px-4 py-2.5 text-left"
              >
                {q}
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto px-6 py-4">
      {messages.map((msg) => (
        <ChatMessage
          key={msg.id}
          message={msg}
          activeSourceIndex={activeSourceIndex}
          onCitationClick={onCitationClick}
        />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
