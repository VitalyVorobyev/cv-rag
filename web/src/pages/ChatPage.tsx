import { ChatInput } from '../components/chat/ChatInput';
import { ChatThread } from '../components/chat/ChatThread';
import { SourcePanel } from '../components/chat/SourcePanel';
import { useChat } from '../hooks/useChat';

export function ChatPage() {
  const {
    messages,
    sendMessage,
    stopGeneration,
    activeSourceIndex,
    setActiveSourceIndex,
  } = useChat();

  const lastAssistant = [...messages].reverse().find((m) => m.role === 'assistant');
  const currentSources = lastAssistant?.sources ?? [];
  const isStreaming = messages.some((m) => m.isStreaming);

  return (
    <div className="flex h-full">
      <div className="flex-1 flex flex-col min-w-0">
        <ChatThread
          messages={messages}
          activeSourceIndex={activeSourceIndex}
          onCitationClick={setActiveSourceIndex}
        />
        <ChatInput
          onSend={sendMessage}
          isStreaming={isStreaming}
          onStop={stopGeneration}
        />
      </div>
      <div className="w-80 border-l border-gray-800 shrink-0 bg-gray-950">
        <SourcePanel sources={currentSources} activeIndex={activeSourceIndex} />
      </div>
    </div>
  );
}
