import { useCallback, useRef, useState } from 'react';
import { streamAnswer } from '../api/client';
import type { AnswerMode, ChunkResponse, RouteInfo } from '../api/types';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources: ChunkResponse[];
  route: RouteInfo | null;
  isStreaming: boolean;
  error: string | null;
  citationValid: boolean | null;
  elapsedMs: number | null;
}

let messageIdCounter = 0;
function nextId() {
  return `msg-${++messageIdCounter}`;
}

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [activeSourceIndex, setActiveSourceIndex] = useState<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback((
    question: string,
    model: string,
    mode: AnswerMode = 'auto',
  ) => {
    // Cancel any in-flight request
    abortRef.current?.abort();

    const userMsg: ChatMessage = {
      id: nextId(),
      role: 'user',
      content: question,
      sources: [],
      route: null,
      isStreaming: false,
      error: null,
      citationValid: null,
      elapsedMs: null,
    };

    const assistantId = nextId();
    const assistantMsg: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      sources: [],
      route: null,
      isStreaming: true,
      error: null,
      citationValid: null,
      elapsedMs: null,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setActiveSourceIndex(null);

    const update = (patch: Partial<ChatMessage>) => {
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantId ? { ...m, ...patch } : m)),
      );
    };

    const controller = streamAnswer(
      {
        question,
        model,
        mode,
        router_strategy: 'rules',
        max_tokens: 600,
        temperature: 0.2,
        top_p: 0.9,
        section_boost: 0.05,
        no_refuse: false,
      },
      {
        onRoute: (route) => update({ route }),
        onSources: (sources) => update({ sources }),
        onToken: (token) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, content: m.content + token } : m,
            ),
          );
        },
        onRepair: (repaired) => update({ content: repaired }),
        onDone: (response) => {
          update({
            content: response.answer,
            sources: response.sources,
            route: response.route,
            isStreaming: false,
            citationValid: response.citation_valid,
            elapsedMs: response.elapsed_ms,
          });
        },
        onError: (err) => {
          update({ isStreaming: false, error: err.message });
        },
      },
    );

    abortRef.current = controller;
  }, []);

  const stopGeneration = useCallback(() => {
    abortRef.current?.abort();
    setMessages((prev) =>
      prev.map((m) => (m.isStreaming ? { ...m, isStreaming: false } : m)),
    );
  }, []);

  return {
    messages,
    sendMessage,
    stopGeneration,
    activeSourceIndex,
    setActiveSourceIndex,
  };
}
