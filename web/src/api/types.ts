export interface ChunkResponse {
  chunk_id: string;
  arxiv_id: string;
  title: string;
  section_title: string;
  text: string;
  fused_score: number;
  vector_score: number | null;
  keyword_score: number | null;
  sources: string[];
}

export interface SearchResponse {
  chunks: ChunkResponse[];
  query: string;
  elapsed_ms: number;
}

export interface RouteInfo {
  mode: string;
  targets: string[];
  k: number;
  max_per_doc: number;
  confidence: number;
  notes: string;
  preface: string | null;
  reason_codes?: string[];
  policy_version?: string;
}

export interface AnswerResponse {
  answer: string;
  sources: ChunkResponse[];
  route: RouteInfo;
  citation_valid: boolean;
  citation_reason: string;
  elapsed_ms: number;
}

export interface PaperSummary {
  arxiv_id: string;
  title: string;
  summary: string | null;
  published: string | null;
  updated: string | null;
  authors: string[];
  pdf_url: string | null;
  abs_url: string | null;
  chunk_count: number;
  tier: number | null;
  citation_count: number | null;
  venue: string | null;
}

export interface PaperListResponse {
  papers: PaperSummary[];
  total: number;
  offset: number;
  limit: number;
}

export interface PaperDetailResponse {
  paper: PaperSummary;
  chunks: ChunkResponse[];
}

export interface StatsResponse {
  papers_count: number;
  chunks_count: number;
  chunk_docs_count: number;
  metrics_count: number;
  papers_without_metrics: number;
  pdf_files: number;
  tei_files: number;
  tier_distribution: Record<string, number>;
  top_venues: { venue: string; count: number }[];
}

export interface ServiceHealth {
  service: string;
  status: string;
  detail: string;
}

export interface HealthResponse {
  services: ServiceHealth[];
}

export type AnswerMode =
  | 'auto'
  | 'single'
  | 'explain'
  | 'compare'
  | 'survey'
  | 'implement'
  | 'evidence'
  | 'decision';

export interface AnswerRequest {
  question: string;
  model: string;
  mode: AnswerMode;
  router_strategy: string;
  max_tokens: number;
  temperature: number;
  top_p: number;
  seed?: number | null;
  k?: number | null;
  max_per_doc?: number | null;
  section_boost: number;
  no_refuse: boolean;
}
