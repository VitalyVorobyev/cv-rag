import { useCallback, useEffect, useState } from 'react';
import { fetchPapers } from '../api/client';
import type { PaperSummary } from '../api/types';

export function usePapers(pageSize = 20) {
  const [papers, setPapers] = useState<PaperSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    fetchPapers(offset, pageSize, search)
      .then((data) => {
        setPapers(data.papers);
        setTotal(data.total);
      })
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, [offset, pageSize, search]);

  useEffect(() => { load(); }, [load]);

  const goToPage = (page: number) => setOffset(page * pageSize);
  const currentPage = Math.floor(offset / pageSize);
  const totalPages = Math.ceil(total / pageSize);

  return {
    papers, total, loading, error, search, setSearch,
    currentPage, totalPages, goToPage,
  };
}
