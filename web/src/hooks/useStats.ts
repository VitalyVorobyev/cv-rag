import { useEffect, useState } from 'react';
import { fetchStats } from '../api/client';
import type { StatsResponse } from '../api/types';

export function useStats() {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchStats()
      .then(setStats)
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, []);

  return { stats, loading, error };
}
