import { useCallback, useEffect, useState } from 'react';
import { fetchHealth } from '../api/client';
import type { ServiceHealth } from '../api/types';

export function useHealth() {
  const [services, setServices] = useState<ServiceHealth[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    setLoading(true);
    setError(null);
    fetchHealth()
      .then((data) => setServices(data.services))
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  return { services, loading, error, refresh };
}
