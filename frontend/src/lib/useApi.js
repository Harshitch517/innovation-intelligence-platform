import { useEffect, useRef, useState } from "react";

// Run an async loader whenever `deps` change. Keeps the previous data while refetching so
// charts hold their frame instead of flashing.
export function useApi(loader, deps) {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  const seq = useRef(0);
  useEffect(() => {
    const id = ++seq.current;
    setState((s) => ({ ...s, loading: true, error: null }));
    loader()
      .then((data) => id === seq.current && setState({ data, error: null, loading: false }))
      .catch((error) => id === seq.current && setState((s) => ({ ...s, error, loading: false })));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}
