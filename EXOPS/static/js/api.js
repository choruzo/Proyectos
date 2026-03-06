/**
 * api.js — Cliente fetch centralizado para EXOPS
 * Maneja headers comunes, errores 401 y parseo JSON.
 */
const api = {
  async request(method, url, body = null) {
    const options = {
      method,
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
    };
    if (body !== null) {
      options.body = JSON.stringify(body);
    }

    const res = await fetch(url, options);

    if (res.status === 401) {
      window.location.href = '/login';
      return null;
    }

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || `Error ${res.status}`);
    }

    const contentType = res.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
      return res.json();
    }
    return null;
  },

  get(url) {
    return this.request('GET', url);
  },

  post(url, body = null) {
    return this.request('POST', url, body);
  },

  put(url, body = null) {
    return this.request('PUT', url, body);
  },

  patch(url, body = null) {
    return this.request('PATCH', url, body);
  },

  delete(url) {
    return this.request('DELETE', url);
  },
};
