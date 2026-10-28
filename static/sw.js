// Puck Prophet Service Worker
// Minimal SW for PWA installability + basic offline shell caching

const CACHE_NAME = 'puck-prophet-v1';
const SHELL_ASSETS = [
  '/',
];

// Cache the app shell on install
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(SHELL_ASSETS))
      .then(() => self.skipWaiting())
  );
});

// Clean up old caches on activate
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))
      )
    ).then(() => self.clients.claim())
  );
});

// Network-first strategy: try network, fall back to cache
// This ensures predictions are always fresh when online
self.addEventListener('fetch', event => {
  // Skip non-GET and API requests for caching
  if (event.request.method !== 'GET') return;

  // For API calls, always go to network (no cache fallback for data)
  if (event.request.url.includes('/api/')) return;

  event.respondWith(
    fetch(event.request)
      .then(response => {
        // Cache successful responses for offline fallback
        if (response.ok) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
        }
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});
