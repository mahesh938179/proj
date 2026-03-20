// static/service-worker.js
const CACHE_NAME = 'stock-ai-v1';
const ASSETS_TO_CACHE = [
  '/',
  '/static/css/style.css',
  '/static/manifest.json',
  '/static/icon-512.png'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(ASSETS_TO_CACHE))
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => response || fetch(event.request))
  );
});
