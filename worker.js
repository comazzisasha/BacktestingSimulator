/* Runs the Python engine off the main thread so the page stays responsive
   and long runs can be stopped. */
importScripts('runtime.js');
self.onmessage = function (e) {
  self.QLRuntime.onMessage(e.data, function (m) { self.postMessage(m); });
};
