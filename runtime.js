/*
 * QuantLab Python runtime.
 * Loads Pyodide (Python compiled to WebAssembly) and engine.py, then answers
 * messages of the form {id, type: 'init' | 'call', ...}. Used inside the Web
 * Worker (worker.js) and, if workers are unavailable, on the page itself.
 * Before a call that carries strategy code, the packages it imports (NumPy,
 * pandas, SciPy, scikit-learn, ...) are loaded; progress is reported as
 * {id, progress: 'Loading numpy'} messages ahead of the final reply.
 */
(function (root) {
  'use strict';
  let py = null;
  let handle = null;
  let packagesFor = null;   // last strategy code whose imports loaded successfully

  function loadScript(url) {
    if (typeof importScripts === 'function') {
      importScripts(url);
      return Promise.resolve();
    }
    return new Promise(function (resolve, reject) {
      const s = document.createElement('script');
      s.src = url;
      s.onload = resolve;
      s.onerror = function () { reject(new Error('could not load ' + url)); };
      document.head.appendChild(s);
    });
  }

  // Some static hosts refuse to serve .zip files. If python_stdlib.zip can't be
  // fetched, fall back to a base64 copy (python_stdlib.b64.txt) next to it.
  // Package wheels are looked for next to the runtime first (pyodide/ bundles
  // NumPy); any that aren't there come from packageURL, the full Pyodide CDN.
  function installFetchFallbacks(packageURL) {
    if (root.__qlFetchPatched || typeof root.fetch !== 'function') return;
    root.__qlFetchPatched = true;
    const orig = root.fetch.bind(root);
    root.fetch = async function (input, opts) {
      const url = typeof input === 'string' ? input : (input && input.url) || String(input);
      const pkg = url.split('?')[0].split('/').pop();
      if (packageURL && /\.(whl|zip|tar)$/.test(pkg) && pkg !== 'python_stdlib.zip' && !url.startsWith(packageURL)) {
        const r = await orig(input, opts).catch(function () { return null; });
        return r && r.ok ? r : orig(packageURL + pkg, opts);
      }
      if (!/python_stdlib\.zip(\?|$)/.test(url)) return orig(input, opts);
      const r = await orig(input, opts).catch(function () { return null; });
      if (r && r.ok) return r;
      const alt = await orig(url.replace(/python_stdlib\.zip.*$/, 'python_stdlib.b64.txt'));
      if (!alt.ok) return r || alt;
      const bin = atob((await alt.text()).replace(/\s+/g, ''));
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      return new Response(bytes, { status: 200, headers: { 'Content-Type': 'application/zip' } });
    };
  }

  async function init(msg) {
    installFetchFallbacks(msg.packageURL);
    const errors = [];
    for (const base of msg.indexURLs) {
      try {
        if (typeof root.loadPyodide !== 'function') await loadScript(base + 'pyodide.js');
        py = await root.loadPyodide({ indexURL: base, stdout: function () {}, stderr: function () {} });
        break;
      } catch (e) {
        errors.push(base + ' (' + (e && e.message ? e.message : e) + ')');
        py = null;
      }
    }
    if (!py) throw new Error('The Python runtime could not be loaded from ' + errors.join(' or '));
    py.FS.mkdirTree('/home/pyodide/quantlab');
    py.FS.writeFile('/home/pyodide/quantlab/engine.py', msg.engineSrc);
    py.runPython('import sys\nsys.path.insert(0, "/home/pyodide/quantlab")\nimport engine');
    handle = py.globals.get('engine').handle;
    return { version: py.version };
  }

  // Loads the packages that `code` imports. Failures surface later as a clear
  // ModuleNotFoundError from the strategy, so they are only remembered, not thrown.
  async function loadImports(code, progress) {
    if (!code || code === packagesFor) return;
    let failed = false;
    await py.loadPackagesFromImports(code, {
      messageCallback: function (m) { if (/^Loading /.test(m)) progress(m.replace(/^Loading /, 'Loading Python packages: ') + '…'); },
      errorCallback: function () { failed = true; },
    });
    if (!failed) packagesFor = code;
  }

  async function onMessage(msg, reply) {
    try {
      let result;
      if (msg.type === 'init') result = await init(msg);
      else if (msg.type === 'call') {
        if (!handle) throw new Error('Python is not ready yet');
        let code = null;
        try { code = JSON.parse(msg.payload).code; } catch (e) { /* no code in this call */ }
        await loadImports(code, function (text) { reply({ id: msg.id, progress: text }); });
        result = handle(msg.method, msg.payload);
      } else throw new Error('unknown message type ' + msg.type);
      reply({ id: msg.id, ok: true, result: result });
    } catch (e) {
      reply({ id: msg.id, ok: false, error: String(e && e.message ? e.message : e) });
    }
  }

  root.QLRuntime = { onMessage: onMessage };
})(typeof self !== 'undefined' ? self : this);
