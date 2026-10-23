/*
 * QuantLab GitHub sync: an optional add-on, loaded only when you use it.
 *
 * Keeps your saved strategies and modules in a GitHub repository as plain .py
 * files, so they get version history, review and sharing:
 *
 *   <folder>/strategies/<name>.py   one file per saved strategy
 *   <folder>/modules/<name>.py      one file per module in the module library
 *   <folder>/quantlab.json          strategy names (file names are simplified)
 *
 * The browser talks to api.github.com directly with a token you provide; no
 * other server is involved. Push makes one commit with every file (files you
 * deleted in QuantLab stay in the repository). Pull reads the same layout.
 */
(function (root) {
  'use strict';
  const API = 'https://api.github.com';

  function slug(name) {
    return String(name).toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '') || 'strategy';
  }
  function encode(text) {   // UTF-8 text -> base64
    const bytes = new TextEncoder().encode(text); let bin = '';
    for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return btoa(bin);
  }
  function decode(b64) {    // base64 (GitHub wraps it at 60 characters) -> UTF-8 text
    const bin = atob(String(b64).replace(/\s+/g, ''));
    return new TextDecoder().decode(Uint8Array.from(bin, c => c.charCodeAt(0)));
  }
  function explain(status, msg) {
    if (status === 401) return 'GitHub didn’t accept the token (401). Check that it is complete and hasn’t expired.';
    if (status === 403) return 'The token isn’t allowed to do this (403). Give it “Contents: read and write” on this repository.' + (msg ? ' GitHub said: ' + msg : '');
    if (status === 404) return 'Repository or branch not found (404). Check the name and branch; private repositories need a token with access to them.';
    if (status === 422) return 'GitHub refused the change (422)' + (msg ? ': ' + msg : '.');
    return `GitHub returned ${status}${msg ? ': ' + msg : ''}.`;
  }
  async function gh(cfg, method, path, body) {
    let r;
    try {
      r = await fetch(API + path, {
        method, cache: 'no-store',
        headers: Object.assign({ Authorization: 'Bearer ' + cfg.token, Accept: 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28' },
          body ? { 'Content-Type': 'application/json' } : {}),
        body: body ? JSON.stringify(body) : undefined,
      });
    } catch (e) { throw new Error('Couldn’t reach api.github.com. Check your connection.'); }
    if (!r.ok) {
      let msg = ''; try { msg = (await r.json()).message || ''; } catch (e) { /* not JSON */ }
      const err = new Error(explain(r.status, msg)); err.status = r.status; err.ghMessage = msg; throw err;
    }
    return r.status === 204 ? null : r.json();
  }
  function repoPath(cfg) {
    if (!/^[\w.-]+\/[\w.-]+$/.test(cfg.repo || '')) throw new Error('Enter the repository as owner/name, e.g. octocat/quant-research.');
    return '/repos/' + cfg.repo;
  }
  const folder = cfg => (cfg.dir || '').replace(/^\/+|\/+$/g, '');
  const join = (dir, p) => (dir ? dir + '/' + p : p);

  // The branch's latest commit & tree, or null when the repository has no commits yet.
  // (The branches endpoints answer 200 for an empty repository, where git/refs answers 409.)
  async function head(cfg) {
    const repo = repoPath(cfg), branch = cfg.branch || 'main';
    const any = await gh(cfg, 'GET', `${repo}/branches?per_page=1`);
    if (!any.length) return null;
    let b;
    try { b = await gh(cfg, 'GET', `${repo}/branches/${encodeURIComponent(branch)}`); }
    catch (e) { if (e.status === 404) throw new Error(`The repository has no branch named ${branch}. Create it on GitHub or pick another.`); throw e; }
    return { commit: b.commit.sha, tree: b.commit.commit.tree.sha };
  }

  // Files for a push: {path: text}.
  function layout(cfg, strategies, modules) {
    const dir = folder(cfg), out = {}, names = {}, used = new Set();
    for (const name of Object.keys(strategies).sort()) {
      let s = slug(name), k = 2; while (used.has(s)) s = slug(name) + '_' + k++;
      used.add(s); names[s] = name; out[join(dir, `strategies/${s}.py`)] = strategies[name];
    }
    for (const name of Object.keys(modules).sort()) out[join(dir, `modules/${name}.py`)] = modules[name];
    out[join(dir, 'quantlab.json')] = JSON.stringify({ format: 'quantlab-repo', version: 1, strategies: names, modules: Object.keys(modules).sort() }, null, 2) + '\n';
    return out;
  }

  async function push(cfg, { strategies, modules }, message) {
    const repo = repoPath(cfg), branch = cfg.branch || 'main', want = layout(cfg, strategies, modules);
    let base = await head(cfg);
    if (!base) {   // the Git Data API can't write to an empty repository: create a first commit with the contents API
      await gh(cfg, 'PUT', `${repo}/contents/${join(folder(cfg), 'quantlab.json').split('/').map(encodeURIComponent).join('/')}`,
        { message: 'QuantLab: start the strategy folder', content: encode(want[join(folder(cfg), 'quantlab.json')]), branch });
      base = await head(cfg);
    }
    const tree = await gh(cfg, 'POST', `${repo}/git/trees`, {
      base_tree: base.tree, tree: Object.entries(want).map(([path, content]) => ({ path, mode: '100644', type: 'blob', content })),
    });
    const files = Object.keys(want).length;
    if (tree.sha === base.tree) return { unchanged: true, files, commit: base.commit };
    const commit = await gh(cfg, 'POST', `${repo}/git/commits`, { message: message || 'QuantLab: update strategies and modules', tree: tree.sha, parents: [base.commit] });
    await gh(cfg, 'PATCH', `${repo}/git/refs/heads/${encodeURIComponent(branch)}`, { sha: commit.sha });
    return { unchanged: false, files, commit: commit.sha, url: commit.html_url || `https://github.com/${cfg.repo}/commit/${commit.sha}` };
  }

  async function pull(cfg) {
    const repo = repoPath(cfg), base = await head(cfg), out = { strategies: {}, modules: {}, commit: base && base.commit };
    if (!base) return out;
    const dir = folder(cfg), pre = dir ? dir + '/' : '';
    const tree = await gh(cfg, 'GET', `${repo}/git/trees/${base.tree}?recursive=1`);
    const blobs = (tree.tree || []).filter(e => e.type === 'blob' && e.path.startsWith(pre));
    const read = async e => decode((await gh(cfg, 'GET', `${repo}/git/blobs/${e.sha}`)).content);
    const man = blobs.find(e => e.path === pre + 'quantlab.json');
    let names = {};
    if (man) { try { names = JSON.parse(await read(man)).strategies || {}; } catch (e) { /* use the file names */ } }
    for (const e of blobs) {
      const rest = e.path.slice(pre.length), m = rest.match(/^(strategies|modules)\/([^/]+)\.py$/); if (!m) continue;
      if (m[1] === 'strategies') out.strategies[names[m[2]] || m[2]] = await read(e);
      else if (/^[A-Za-z_]\w*$/.test(m[2])) out.modules[m[2]] = await read(e);
    }
    if (tree.truncated) out.truncated = true;
    return out;
  }

  root.QLGitHub = { push, pull, layout, slug, encode, decode };
})(typeof self !== 'undefined' ? self : this);
