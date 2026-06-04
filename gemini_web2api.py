#!/usr/bin/env python3
"""
gemini-web2api - Gemini Web to OpenAI API proxy.

Converts Google Gemini's web interface into an OpenAI-compatible API server.
Zero authentication required. Works on any platform (Windows/macOS/Linux).

Usage:
    pip install httpx
    python gemini_web2api.py [--port 8081] [--config config.json]

Client configuration (Cherry Studio, ChatBox, etc.):
    Base URL: http://localhost:8081/v1
    API Key: (anything or empty)

How it works:
    Sends requests directly to Gemini's public StreamGenerate endpoint.
    The backend does not verify authentication for basic text generation.
    Model selection via MODE_CATEGORY field [79] in the request payload.
    This is NOT a user-tier spoofing attack - the endpoint simply doesn't
    require auth for anonymous access.
"""
import json
import urllib.request
import urllib.parse
import urllib.error
import time
import ssl
import sys
import uuid
import re
import os
import hashlib
import argparse
import base64
import threading
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

__version__ = "1.1.0"

# ─── Configuration ───────────────────────────────────────────────────────────

# Optional inline Gemini Web cookie for local single-file use.
# Keep these blank in public repositories. Paste real cookies only in a private copy.
INLINE_COOKIE = ""
INLINE_SAPISID = ""

DEFAULT_CONFIG = {
    "port": 8081,
    "host": "0.0.0.0",
    "retry_attempts": 3,
    "retry_delay_sec": 2,
    "request_timeout_sec": 180,
    "gemini_bl": "boq_assistant-bard-web-server_20260525.09_p0",
    "auth_user": None,
    "xsrf_token": None,
    "default_model": "gemini-3.5-flash",
    "log_requests": True,
    "cookie_file": None,
    "proxy": None,
    "api_keys": [],
    "empty_response_policy": "error",
}

CONFIG = dict(DEFAULT_CONFIG)

STATS_LOCK = threading.Lock()
REQUEST_STATS = {
    "started_at": int(time.time()),
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "streaming_requests": 0,
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_tokens": 0,
    "by_model": {},
    "recent": deque(maxlen=50),
}

# ─── Models ──────────────────────────────────────────────────────────────────
# Mapping from JS source: MODE_CATEGORY enum (028-6eb337387583.js)
#   1=FAST, 2=THINKING, 3=PRO, 4=AUTO, 5=FAST_DYNAMIC_THINKING, 6=FLASH_LITE

MODELS = {
    "gemini-3.5-flash": {
        "mode": 1, "think": 4,
        "desc": "Fast general-purpose model",
    },
    "gemini-3.5-flash-thinking": {
        "mode": 2, "think": 0,
        "desc": "Deep thinking mode, longest output (~20k chars)",
    },
    "gemini-3.1-pro": {
        "mode": 3, "think": 4,
        "desc": "Pro model (requires cookie for real routing)",
    },
    "gemini-auto": {
        "mode": 4, "think": 4,
        "desc": "Auto model selection",
    },
    "gemini-3.5-flash-thinking-lite": {
        "mode": 5, "think": 0,
        "desc": "Dynamic thinking with adaptive depth",
    },
    "gemini-flash-lite": {
        "mode": 6, "think": 4,
        "desc": "Lightweight fast model",
    },
}

# ─── Utilities ───────────────────────────────────────────────────────────────

def log(msg: str):
    if CONFIG["log_requests"]:
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        sys.stderr.flush()


def estimate_tokens(text: str) -> int:
    """Rough token estimate used by the OpenAI-compatible usage fields."""
    return len(text or "") // 4


def record_usage(endpoint: str, model: str, prompt: str, completion: str,
                 success: bool, stream: bool, started_at: float, error: str = None):
    prompt_tokens = estimate_tokens(prompt)
    completion_tokens = estimate_tokens(completion)
    total_tokens = prompt_tokens + completion_tokens
    latency_ms = int((time.time() - started_at) * 1000) if started_at else None
    model_name = model or "unknown"

    with STATS_LOCK:
        REQUEST_STATS["total_requests"] += 1
        if success:
            REQUEST_STATS["successful_requests"] += 1
        else:
            REQUEST_STATS["failed_requests"] += 1
        if stream:
            REQUEST_STATS["streaming_requests"] += 1
        REQUEST_STATS["total_prompt_tokens"] += prompt_tokens
        REQUEST_STATS["total_completion_tokens"] += completion_tokens
        REQUEST_STATS["total_tokens"] += total_tokens

        by_model = REQUEST_STATS["by_model"].setdefault(model_name, {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        })
        by_model["requests"] += 1
        by_model["successes" if success else "failures"] += 1
        by_model["prompt_tokens"] += prompt_tokens
        by_model["completion_tokens"] += completion_tokens
        by_model["total_tokens"] += total_tokens

        item = {
            "time": int(time.time()),
            "endpoint": endpoint,
            "model": model_name,
            "status": "ok" if success else "error",
            "stream": bool(stream),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_ms": latency_ms,
        }
        if error:
            item["error"] = str(error)[:180]
        REQUEST_STATS["recent"].append(item)


def stats_snapshot() -> dict:
    with STATS_LOCK:
        by_model = []
        for name, data in REQUEST_STATS["by_model"].items():
            row = dict(data)
            row["model"] = name
            by_model.append(row)
        by_model.sort(key=lambda row: row["total_tokens"], reverse=True)

        return {
            "started_at": REQUEST_STATS["started_at"],
            "uptime_sec": max(0, int(time.time()) - REQUEST_STATS["started_at"]),
            "total_requests": REQUEST_STATS["total_requests"],
            "successful_requests": REQUEST_STATS["successful_requests"],
            "failed_requests": REQUEST_STATS["failed_requests"],
            "streaming_requests": REQUEST_STATS["streaming_requests"],
            "total_prompt_tokens": REQUEST_STATS["total_prompt_tokens"],
            "total_completion_tokens": REQUEST_STATS["total_completion_tokens"],
            "total_tokens": REQUEST_STATS["total_tokens"],
            "by_model": by_model,
            "recent": list(reversed(REQUEST_STATS["recent"])),
            "requires_api_key": bool(CONFIG.get("api_keys")),
            "version": __version__,
        }


def dashboard_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>gemini-web2api dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7f4;
      --ink: #17201b;
      --muted: #647067;
      --line: #d7ddd6;
      --panel: #ffffff;
      --accent: #1b6b55;
      --accent-2: #ba5b35;
      --soft: #eaf0ea;
      --danger: #a13a34;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
    }
    main {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 44px;
    }
    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 18px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 18px;
    }
    h1 {
      margin: 0;
      font-size: 30px;
      font-weight: 700;
      letter-spacing: 0;
    }
    .sub {
      margin-top: 6px;
      color: var(--muted);
      font-size: 14px;
      font-family: "Trebuchet MS", Verdana, sans-serif;
    }
    .keybox {
      display: flex;
      gap: 8px;
      align-items: center;
      min-width: min(420px, 100%);
      font-family: "Trebuchet MS", Verdana, sans-serif;
    }
    input {
      width: 100%;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 10px 12px;
      font: inherit;
    }
    button {
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      padding: 10px 14px;
      cursor: pointer;
      font: inherit;
      white-space: nowrap;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 22px 0;
    }
    .metric, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .metric {
      padding: 16px;
      min-height: 112px;
    }
    .metric span, th {
      color: var(--muted);
      font-family: "Trebuchet MS", Verdana, sans-serif;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .08em;
    }
    .metric strong {
      display: block;
      margin-top: 18px;
      font-size: 30px;
      line-height: 1;
    }
    .split {
      display: grid;
      grid-template-columns: 1fr 1.4fr;
      gap: 12px;
    }
    .panel {
      overflow: hidden;
    }
    .panel h2 {
      margin: 0;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      font-size: 17px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-family: "Trebuchet MS", Verdana, sans-serif;
      font-size: 13px;
    }
    th, td {
      padding: 12px 14px;
      text-align: left;
      border-bottom: 1px solid var(--soft);
      vertical-align: top;
    }
    tr:last-child td { border-bottom: 0; }
    .num { text-align: right; font-variant-numeric: tabular-nums; }
    .ok { color: var(--accent); font-weight: 700; }
    .error { color: var(--danger); font-weight: 700; }
    .bar {
      height: 7px;
      border-radius: 999px;
      background: var(--soft);
      overflow: hidden;
      margin-top: 7px;
    }
    .bar i {
      display: block;
      height: 100%;
      width: var(--w, 0%);
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
    }
    .empty {
      padding: 24px 16px;
      color: var(--muted);
      font-family: "Trebuchet MS", Verdana, sans-serif;
    }
    footer {
      margin-top: 16px;
      color: var(--muted);
      font-family: "Trebuchet MS", Verdana, sans-serif;
      font-size: 12px;
    }
    @media (max-width: 860px) {
      header, .split { display: block; }
      .keybox { margin-top: 16px; }
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .panel { margin-top: 12px; }
    }
    @media (max-width: 520px) {
      main { width: min(100vw - 20px, 1180px); padding-top: 18px; }
      .grid { grid-template-columns: 1fr; }
      h1 { font-size: 24px; }
      th, td { padding: 10px 8px; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>gemini-web2api dashboard</h1>
        <div class="sub" id="status">Loading runtime stats...</div>
      </div>
      <div class="keybox" id="keybox" hidden>
        <input id="apiKey" type="password" placeholder="API key">
        <button id="saveKey" type="button">Connect</button>
      </div>
    </header>

    <section class="grid">
      <div class="metric"><span>Total requests</span><strong id="totalRequests">0</strong></div>
      <div class="metric"><span>Total tokens</span><strong id="totalTokens">0</strong></div>
      <div class="metric"><span>Success rate</span><strong id="successRate">0%</strong></div>
      <div class="metric"><span>Uptime</span><strong id="uptime">0s</strong></div>
    </section>

    <section class="split">
      <div class="panel">
        <h2>Models</h2>
        <div id="modelsEmpty" class="empty">No model usage yet.</div>
        <table id="modelsTable" hidden>
          <thead><tr><th>Model</th><th class="num">Req</th><th class="num">Tokens</th></tr></thead>
          <tbody id="modelsBody"></tbody>
        </table>
      </div>
      <div class="panel">
        <h2>Recent requests</h2>
        <div id="recentEmpty" class="empty">No requests yet.</div>
        <table id="recentTable" hidden>
          <thead><tr><th>Time</th><th>Model</th><th>Status</th><th class="num">Tokens</th><th class="num">Latency</th></tr></thead>
          <tbody id="recentBody"></tbody>
        </table>
      </div>
    </section>
    <footer>Token counts are rough estimates from text length. Prompt and response contents are not stored.</footer>
  </main>
  <script>
    const keybox = document.getElementById('keybox');
    const apiKeyInput = document.getElementById('apiKey');
    const saveKey = document.getElementById('saveKey');
    const savedKey = localStorage.getItem('gemini_web2api_dashboard_key') || '';
    apiKeyInput.value = savedKey;
    saveKey.onclick = () => {
      localStorage.setItem('gemini_web2api_dashboard_key', apiKeyInput.value.trim());
      loadStats();
    };

    function fmt(n) {
      return new Intl.NumberFormat().format(n || 0);
    }
    function uptime(seconds) {
      seconds = Math.max(0, seconds || 0);
      const d = Math.floor(seconds / 86400);
      const h = Math.floor((seconds % 86400) / 3600);
      const m = Math.floor((seconds % 3600) / 60);
      if (d) return d + 'd ' + h + 'h';
      if (h) return h + 'h ' + m + 'm';
      if (m) return m + 'm';
      return seconds + 's';
    }
    function textCell(value, className) {
      const td = document.createElement('td');
      td.textContent = value == null ? '' : value;
      if (className) td.className = className;
      return td;
    }
    async function loadStats() {
      const headers = {};
      const key = localStorage.getItem('gemini_web2api_dashboard_key') || '';
      if (key) headers.Authorization = 'Bearer ' + key;
      const res = await fetch('/dashboard/data', { headers });
      if (res.status === 401) {
        keybox.hidden = false;
        document.getElementById('status').textContent = 'Enter your configured API key to view stats.';
        return;
      }
      const data = await res.json();
      keybox.hidden = !data.requires_api_key;
      document.getElementById('status').textContent = 'v' + data.version + ' running for ' + uptime(data.uptime_sec);
      document.getElementById('totalRequests').textContent = fmt(data.total_requests);
      document.getElementById('totalTokens').textContent = fmt(data.total_tokens);
      const rate = data.total_requests ? Math.round((data.successful_requests / data.total_requests) * 100) : 0;
      document.getElementById('successRate').textContent = rate + '%';
      document.getElementById('uptime').textContent = uptime(data.uptime_sec);

      const modelsBody = document.getElementById('modelsBody');
      modelsBody.textContent = '';
      const maxTokens = Math.max(1, ...data.by_model.map(row => row.total_tokens || 0));
      data.by_model.forEach(row => {
        const tr = document.createElement('tr');
        const model = textCell(row.model);
        const bar = document.createElement('div');
        bar.className = 'bar';
        bar.style.setProperty('--w', Math.round((row.total_tokens || 0) / maxTokens * 100) + '%');
        bar.innerHTML = '<i></i>';
        model.appendChild(bar);
        tr.appendChild(model);
        tr.appendChild(textCell(fmt(row.requests), 'num'));
        tr.appendChild(textCell(fmt(row.total_tokens), 'num'));
        modelsBody.appendChild(tr);
      });
      document.getElementById('modelsEmpty').hidden = data.by_model.length > 0;
      document.getElementById('modelsTable').hidden = data.by_model.length === 0;

      const recentBody = document.getElementById('recentBody');
      recentBody.textContent = '';
      data.recent.forEach(row => {
        const tr = document.createElement('tr');
        tr.appendChild(textCell(new Date(row.time * 1000).toLocaleTimeString()));
        tr.appendChild(textCell(row.model));
        tr.appendChild(textCell(row.status, row.status === 'ok' ? 'ok' : 'error'));
        tr.appendChild(textCell(fmt(row.total_tokens), 'num'));
        tr.appendChild(textCell(row.latency_ms == null ? '' : row.latency_ms + 'ms', 'num'));
        recentBody.appendChild(tr);
      });
      document.getElementById('recentEmpty').hidden = data.recent.length > 0;
      document.getElementById('recentTable').hidden = data.recent.length === 0;
    }
    loadStats().catch(err => {
      document.getElementById('status').textContent = 'Dashboard error: ' + err.message;
    });
    setInterval(() => loadStats().catch(() => {}), 3000);
  </script>
</body>
</html>"""


class GeminiUpstreamError(RuntimeError):
    """Gemini Web returned an error payload or no usable text."""

    def __init__(self, message: str, code: str = None, raw_excerpt: str = None):
        super().__init__(message)
        self.code = code
        self.raw_excerpt = raw_excerpt


def load_cookie() -> tuple:
    """Load cookie from inline constants or file. Returns (cookie_str, sapisid)."""
    inline_cookie = INLINE_COOKIE.strip()
    if inline_cookie:
        try:
            return parse_cookie_content(inline_cookie, INLINE_SAPISID)
        except Exception as e:
            log(f"Inline cookie parse error: {e}")
            return "", None

    cookie_file = CONFIG.get("cookie_file")
    if not cookie_file:
        return "", None
    if not os.path.exists(cookie_file):
        return "", None
    try:
        with open(cookie_file, "r") as f:
            content = f.read().strip()
        return parse_cookie_content(content)
    except Exception as e:
        log(f"Cookie load error: {e}")
        return "", None


def parse_cookie_content(content: str, sapisid_override: str = "") -> tuple:
    """Parse plain or JSON cookie content into (cookie_str, sapisid)."""
    content = (content or "").strip()
    if not content:
        return "", None

    if content.startswith("{"):
        data = json.loads(content)
        cookie_str = (data.get("cookie") or "").strip()
        sapisid = (data.get("sapisid") or "").strip()
    else:
        cookie_str = content
        pairs = {}
        for part in cookie_str.split(";"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            pairs[key.strip()] = value.strip()
        sapisid = pairs.get("SAPISID", "")

    if sapisid_override:
        sapisid = sapisid_override.strip()
    return cookie_str, sapisid if sapisid else None


def make_sapisidhash(sapisid: str) -> str:
    ts = int(time.time())
    h = hashlib.sha1(f"{ts} {sapisid} https://gemini.google.com".encode()).hexdigest()
    return f"SAPISIDHASH {ts}_{h}"


def account_prefix() -> str:
    """Return the Gemini account path prefix for non-default Google accounts."""
    auth_user = CONFIG.get("auth_user")
    if auth_user is None or auth_user == "":
        return ""
    return f"/u/{auth_user}"


# ─── Gemini Protocol ─────────────────────────────────────────────────────────

def gemini_stream_generate(prompt: str, model_id: int, think_mode: int) -> str:
    """Send prompt to Gemini StreamGenerate with retry."""
    inner = [None] * 80
    inner[0] = [prompt, 0, None, None, None, None, 0]
    inner[1] = ["en"]
    inner[2] = ["", "", "", None, None, None, None, None, None, ""]
    inner[6] = [0]
    inner[7] = 1
    inner[10] = 1
    inner[11] = 0
    inner[17] = [[think_mode]]
    inner[18] = 0
    inner[27] = 1
    inner[30] = [4]
    inner[41] = [2]
    inner[53] = 0
    inner[59] = str(uuid.uuid4())
    inner[61] = []
    inner[68] = 1
    inner[79] = model_id

    outer = [None, json.dumps(inner)]
    params = {"f.req": json.dumps(outer)}
    if CONFIG.get("xsrf_token"):
        params["at"] = CONFIG["xsrf_token"]
    body = urllib.parse.urlencode(params).encode()
    reqid = int(time.time()) % 1000000
    prefix = account_prefix()
    url = (
        f"https://gemini.google.com{prefix}/_/BardChatUi/data/"
        "assistant.lamda.BardFrontendService/StreamGenerate"
        f"?bl={CONFIG['gemini_bl']}&hl=en&_reqid={reqid}&rt=c"
    )
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://gemini.google.com",
        "Referer": f"https://gemini.google.com{prefix}/app",
        "X-Same-Domain": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    if prefix:
        headers["X-Goog-AuthUser"] = str(CONFIG["auth_user"])

    cookie_str, sapisid = load_cookie()
    if cookie_str:
        headers["Cookie"] = cookie_str
    if sapisid:
        headers["Authorization"] = make_sapisidhash(sapisid)

    last_err = None
    for attempt in range(CONFIG["retry_attempts"]):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            ctx = ssl.create_default_context()
            proxy = CONFIG.get("proxy")
            if proxy:
                opener = urllib.request.build_opener(
                    urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
                    urllib.request.HTTPSHandler(context=ctx)
                )
                resp = opener.open(req, timeout=CONFIG["request_timeout_sec"])
            else:
                resp = urllib.request.urlopen(req, context=ctx, timeout=CONFIG["request_timeout_sec"])
            return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            raise_http_upstream_error(e.code, raw)
        except Exception as e:
            if isinstance(e, GeminiUpstreamError):
                raise
            last_err = e
            if attempt < CONFIG["retry_attempts"] - 1:
                log(f"Retry {attempt+1}/{CONFIG['retry_attempts']}: {e}")
                time.sleep(CONFIG["retry_delay_sec"])
    raise last_err


def gemini_stream_generate_iter(prompt: str, model_id: int, think_mode: int):
    """Send prompt and yield incremental text deltas using httpx streaming."""
    inner = [None] * 80
    inner[0] = [prompt, 0, None, None, None, None, 0]
    inner[1] = ["en"]
    inner[2] = ["", "", "", None, None, None, None, None, None, ""]
    inner[6] = [0]
    inner[7] = 1
    inner[10] = 1
    inner[11] = 0
    inner[17] = [[think_mode]]
    inner[18] = 0
    inner[27] = 1
    inner[30] = [4]
    inner[41] = [2]
    inner[53] = 0
    inner[59] = str(uuid.uuid4())
    inner[61] = []
    inner[68] = 1
    inner[79] = model_id

    outer = [None, json.dumps(inner)]
    params = {"f.req": json.dumps(outer)}
    if CONFIG.get("xsrf_token"):
        params["at"] = CONFIG["xsrf_token"]
    body = urllib.parse.urlencode(params)
    reqid = int(time.time()) % 1000000
    prefix = account_prefix()
    url = (
        f"https://gemini.google.com{prefix}/_/BardChatUi/data/"
        "assistant.lamda.BardFrontendService/StreamGenerate"
        f"?bl={CONFIG['gemini_bl']}&hl=en&_reqid={reqid}&rt=c"
    )
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://gemini.google.com",
        "Referer": f"https://gemini.google.com{prefix}/app",
        "X-Same-Domain": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    if prefix:
        headers["X-Goog-AuthUser"] = str(CONFIG["auth_user"])
    cookie_str, sapisid = load_cookie()
    if cookie_str:
        headers["Cookie"] = cookie_str
    if sapisid:
        headers["Authorization"] = make_sapisidhash(sapisid)

    proxy = CONFIG.get("proxy")

    if not HAS_HTTPX:
        # Fallback: non-streaming with urllib
        raw = gemini_stream_generate(prompt, model_id, think_mode)
        text = ensure_response_text(extract_response_text(raw), raw, stream=True)
        if text:
            yield text
        return

    prev_text = ""
    raw_tail = ""
    yielded = False
    transport = httpx.HTTPTransport(proxy=proxy) if proxy else None
    with httpx.Client(transport=transport, timeout=CONFIG["request_timeout_sec"], verify=True) as client:
        with client.stream("POST", url, content=body, headers=headers) as resp:
            if resp.status_code >= 400:
                raw = resp.read().decode("utf-8", errors="replace")
                raise_http_upstream_error(resp.status_code, raw, stream=True)
            buf = ""
            for chunk in resp.iter_text():
                raw_tail = (raw_tail + chunk)[-4000:]
                buf += chunk
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    if '"wrb.fr"' not in line or len(line) < 200:
                        continue
                    try:
                        arr = json.loads(line)
                        inner_str = arr[0][2]
                        if not inner_str or len(inner_str) < 50:
                            continue
                        inner2 = json.loads(inner_str)
                        if isinstance(inner2, list) and len(inner2) > 4 and inner2[4]:
                            for part in inner2[4]:
                                if isinstance(part, list) and len(part) > 1 and part[1] and isinstance(part[1], list):
                                    for t in part[1]:
                                        if isinstance(t, str) and len(t) > len(prev_text):
                                            delta = t[len(prev_text):]
                                            delta = clean_gemini_text(delta)
                                            if delta:
                                                yielded = True
                                                yield delta
                                            prev_text = t
                    except (json.JSONDecodeError, IndexError, TypeError):
                        pass
    if not yielded:
        ensure_response_text("", raw_tail, stream=True)


def clean_gemini_text(text: str) -> str:
    """Remove internal code execution artifacts."""
    text = re.sub(
        r'```(?:python|javascript|text)\?code_(?:reference|stdout)&code_event_index=\d+\n.*?```\n?',
        '', text, flags=re.DOTALL
    )
    return text.strip()


def _raw_excerpt(raw: str, limit: int = 500) -> str:
    compact = " ".join((raw or "").split())
    return compact[:limit]


def parse_upstream_error(raw: str) -> GeminiUpstreamError:
    """Return a structured upstream error if Gemini Web sent one."""
    if not raw:
        return None
    m = re.search(r'BardErrorInfo\s*\[([^\]]+)\]', raw)
    if m:
        code = m.group(1).strip()
        return GeminiUpstreamError(
            f"Gemini Web upstream returned BardErrorInfo [{code}]",
            code=f"bard_error_{code}",
            raw_excerpt=_raw_excerpt(raw),
        )
    return None


def ensure_response_text(text: str, raw: str, stream: bool = False) -> str:
    """Validate that a Gemini response has usable text unless configured otherwise."""
    if text:
        return text
    upstream_error = parse_upstream_error(raw)
    if upstream_error:
        log(f"Upstream Gemini error: {upstream_error}; excerpt={upstream_error.raw_excerpt}")
        raise upstream_error
    policy = (CONFIG.get("empty_response_policy") or "error").lower()
    if policy == "null":
        return text
    kind = "streaming response" if stream else "response"
    err = GeminiUpstreamError(
        f"Gemini Web upstream returned an empty {kind}",
        code="empty_response",
        raw_excerpt=_raw_excerpt(raw),
    )
    log(f"Upstream Gemini empty {kind}; excerpt={err.raw_excerpt}")
    raise err


def raise_http_upstream_error(status_code: int, raw: str, stream: bool = False):
    upstream_error = parse_upstream_error(raw)
    if upstream_error:
        log(f"Upstream Gemini HTTP {status_code}: {upstream_error}; excerpt={upstream_error.raw_excerpt}")
        raise upstream_error
    kind = "streaming response" if stream else "response"
    err = GeminiUpstreamError(
        f"Gemini Web upstream returned HTTP {status_code} for {kind}",
        code=f"http_{status_code}",
        raw_excerpt=_raw_excerpt(raw),
    )
    log(f"Upstream Gemini HTTP {status_code}; excerpt={err.raw_excerpt}")
    raise err


def extract_response_text(raw: str) -> str:
    """Parse StreamGenerate response to extract final text."""
    texts = []
    for line in raw.split("\n"):
        if '"wrb.fr"' not in line or len(line) < 200:
            continue
        try:
            arr = json.loads(line)
            inner_str = arr[0][2]
            if not inner_str or len(inner_str) < 50:
                continue
            inner = json.loads(inner_str)
            if isinstance(inner, list) and len(inner) > 4 and inner[4]:
                for part in inner[4]:
                    if isinstance(part, list) and len(part) > 1 and part[1]:
                        if isinstance(part[1], list):
                            for t in part[1]:
                                if isinstance(t, str) and len(t) > 0:
                                    texts.append(t)
        except (json.JSONDecodeError, IndexError, TypeError):
            pass
    text = ""
    for t in reversed(texts):
        if t.strip():
            text = t
            break
    return clean_gemini_text(text)


# ─── OpenAI Format Helpers ───────────────────────────────────────────────────

def messages_to_prompt(messages: list, tools: list = None) -> str:
    """Convert OpenAI messages to prompt string."""
    parts = []
    if tools:
        tool_defs = []
        for tool in tools:
            fn = tool.get("function", tool) if tool.get("type") == "function" else tool
            tool_defs.append({
                "name": fn.get("name", tool.get("name", "")),
                "description": fn.get("description", tool.get("description", "")),
                "parameters": fn.get("parameters", tool.get("parameters", {})),
            })
        if tool_defs:
            parts.append(
                "[System instruction]: You have access to tools. "
                "To call a tool, respond with:\n"
                '```tool_call\n{"name": "func_name", "arguments": {...}}\n```\n'
                "Only use tool_call blocks when needed.\n\n"
                f"Available tools:\n{json.dumps(tool_defs, indent=2)}"
            )
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") for c in content
                if c.get("type") in ("text", "input_text")
            )
        if role == "system":
            parts.append(f"[System instruction]: {content}")
        elif role == "assistant":
            if msg.get("tool_calls"):
                tc_strs = []
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    tc_strs.append(
                        f'```tool_call\n{{"name": "{fn.get("name")}", '
                        f'"arguments": {fn.get("arguments", "{}")}}}\n```'
                    )
                parts.append(f"[Assistant]: {content or ''}\n" + "\n".join(tc_strs))
            else:
                parts.append(f"[Assistant]: {content}")
        elif role == "tool":
            parts.append(f"[Tool result for {msg.get('name', '')}]: {content}")
        else:
            parts.append(content if content else "")
    return "\n\n".join(p for p in parts if p)


def parse_tool_calls(text: str) -> tuple:
    """Extract tool_call blocks. Returns (clean_text, tool_calls_list)."""
    tool_calls = []
    pattern = r'```tool_call\s*\n(.*?)\n```'
    for match in re.findall(pattern, text, re.DOTALL):
        try:
            data = json.loads(match.strip())
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": data["name"],
                    "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False),
                },
            })
        except (json.JSONDecodeError, KeyError):
            pass
    clean = re.sub(pattern, '', text, flags=re.DOTALL).strip()
    return clean, tool_calls


# ─── HTTP Handler ────────────────────────────────────────────────────────────

class GeminiHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log(fmt % args)

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, html: str, status=200):
        body = html.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message: str, status=500, code: str = None):
        error = {"message": message}
        if code:
            error["code"] = code
        self.send_json({"error": error}, status)

    def _send_upstream_error(self, exc: Exception):
        code = exc.code if isinstance(exc, GeminiUpstreamError) else None
        self.send_error_json(f"upstream error: {exc}", 502, code)

    def _authorized(self):
        keys = CONFIG.get("api_keys") or []
        if not keys:
            return True
        auth = self.headers.get("Authorization", "")
        key = auth[7:] if auth.startswith("Bearer ") else self.headers.get("x-api-key", "")
        return key in keys

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        try:
            path = urllib.parse.urlparse(self.path).path
            if path in ("/dashboard", "/dashboard/"):
                self.send_html(dashboard_html())
                return
            if path == "/dashboard/data":
                if not self._authorized():
                    self.send_json({"error": {"message": "invalid api key"}}, 401)
                    return
                self.send_json(stats_snapshot())
                return
            if path.startswith("/v1/") and not self._authorized():
                self.send_json({"error": {"message": "invalid api key"}}, 401)
                return
            if path == "/v1/models":
                self.send_json({"object": "list", "data": [
                    {"id": n, "object": "model", "created": 1700000000,
                     "owned_by": "google", "description": c["desc"]}
                    for n, c in MODELS.items()
                ]})
            elif path.startswith("/v1beta/models"):
                self._handle_google_models_list()
            elif path == "/":
                self.send_json({"status": "ok", "version": __version__,
                                "models": list(MODELS.keys()),
                                "dashboard": "/dashboard"})
            else:
                self.send_json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            log(f"GET error: {e}")

    def do_POST(self):
        try:
            if self.path.startswith("/v1/") and not self._authorized():
                self.send_json({"error": {"message": "invalid api key"}}, 401)
                return
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""
            if self.path == "/v1/chat/completions":
                self.handle_chat(body)
            elif self.path == "/v1/responses":
                self.handle_responses(body)
            elif ":generateContent" in self.path:
                self._handle_google_generate(body, stream=False)
            elif ":streamGenerateContent" in self.path:
                self._handle_google_generate(body, stream=True)
            else:
                self.send_json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            log(f"POST error: {e}")
            try:
                self.send_json({"error": {"message": str(e)}}, 500)
            except:
                pass

    def _resolve_model(self, model_name):
        think_override = None
        if "@think=" in model_name:
            model_name, think_str = model_name.rsplit("@think=", 1)
            think_override = int(think_str)
        cfg = MODELS.get(model_name)
        if not cfg:
            return None, None, None, f"Unknown model: {model_name}"
        return model_name, cfg["mode"], (think_override if think_override is not None else cfg["think"]), None

    def _call_gemini(self, prompt, model_id, think_mode, tools):
        raw = gemini_stream_generate(prompt, model_id, think_mode)
        text = ensure_response_text(extract_response_text(raw), raw)
        tool_calls = None
        if tools and text:
            text, tool_calls = parse_tool_calls(text)
        return text or "", tool_calls

    def handle_chat(self, body: bytes):
        req = json.loads(body)
        model_name, model_id, think_mode, err = self._resolve_model(
            req.get("model", CONFIG["default_model"]))
        if err:
            self.send_json({"error": {"message": err}}, 400)
            return

        tools = req.get("tools")
        prompt = messages_to_prompt(req.get("messages", []), tools)
        if not prompt.strip():
            self.send_json({"error": {"message": "empty prompt"}}, 400)
            return

        stream = req.get("stream", False)
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        started_at = time.time()

        if stream and not tools:
            # True streaming: forward chunks as they arrive
            completion_parts = []
            try:
                stream_iter = gemini_stream_generate_iter(prompt, model_id, think_mode)
                try:
                    first_delta = next(stream_iter)
                    completion_parts.append(first_delta)
                except StopIteration:
                    record_usage("/v1/chat/completions", model_name, prompt, "", False, stream, started_at, "empty streaming response")
                    self.send_error_json("upstream error: Gemini Web upstream returned an empty streaming response", 502, "empty_response")
                    return
                except Exception as e:
                    record_usage("/v1/chat/completions", model_name, prompt, "", False, stream, started_at, str(e))
                    self._send_upstream_error(e)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                         "model": model_name, "choices": [{"index": 0, "delta": {"content": first_delta}, "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
                for delta_text in stream_iter:
                    completion_parts.append(delta_text)
                    chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                             "model": model_name, "choices": [{"index": 0, "delta": {"content": delta_text}, "finish_reason": None}]}
                    self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
                # Final chunk
                chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                         "model": model_name, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                record_usage("/v1/chat/completions", model_name, prompt, "".join(completion_parts), True, stream, started_at)
            except (BrokenPipeError, ConnectionResetError):
                record_usage("/v1/chat/completions", model_name, prompt, "".join(completion_parts), False, stream, started_at, "client disconnected")
            except Exception as e:
                record_usage("/v1/chat/completions", model_name, prompt, "".join(completion_parts), False, stream, started_at, str(e))
                log(f"Stream error: {e}")
            return

        # Non-streaming (or tool calling which needs full response)
        try:
            text, tool_calls = self._call_gemini(prompt, model_id, think_mode, tools)
        except Exception as e:
            record_usage("/v1/chat/completions", model_name, prompt, "", False, stream, started_at, str(e))
            self._send_upstream_error(e)
            return
        record_usage("/v1/chat/completions", model_name, prompt, text or "", True, stream, started_at)

        msg = {"role": "assistant", "content": text or None}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        finish = "tool_calls" if tool_calls else "stop"

        if stream:
            # Stream mode with tools: send as single chunk (need full parse for tool_calls)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                     "model": model_name, "choices": [{"index": 0, "delta": msg, "finish_reason": finish}]}
            self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            self.send_json({
                "id": cid, "object": "chat.completion", "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": estimate_tokens(prompt), "completion_tokens": estimate_tokens(text),
                          "total_tokens": estimate_tokens(prompt) + estimate_tokens(text)},
            })

    def handle_responses(self, body: bytes):
        """OpenAI Responses API for Codex CLI compatibility."""
        req = json.loads(body)
        model_name, model_id, think_mode, err = self._resolve_model(
            req.get("model", CONFIG["default_model"]))
        if err:
            self.send_json({"error": {"message": err}}, 400)
            return

        input_items = req.get("input", [])
        tools = req.get("tools")

        messages = []
        if req.get("instructions"):
            messages.append({"role": "system", "content": req["instructions"]})
        if isinstance(input_items, str):
            messages.append({"role": "user", "content": input_items})
        elif isinstance(input_items, list):
            for item in input_items:
                if isinstance(item, str):
                    messages.append({"role": "user", "content": item})
                elif isinstance(item, dict):
                    if item.get("type") == "function_call_output":
                        messages.append({"role": "tool", "tool_call_id": item.get("call_id", ""),
                                         "name": item.get("name", ""), "content": item.get("output", "")})
                    elif item.get("role") == "assistant" or (item.get("type") == "message" and item.get("role") == "assistant"):
                        cp = item.get("content", [])
                        text_acc, tc_list = "", []
                        if isinstance(cp, list):
                            for c in cp:
                                if isinstance(c, dict):
                                    if c.get("type") == "output_text": text_acc += c.get("text", "")
                                    elif c.get("type") == "function_call": tc_list.append(c)
                        elif isinstance(cp, str):
                            text_acc = cp
                        m = {"role": "assistant", "content": text_acc or None}
                        if tc_list:
                            m["tool_calls"] = [{"id": tc.get("call_id", f"call_{i}"), "type": "function",
                                                "function": {"name": tc.get("name",""), "arguments": tc.get("arguments","{}")}}
                                               for i, tc in enumerate(tc_list)]
                        messages.append(m)
                    else:
                        role = item.get("role", "user")
                        content = item.get("content", "")
                        if isinstance(content, list):
                            content = " ".join(c.get("text", "") for c in content if c.get("type") in ("text", "input_text"))
                        messages.append({"role": role, "content": content})

        if tools:
            tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("parameters", {})}}
                     if t.get("type") == "function" and "function" not in t else t for t in tools]

        prompt = messages_to_prompt(messages, tools)
        if not prompt.strip():
            self.send_json({"error": {"message": "empty input"}}, 400)
            return

        started_at = time.time()
        stream = bool(req.get("stream"))
        try:
            text, tool_calls = self._call_gemini(prompt, model_id, think_mode, tools)
        except Exception as e:
            record_usage("/v1/responses", model_name, prompt, "", False, stream, started_at, str(e))
            self._send_upstream_error(e)
            return
        record_usage("/v1/responses", model_name, prompt, text or "", True, stream, started_at)

        rid = f"resp_{uuid.uuid4().hex[:16]}"
        mid = f"msg_{uuid.uuid4().hex[:12]}"
        output = []
        if tool_calls:
            for tc in tool_calls:
                output.append({"type": "function_call", "id": tc["id"], "call_id": tc["id"],
                               "name": tc["function"]["name"], "arguments": tc["function"]["arguments"], "status": "completed"})
        if text or not tool_calls:
            output.append({"type": "message", "id": mid, "role": "assistant", "status": "completed",
                           "content": [{"type": "output_text", "text": text or "", "annotations": []}]})

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            ev = {"type": "response.created", "response": {"id": rid, "object": "response", "status": "in_progress", "model": model_name, "output": []}}
            self.wfile.write(f"event: response.created\ndata: {json.dumps(ev)}\n\n".encode())
            for item in output:
                if item["type"] == "function_call":
                    ev = {"type": "response.function_call_arguments.done", "item_id": item["id"], "call_id": item["call_id"], "name": item["name"], "arguments": item["arguments"]}
                    self.wfile.write(f"event: response.function_call_arguments.done\ndata: {json.dumps(ev)}\n\n".encode())
                elif item["type"] == "message":
                    for ci, cp in enumerate(item["content"]):
                        ev = {"type": "response.output_text.done", "item_id": item["id"], "content_index": ci, "text": cp["text"]}
                        self.wfile.write(f"event: response.output_text.done\ndata: {json.dumps(ev)}\n\n".encode())
            resp_obj = {"id": rid, "object": "response", "status": "completed", "model": model_name, "output": output,
                        "usage": {"input_tokens": estimate_tokens(prompt), "output_tokens": estimate_tokens(text),
                                  "total_tokens": estimate_tokens(prompt) + estimate_tokens(text)}}
            self.wfile.write(f"event: response.completed\ndata: {json.dumps({'type': 'response.completed', 'response': resp_obj})}\n\n".encode())
            self.wfile.flush()
        else:
            self.send_json({"id": rid, "object": "response", "created_at": int(time.time()), "status": "completed",
                            "model": model_name, "output": output,
                            "usage": {"input_tokens": estimate_tokens(prompt), "output_tokens": estimate_tokens(text),
                                      "total_tokens": estimate_tokens(prompt) + estimate_tokens(text)}})


    # ─── Google Native API (Gemini CLI compatible) ────────────────────────────

    def _parse_google_model_from_path(self):
        """Extract model name from /v1beta/models/{model}:method path."""
        m = re.match(r'/v1beta/models/([^:?]+)', self.path)
        if m:
            return m.group(1)
        return None

    def _handle_google_models_list(self):
        """GET /v1beta/models — Google AI format model list."""
        models = []
        for name, cfg in MODELS.items():
            models.append({
                "name": f"models/{name}",
                "displayName": name,
                "description": cfg["desc"],
                "supportedGenerationMethods": ["generateContent", "streamGenerateContent"],
            })
        self.send_json({"models": models})

    def _google_contents_to_prompt(self, req: dict) -> str:
        """Convert Google API contents format to prompt string."""
        parts = []
        sys_inst = req.get("systemInstruction")
        if sys_inst:
            sys_parts = sys_inst.get("parts", [])
            sys_text = " ".join(p.get("text", "") for p in sys_parts if p.get("text"))
            if sys_text:
                parts.append(f"[System instruction]: {sys_text}")

        for content in req.get("contents", []):
            role = content.get("role", "user")
            text_parts = []
            for p in content.get("parts", []):
                if p.get("text"):
                    text_parts.append(p["text"])
            text = " ".join(text_parts)
            if role == "model":
                parts.append(f"[Assistant]: {text}")
            else:
                parts.append(text)
        return "\n\n".join(p for p in parts if p)

    def _handle_google_generate(self, body: bytes, stream: bool):
        """Handle Google native generateContent / streamGenerateContent."""
        req = json.loads(body)
        model_name = self._parse_google_model_from_path()
        if not model_name:
            self.send_json({"error": {"message": "model not specified in path"}}, 400)
            return

        model_name, model_id, think_mode, err = self._resolve_model(model_name)
        if err:
            self.send_json({"error": {"message": err}}, 400)
            return

        prompt = self._google_contents_to_prompt(req)
        if not prompt.strip():
            self.send_json({"error": {"message": "empty content"}}, 400)
            return

        endpoint = "/v1beta/models:streamGenerateContent" if stream else "/v1beta/models:generateContent"
        started_at = time.time()
        try:
            text, _ = self._call_gemini(prompt, model_id, think_mode, None)
        except Exception as e:
            record_usage(endpoint, model_name, prompt, "", False, stream, started_at, str(e))
            self._send_upstream_error(e)
            return
        record_usage(endpoint, model_name, prompt, text or "", True, stream, started_at)

        candidate = {
            "content": {"parts": [{"text": text or ""}], "role": "model"},
            "finishReason": "STOP",
            "index": 0,
        }
        usage = {
            "promptTokenCount": estimate_tokens(prompt),
            "candidatesTokenCount": estimate_tokens(text),
            "totalTokenCount": estimate_tokens(prompt) + estimate_tokens(text),
        }
        response_obj = {
            "candidates": [candidate],
            "usageMetadata": usage,
            "modelVersion": model_name,
        }

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(f"data: {json.dumps(response_obj)}\n\n".encode())
            self.wfile.flush()
        else:
            self.send_json(response_obj)


# ─── Main ────────────────────────────────────────────────────────────────────

def load_config(path: str):
    if path and os.path.exists(path):
        with open(path) as f:
            CONFIG.update(json.load(f))
        log(f"Config loaded: {path}")


def main():
    parser = argparse.ArgumentParser(description="Gemini Web to OpenAI API")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--cookie-file", type=str, default=None, help="Path to cookie file")
    parser.add_argument("--proxy", type=str, default=None, help="HTTP proxy, e.g. http://127.0.0.1:7890")
    parser.add_argument("--version", action="version", version=f"gemini-web2api {__version__}")
    args = parser.parse_args()

    config_path = args.config or os.environ.get("GEMINI_WEB2API_CONFIG")
    if not config_path:
        for p in ["./config.json", os.path.expanduser("~/.config/gemini-web2api/config.json")]:
            if os.path.exists(p):
                config_path = p
                break
    load_config(config_path)

    if args.port:
        CONFIG["port"] = args.port
    if args.cookie_file:
        CONFIG["cookie_file"] = args.cookie_file
    if args.proxy:
        CONFIG["proxy"] = args.proxy

    class ThreadedServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    port = CONFIG["port"]
    server = ThreadedServer((CONFIG["host"], port), GeminiHandler)
    print(f"gemini-web2api v{__version__}")
    print(f"  Listening: http://0.0.0.0:{port}")
    print(f"  Base URL:  http://localhost:{port}/v1")
    print(f"  Dashboard: http://localhost:{port}/dashboard")
    print(f"  Models:    {', '.join(MODELS.keys())}")
    if INLINE_COOKIE.strip():
        cookie_status = "yes (inline)"
    elif CONFIG.get("cookie_file"):
        cookie_status = f"yes ({CONFIG['cookie_file']})"
    else:
        cookie_status = "none (anonymous)"
    print(f"  Cookie:    {cookie_status}")
    print(f"  Proxy:     {CONFIG.get('proxy') or 'none (uses system env HTTP_PROXY/HTTPS_PROXY)'}")
    print(f"  Retry:     {CONFIG['retry_attempts']}x / {CONFIG['retry_delay_sec']}s")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()


if __name__ == "__main__":
    main()
