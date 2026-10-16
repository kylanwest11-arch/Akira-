/*!
 * Akira embed: add Akira to any website with one tag.
 *
 *   <script src="https://YOUR-AKIRA-SERVER/embed.js" data-key="optional-api-key" defer></script>
 *
 * Options (data- attributes on the script tag):
 *   data-server    Akira server URL (defaults to where this script was loaded from)
 *   data-key       API key, if the server requires one
 *   data-position  "right" (default) or "left"
 *   data-open      "true" to start open
 *   data-launcher  "false" to hide the floating button and open it from your own code
 *
 * JavaScript API:
 *   Akira.open(), Akira.close(), Akira.toggle()
 *   Akira.send("text")                         ask in the widget
 *   Akira.chat({messages, onEvent})            headless: stream events to your own UI
 *   Akira.listen("partial text")               First-Hand Processing for your own input box
 */
(function () {
  "use strict";
  if (window.Akira && window.Akira.__loaded) return;

  var script = document.currentScript || document.querySelector('script[src*="embed.js"],script[src*="akira.js"]');
  var ds = (script && script.dataset) || {};
  var SERVER = (ds.server || (script ? new URL(script.src, location.href).origin : location.origin)).replace(/\/$/, "");
  var KEY = ds.key || "";
  var session = Math.random().toString(36).slice(2) + Date.now().toString(36);

  function headers() {
    var h = { "Content-Type": "application/json" };
    if (KEY) h.Authorization = "Bearer " + KEY;
    return h;
  }

  // ---- API ---------------------------------------------------------------
  function chat(opts) {
    var ctl = new AbortController();
    var body = {
      messages: opts.messages, images: opts.images || [], modality: opts.modality || "text",
      session: opts.session || null, stream: true
    };
    var done = fetch(SERVER + "/v1/chat", { method: "POST", headers: headers(), body: JSON.stringify(body), signal: ctl.signal })
      .then(function (res) {
        if (!res.ok) return res.json().catch(function () { return {}; }).then(function (j) {
          throw new Error(j.detail || ("Akira returned " + res.status));
        });
        var reader = res.body.getReader(), dec = new TextDecoder(), buf = "";
        function pump() {
          return reader.read().then(function (r) {
            if (r.done) return;
            buf += dec.decode(r.value, { stream: true });
            var parts = buf.split("\n\n");
            buf = parts.pop();
            parts.forEach(function (p) {
              if (p.indexOf("data: ") === 0) {
                try { opts.onEvent(JSON.parse(p.slice(6))); } catch (e) { /* ignore a malformed event */ }
              }
            });
            return pump();
          });
        }
        return pump();
      });
    return { done: done, abort: function () { ctl.abort(); } };
  }

  function listen(text, sid) {
    return fetch(SERVER + "/v1/listen", {
      method: "POST", headers: headers(), body: JSON.stringify({ session: sid || session, text: text })
    }).then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; });
  }

  // ---- widget --------------------------------------------------------------
  var CSS = [
    ":host{all:initial}",
    ".w{--bg:#fbfaf7;--panel:#ffffff;--fg:#1d1b26;--muted:#6e6a7c;--line:#e6e2ea;--accent:#d2452f;--accent-ink:#fff;--soft:#fbe9e5;--chip:#f1eef5;",
    "font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:var(--fg)}",
    "@media (prefers-color-scheme:dark){.w{--bg:#14131a;--panel:#1c1a24;--fg:#ece9f2;--muted:#9a95a8;--line:#2d2a38;--accent:#ff7a5c;--accent-ink:#1c0d08;--soft:#3a211c;--chip:#272433}}",
    ".launch{position:fixed;bottom:calc(20px + env(safe-area-inset-bottom,0px));width:56px;height:56px;border-radius:50%;border:0;cursor:pointer;",
    "background:var(--accent);color:var(--accent-ink);box-shadow:0 8px 24px rgba(0,0,0,.25);display:grid;place-items:center;z-index:2147483000}",
    ".launch svg{width:26px;height:26px}",
    ".panel{position:fixed;bottom:calc(88px + env(safe-area-inset-bottom,0px));width:min(400px,calc(100vw - 24px));height:min(620px,calc(100vh - 120px));",
    "background:var(--bg);border:1px solid var(--line);border-radius:18px;box-shadow:0 20px 60px rgba(0,0,0,.28);display:flex;flex-direction:column;overflow:hidden;z-index:2147483000}",
    ".right{right:max(12px,env(safe-area-inset-right,0px))}.left{left:max(12px,env(safe-area-inset-left,0px))}",
    ".panel[hidden],.launch[hidden],[hidden]{display:none!important}",
    "header{display:flex;align-items:center;gap:10px;padding:14px 16px;border-bottom:1px solid var(--line);background:var(--panel)}",
    ".dot{width:12px;height:12px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 4px var(--soft)}",
    "header b{font-size:16px;letter-spacing:.01em}header span{color:var(--muted);font-size:12px;margin-left:auto}",
    "header button{border:0;background:none;color:var(--muted);cursor:pointer;font-size:18px;padding:2px 6px;border-radius:6px}",
    ".log{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px}",
    ".m{max-width:88%;padding:9px 13px;border-radius:14px;overflow-wrap:anywhere}",
    ".u{align-self:flex-end;background:var(--accent);color:var(--accent-ink);border-bottom-right-radius:4px;white-space:pre-wrap}",
    ".a{align-self:flex-start;background:var(--panel);border:1px solid var(--line);border-bottom-left-radius:4px}",
    ".a p{margin:0 0 .5em}.a p:last-child{margin:0}.a blockquote{margin:.3em 0;padding-left:10px;border-left:3px solid var(--line);color:var(--muted)}",
    ".a ul{margin:.2em 0;padding-left:18px}.a code{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;background:var(--chip);padding:1px 4px;border-radius:4px}",
    ".a pre{background:var(--chip);padding:8px 10px;border-radius:8px;overflow-x:auto}.a pre code{background:none;padding:0}",
    ".meta{font-size:11px;color:var(--muted);margin-top:6px;display:flex;gap:6px;flex-wrap:wrap}",
    ".tag{background:var(--chip);border-radius:999px;padding:1px 8px}",
    ".think{color:var(--muted);font-style:italic}",
    ".gallery{display:grid;grid-template-columns:repeat(2,1fr);gap:6px;margin-top:6px}.gallery:empty{display:none}",
    ".gallery img{width:100%;border-radius:8px;image-rendering:auto;border:1px solid var(--line)}",
    ".fhp{display:flex;gap:6px;flex-wrap:wrap;padding:0 14px;min-height:0}",
    ".fhp .tag{font-size:11.5px;color:var(--muted);animation:in .2s ease}",
    "@keyframes in{from{opacity:0;transform:translateY(3px)}}",
    "form{display:flex;gap:6px;align-items:flex-end;padding:10px 12px calc(12px + env(safe-area-inset-bottom,0px));border-top:1px solid var(--line);background:var(--panel)}",
    "textarea{flex:1;resize:none;border:1px solid var(--line);border-radius:12px;padding:9px 12px;font:inherit;color:var(--fg);background:var(--bg);max-height:120px;outline:none}",
    "textarea:focus{border-color:var(--accent)}",
    ".ic{border:1px solid var(--line);background:var(--bg);color:var(--muted);width:38px;height:38px;border-radius:10px;cursor:pointer;display:grid;place-items:center;flex:none}",
    ".ic svg{width:18px;height:18px}.ic.on{background:var(--soft);color:var(--accent);border-color:var(--accent)}",
    ".send{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}",
    "video{width:100%;max-height:150px;object-fit:cover;background:#000}",
    ".thumbs{display:flex;gap:6px;padding:6px 12px 0}.thumbs img{width:44px;height:44px;object-fit:cover;border-radius:8px;border:1px solid var(--line)}",
    "button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}",
    "@media (prefers-reduced-motion:reduce){*{animation:none!important}}"
  ].join("\n");

  var ICON = {
    chat: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/></svg>',
    mic: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/></svg>',
    cam: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="6" width="13" height="12" rx="2"/><path d="m16 10 5-3v10l-5-3"/></svg>',
    img: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="m21 16-5-5-9 9"/></svg>',
    send: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M13 6l6 6-6 6"/></svg>',
    stop: '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="7" y="7" width="10" height="10" rx="1.5"/></svg>'
  };

  function esc(s) { return s.replace(/[&<>"']/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]; }); }
  function inline(s) {
    return esc(s).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>").replace(/\*([^*]+)\*/g, "<i>$1</i>");
  }
  function md(text) {
    var out = [], list = false, code = null;
    text.split("\n").forEach(function (line) {
      if (/^```/.test(line)) {
        if (code === null) { code = []; } else { out.push("<pre><code>" + esc(code.join("\n")) + "</code></pre>"); code = null; }
        return;
      }
      if (code !== null) { code.push(line); return; }
      var li = /^\s*[-*]\s+(.*)/.exec(line);
      if (li) { if (!list) { out.push("<ul>"); list = true; } out.push("<li>" + inline(li[1]) + "</li>"); return; }
      if (list) { out.push("</ul>"); list = false; }
      if (/^>\s?/.test(line)) out.push("<blockquote>" + inline(line.replace(/^>\s?/, "")) + "</blockquote>");
      else if (/^#{1,4}\s/.test(line)) out.push("<p><b>" + inline(line.replace(/^#+\s/, "")) + "</b></p>");
      else if (line.trim()) out.push("<p>" + inline(line) + "</p>");
    });
    if (list) out.push("</ul>");
    if (code !== null) out.push("<pre><code>" + esc(code.join("\n")) + "</code></pre>");
    return out.join("");
  }

  var host = document.createElement("div");
  host.id = "akira-widget";
  var root = host.attachShadow({ mode: "open" });
  var side = ds.position === "left" ? "left" : "right";
  root.innerHTML =
    "<style>" + CSS + "</style><div class='w'>" +
    "<button class='launch " + side + "' aria-label='Open Akira'>" + ICON.chat + "</button>" +
    "<section class='panel " + side + "' role='dialog' aria-label='Akira' hidden>" +
    "<header><div class='dot'></div><b>Akira</b><span class='st'>connecting…</span><button class='x' aria-label='Close'>×</button></header>" +
    "<video playsinline muted hidden></video>" +
    "<div class='log' aria-live='polite'></div>" +
    "<div class='fhp'></div><div class='thumbs' hidden></div>" +
    "<form><button type='button' class='ic img' aria-label='Add image'>" + ICON.img + "</button>" +
    "<button type='button' class='ic cam' aria-label='Camera (Cast)'>" + ICON.cam + "</button>" +
    "<button type='button' class='ic mic' aria-label='Talk (Live)'>" + ICON.mic + "</button>" +
    "<textarea rows='1' placeholder='Ask Akira…' aria-label='Message'></textarea>" +
    "<button type='submit' class='ic send' aria-label='Send'>" + ICON.send + "</button>" +
    "<input type='file' accept='image/*' hidden></form></section></div>";

  var $ = function (s) { return root.querySelector(s); };
  var launch = $(".launch"), panel = $(".panel"), log = $(".log"), input = $("textarea"), form = $("form");
  var fhpBox = $(".fhp"), st = $(".st"), video = $("video"), thumbs = $(".thumbs"), fileIn = $("input[type=file]");
  if (ds.launcher === "false") launch.hidden = true;

  var history = [], busy = null, pendingImages = [], camStream = null, voiceMode = false, liveNative = false;

  function open() { panel.hidden = false; launch.setAttribute("aria-expanded", "true"); input.focus(); }
  function close() { panel.hidden = true; launch.setAttribute("aria-expanded", "false"); }
  function toggle() { panel.hidden ? open() : close(); }
  launch.onclick = toggle;
  $(".x").onclick = close;

  fetch(SERVER + "/v1/status", { headers: headers() }).then(function (r) { return r.json(); }).then(function (s) {
    var n = Object.keys(s.native_brains || {}).length;
    liveNative = Object.keys(s.native_brains || {}).some(function (k) { return k.indexOf("live-") === 0; });
    if (liveNative) $(".mic").hidden = false;
    st.textContent = s.cloud ? "on-device + cloud" : n ? "on-device" : "setting up";
  }).catch(function () { st.textContent = "offline"; });

  function bubble(cls, html) {
    var d = document.createElement("div");
    d.className = "m " + cls;
    d.innerHTML = html;
    log.appendChild(d);
    log.scrollTop = log.scrollHeight;
    return d;
  }

  // First-Hand Processing: tell Akira what you are typing or saying, as you go
  var listenTimer = null, lastListened = "";
  function showUnderstood(snap) {
    if (!snap || !snap.understood) return;
    fhpBox.innerHTML = snap.understood.map(function (n) { return "<span class='tag'>" + esc(n.note) + "</span>"; }).join("");
  }
  function onDraft(text) {
    clearTimeout(listenTimer);
    listenTimer = setTimeout(function () {
      if (text.trim() === lastListened || text.trim().split(/\s+/).length < 2) return;
      lastListened = text.trim();
      listen(text, session).then(showUnderstood);
    }, 180);
  }
  input.addEventListener("input", function () {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 120) + "px";
    onDraft(input.value);
  });
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); form.requestSubmit(); }
  });

  function where(engine) {
    if (!engine) return "";
    if (engine === "device") return "answered on your device";
    if (engine.indexOf("native:") === 0) return "Akira on-device model";
    if (engine.indexOf("cloud:") === 0) return "Akira cloud";
    return "";
  }

  function send(text, modality) {
    text = (text || "").trim();
    if (busy || (!text && !pendingImages.length)) return;
    open();
    var images = pendingImages.slice();
    if (camStream) images.push(grabFrame());
    var mode = modality || (camStream ? "video" : voiceMode ? "audio" : "text");
    bubble("u", esc(text || "(image)"));
    history.push({ role: "user", content: text || "What's in this image?" });
    input.value = ""; input.style.height = "auto"; fhpBox.innerHTML = ""; lastListened = "";
    pendingImages = []; thumbs.hidden = true; thumbs.innerHTML = "";
    var a = bubble("a", "<span class='think'>Thinking…</span>");
    var meta = document.createElement("div"); meta.className = "meta";
    var gallery = document.createElement("div"); gallery.className = "gallery";
    var answer = "", tools = [];
    $(".send").innerHTML = ICON.stop; $(".send").setAttribute("aria-label", "Stop");
    busy = chat({
      messages: history.slice(-20), images: images.filter(Boolean), modality: mode, session: session,
      onEvent: function (ev) {
        if (ev.type === "text") { answer += ev.text; a.innerHTML = md(answer); a.appendChild(gallery); a.appendChild(meta); }
        else if (ev.type === "image") {
          var im = document.createElement("img");
          im.src = "data:" + ev.media_type + ";base64," + ev.data; im.alt = "Image drawn by Akira";
          gallery.appendChild(im); a.appendChild(gallery); a.appendChild(meta);
        }
        else if (ev.type === "tool") { tools.push(ev.name); }
        else if (ev.type === "understood" && !answer) { a.innerHTML = "<span class='think'>" + esc(ev.notes.map(function (n) { return n.note; }).join(" · ")) + "…</span>"; }
        else if (ev.type === "done") {
          meta.innerHTML = [where(ev.engine)].concat(tools.length ? ["used " + tools.filter(function (t, i) { return tools.indexOf(t) === i; }).join(", ").replace(/_/g, " ")] : [])
            .filter(Boolean).map(function (t) { return "<span class='tag'>" + esc(t) + "</span>"; }).join("");
          a.innerHTML = md(answer || "…"); a.appendChild(gallery); a.appendChild(meta);
        } else if (ev.type === "error") { answer += "\n\nSomething went wrong: " + ev.message; a.innerHTML = md(answer); }
        log.scrollTop = log.scrollHeight;
      }
    });
    busy.done.catch(function (e) {
      if (e.name !== "AbortError") { answer = answer || ("Akira couldn't be reached. " + e.message); a.innerHTML = md(answer); }
    }).then(function () {
      busy = null;
      $(".send").innerHTML = ICON.send; $(".send").setAttribute("aria-label", "Send");
      if (answer) history.push({ role: "assistant", content: answer });
      if (mode === "audio" || mode === "video") speak(answer);
      session = Math.random().toString(36).slice(2) + Date.now().toString(36);
    });
  }
  form.addEventListener("submit", function (e) {
    e.preventDefault();
    if (busy) { busy.abort(); return; }
    send(input.value);
  });

  // Images
  $(".img").onclick = function () { fileIn.click(); };
  fileIn.onchange = function () {
    Array.prototype.slice.call(fileIn.files, 0, 4).forEach(function (f) {
      var r = new FileReader();
      r.onload = function () {
        var url = String(r.result);
        pendingImages.push({ data: url.split(",")[1], media_type: f.type || "image/jpeg" });
        var im = document.createElement("img"); im.src = url; im.alt = f.name;
        thumbs.appendChild(im); thumbs.hidden = false;
      };
      r.readAsDataURL(f);
    });
    fileIn.value = "";
  };

  // Live: talk to Akira. Words are sent to First-Hand Processing while you speak.
  var SR = window.SpeechRecognition || window.webkitSpeechRecognition, rec = null;
  if (!SR) $(".mic").hidden = true;   // shown again if Akira's own speech brain is trained
  // Akira's own speech brain: record 16 kHz audio, re-transcribe it every second while you talk
  // (First-Hand Processing starts on the words so far), stop after a pause.
  var recording = null;
  function b64pcm(chunks) {
    var n = chunks.reduce(function (s, c) { return s + c.length; }, 0), pcm = new Int16Array(n), o = 0;
    chunks.forEach(function (c) { pcm.set(c, o); o += c.length; });
    var bytes = new Uint8Array(pcm.buffer), s = "";
    for (var i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return btoa(s);
  }
  function transcribe(chunks) {
    return fetch(SERVER + "/v1/transcribe", { method: "POST", headers: headers(), body: JSON.stringify({ audio: b64pcm(chunks) }) })
      .then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; });
  }
  function startLive() {
    navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } }).then(function (stream) {
      var ctx = new (window.AudioContext || window.webkitAudioContext)();
      var src = ctx.createMediaStreamSource(stream), proc = ctx.createScriptProcessor(4096, 1, 1);
      var ratio = ctx.sampleRate / 16000, chunks = [], heard = false, quietSince = Date.now(), busyT = false;
      recording = { stop: function (send_) {
        clearInterval(tick); proc.disconnect(); src.disconnect(); stream.getTracks().forEach(function (t) { t.stop(); }); ctx.close();
        $(".mic").classList.remove("on"); recording = null;
        if (send_ && chunks.length) transcribe(chunks).then(function (r) {
          var text = (r && r.text) || input.value;
          if (text.trim()) send(text, camStream ? "video" : "audio");
        });
      } };
      proc.onaudioprocess = function (e) {
        var x = e.inputBuffer.getChannelData(0), n = Math.floor(x.length / ratio), out = new Int16Array(n), energy = 0;
        for (var i = 0; i < n; i++) { var v = x[Math.floor(i * ratio)]; out[i] = Math.max(-1, Math.min(1, v)) * 32767; energy += v * v; }
        chunks.push(out);
        if (Math.sqrt(energy / n) > 0.015) { heard = true; quietSince = Date.now(); }
      };
      src.connect(proc); proc.connect(ctx.destination);
      var tick = setInterval(function () {
        if (heard && Date.now() - quietSince > 1400) { recording && recording.stop(true); return; }
        if (!heard || busyT) return;
        busyT = true;
        transcribe(chunks).then(function (r) { busyT = false; if (r && r.text && recording) { input.value = r.text; onDraft(r.text); } });
      }, 1000);
      $(".mic").classList.add("on");
    }).catch(function () { bubble("a", "<span class='think'>Microphone access was blocked, so Live can't hear you.</span>"); });
  }

  $(".mic").onclick = function () {
    if (recording) { recording.stop(true); return; }
    if (liveNative && navigator.mediaDevices) { window.speechSynthesis && speechSynthesis.cancel(); startLive(); return; }
    if (!SR) return;
    if (rec) { rec.stop(); return; }
    rec = new SR(); rec.interimResults = true; rec.continuous = false; rec.lang = document.documentElement.lang || "en-US";
    voiceMode = true; $(".mic").classList.add("on"); window.speechSynthesis && speechSynthesis.cancel();
    var finalText = "";
    rec.onresult = function (e) {
      var interim = "";
      for (var i = e.resultIndex; i < e.results.length; i++) {
        if (e.results[i].isFinal) finalText += e.results[i][0].transcript; else interim += e.results[i][0].transcript;
      }
      input.value = finalText + interim;
      onDraft(input.value);
    };
    rec.onend = function () {
      $(".mic").classList.remove("on"); rec = null;
      if (input.value.trim()) send(input.value, camStream ? "video" : "audio");
      voiceMode = false;
    };
    rec.onerror = function () { $(".mic").classList.remove("on"); rec = null; voiceMode = false; };
    rec.start();
  };
  function speak(text) {
    if (!window.speechSynthesis || !text) return;
    var plain = text.replace(/```[\s\S]*?```/g, "").replace(/[*_`>#-]/g, "").slice(0, 600);
    speechSynthesis.speak(new SpeechSynthesisUtterance(plain));
  }

  // Cast: show Akira what your camera sees
  if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) $(".cam").hidden = true;
  $(".cam").onclick = function () {
    if (camStream) {
      camStream.getTracks().forEach(function (t) { t.stop(); });
      camStream = null; video.hidden = true; $(".cam").classList.remove("on");
      return;
    }
    navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false }).then(function (s) {
      camStream = s; video.srcObject = s; video.hidden = false; video.play(); $(".cam").classList.add("on");
    }).catch(function () { bubble("a", "<span class='think'>Camera access was blocked, so Cast can't see anything.</span>"); });
  };
  function grabFrame() {
    if (!camStream || !video.videoWidth) return null;
    var c = document.createElement("canvas"), scale = Math.min(1, 1024 / video.videoWidth);
    c.width = video.videoWidth * scale; c.height = video.videoHeight * scale;
    c.getContext("2d").drawImage(video, 0, 0, c.width, c.height);
    return { data: c.toDataURL("image/jpeg", 0.8).split(",")[1], media_type: "image/jpeg" };
  }

  function mount() { document.body.appendChild(host); if (ds.open === "true") open(); }
  if (document.body) mount(); else document.addEventListener("DOMContentLoaded", mount);

  window.Akira = {
    __loaded: true, open: open, close: close, toggle: toggle,
    send: function (t) { send(t); },
    chat: chat, listen: function (t) { return listen(t); }, server: SERVER
  };
})();
