const app = document.querySelector('#app');
const statusEl = document.querySelector('#serviceStatus');
const newButton = document.querySelector('#newButton');
let previewUrl = null, hostedVideoUrl = null, hosted = false, maxUploadBytes = 250 * 1024 * 1024;
let file = null, report = null, selected = null, filter = 'all', poll = null, video = null;

const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const clock = ms => { const s = Math.max(0,Math.floor(ms/1000)); return `${Math.floor(s/60).toString().padStart(2,'0')}:${(s%60).toString().padStart(2,'0')}`; };
const api = async (path, options) => { const response = await fetch(path, options); const body = await response.json(); if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`); return body; };
const title = id => id.split('_').map(s => s[0].toUpperCase()+s.slice(1)).join(' ');

const hostedDb = () => new Promise((resolve, reject) => {
  const request = indexedDB.open('videolint-reviews', 1);
  request.onupgradeneeded = () => request.result.createObjectStore('reviews', {keyPath:'id'});
  request.onsuccess = () => resolve(request.result);
  request.onerror = () => reject(request.error);
});
async function hostedStore(method, value) {
  const db = await hostedDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('reviews', method === 'get' || method === 'getAll' ? 'readonly' : 'readwrite');
    const request = method === 'getAll' ? tx.objectStore('reviews').getAll() : tx.objectStore('reviews')[method](value);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
    tx.oncomplete = () => db.close();
  });
}
function releaseHostedVideo() { if(hostedVideoUrl){URL.revokeObjectURL(hostedVideoUrl); hostedVideoUrl = null;} }

async function health() {
  try {
    const h = await api('/api/health');
    hosted = Boolean(h.hosted);
    maxUploadBytes = h.maxUploadBytes || maxUploadBytes;
    const provider = h.aiProvider === 'jev' ? 'Jev' : h.aiProvider === 'gemini' ? 'Gemini' : h.aiProvider;
    statusEl.classList.remove('unavailable');
    statusEl.hidden = hosted;
    statusEl.textContent = hosted ? '' : `Engine ready · ${h.aiConfigured ? `${provider} configured` : `${provider} key missing`}${h.aiProvider === 'jev' && !h.transcriptionConfigured ? ' · Gemini transcription key missing' : ''}`;
  } catch { statusEl.hidden = false; statusEl.classList.add('unavailable'); statusEl.textContent = 'Engine unavailable'; }
}

async function home() {
  clearInterval(poll); poll = null; report = null; selected = null; file = null; video = null; filter = 'all'; newButton.hidden = true;
  releasePreview();
  releaseHostedVideo();
  history.replaceState({}, '', '/');
  app.innerHTML = `<section class="landing">
    <div class="home-heading"><h1>A second look.<br>A <span>better final cut.</span></h1><p class="intro">Catch the gaps, broken thoughts, and missed details.<br>Review your edit with evidence at every timestamp.</p><div class="hero-actions"><a class="subtle" href="#library">Your reviews</a><button class="primary" id="heroImport">Review a video <span aria-hidden="true">→</span></button></div></div>
    <section class="editor" aria-label="Video review workspace"><div class="editor-top"><span class="project-name"><i class="project-dot"></i><span id="projectName">Untitled review</span></span><span class="editor-tag">VideoLint workspace</span></div>
    <div class="editor-grid">
      <aside class="media-panel"><h2 class="panel-title">Media library <small>${hosted?'This browser':'Local'}</small></h2><button class="media-import" id="importMedia"><span aria-hidden="true">＋</span> Import video</button><div class="media-list" id="mediaList"></div><p class="media-note">MP4 or MOV · up to ${hosted?'4':'250'} MB<br>Your previous reviews live here.</p></aside>
      <div class="preview-panel"><div class="preview-surface"><div class="dropzone" id="drop"><div class="film-ring" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i></div><strong>${hosted?'Review your final cut':'Review your video with AI'}</strong><p>Drop your final cut here to get started.</p><button id="choose" type="button">Import media <span aria-hidden="true">↑</span></button><small>MP4 or MOV · up to ${hosted?'4':'250'} MB</small></div><div id="selectedFile"></div><input id="fileInput" type="file" accept="video/mp4,video/quicktime,.mp4,.mov" hidden></div><div class="preview-controls"><span id="previewDuration">0:00 / 0:00</span><div class="transport" aria-hidden="true"><span>Ⅰ◀</span><span>▷</span><span>▶Ⅰ</span></div><span class="format" id="previewFormat">FIT</span></div></div>
      <aside class="assistant-panel"><h2 class="panel-title">Review assistant <small>✦</small></h2><div class="assistant-welcome"><span class="assistant-icon" aria-hidden="true">✦</span><p>Hi! What should this edit accomplish?</p></div><div class="brief-presets"><button type="button" data-goal="Make a concise, clear social video under 60 seconds. Keep the story moving and avoid repeated ideas.">A short social video</button><button type="button" data-goal="Create a clear interview edit with complete thoughts, natural pacing, and no awkward gaps.">A polished interview</button></div><div class="composer"><label class="goal-label" for="goal">Your editing goal</label><textarea id="goal" class="goal-input" maxlength="2000" placeholder="Describe the audience, story, or length you’re aiming for…"></textarea><button class="primary" id="analyze" disabled>Review video <span aria-hidden="true">↑</span></button></div><p class="brief-note">${hosted?'Hosted reviews run technical checks. Reports and videos are saved in this browser.':'Technical checks run locally. Speech and editorial review use your configured AI provider.'}</p><p class="status-line" id="uploadStatus" role="alert"></p></aside>
    </div>
    <div class="home-timeline" aria-label="Preview timeline"><div class="timeline-toolbar"><strong>Timeline</strong><span id="timelineMessage">Import a video to begin</span></div><div class="timeline-ruler"><span>00:00</span><span>00:15</span><span>00:30</span><span>00:45</span><span>01:00</span></div><div class="track-row"><span class="track-label">VIDEO</span><div class="track-bed" id="videoTrack"><span class="track-placeholder"></span></div></div><div class="track-row"><span class="track-label">FINDINGS</span><div class="track-bed"></div></div><div class="timeline-footnote">Your findings will appear at their timestamps after review.</div></div></section>
    <section class="library-section" id="library"><div class="section-heading"><h2>Pick up where you left off</h2><span>Your recent reviews</span></div><div class="history" id="history"><div class="history-empty">Loading your reviews…</div></div></section>
  </section>`;
  const input = document.querySelector('#fileInput'), drop = document.querySelector('#drop');
  ['choose','heroImport','importMedia'].forEach(id => document.getElementById(id).onclick = () => input.click());
  input.onchange = () => choose(input.files[0]);
  drop.ondragover = e => {e.preventDefault();drop.classList.add('dragging')};
  drop.ondragleave = () => drop.classList.remove('dragging');
  drop.ondrop = e => {e.preventDefault();drop.classList.remove('dragging');choose(e.dataTransfer.files[0])};
  const updateReady = () => { document.querySelector('#analyze').disabled = !file || document.querySelector('#goal').value.trim().length < 5; };
  document.querySelector('#goal').oninput = updateReady;
  document.querySelectorAll('[data-goal]').forEach(button => button.onclick = () => {document.querySelector('#goal').value = button.dataset.goal; updateReady(); document.querySelector('#goal').focus();});
  document.querySelector('#analyze').onclick = upload;
  const transport = document.querySelector('.transport');
  transport.removeAttribute('aria-hidden');
  transport.innerHTML = '<button id="previewBack" aria-label="Back 5 seconds" disabled>Ⅰ◀</button><button id="previewPlay" aria-label="Play preview" disabled>▷</button><button id="previewForward" aria-label="Forward 5 seconds" disabled>▶Ⅰ</button>';
  try {
    const jobs = hosted ? (await hostedStore('getAll')).map(x => x.report).sort((a,b)=>b.createdAt-a.createdAt) : await api('/api/analysis');
    const historyEl = document.querySelector('#history'); if (!historyEl) return;
    historyEl.innerHTML = jobs.length ? jobs.slice(0,6).map(j => `<div class="history-row"><span class="history-icon" aria-hidden="true">▷</span><button data-id="${esc(j.id)}">${esc(j.filename)}<small>${new Date(j.createdAt*1000).toLocaleDateString(undefined,{month:'short',day:'numeric',year:'numeric'})}</small></button><span class="history-status">${esc(j.status)}</span></div>`).join('') : '<div class="history-empty">Your first review starts here.<p>Import a video above. Completed reviews will be saved in this workspace.</p></div>';
    document.querySelector('#mediaList').innerHTML = jobs.slice(0,4).map(j => `<button class="media-item" data-id="${esc(j.id)}" title="${esc(j.filename)}"><span class="media-symbol" aria-hidden="true">▷</span><span class="media-name">${esc(j.filename)}<small>${esc(j.status)}</small></span></button>`).join('');
    document.querySelectorAll('[data-id]').forEach(b => b.onclick = () => openReport(b.dataset.id));
  } catch { const el = document.querySelector('#history'); if(el) el.innerHTML = '<div class="history-empty">Unable to load your reviews. Check that the local server is running.</div>'; }
}

function releasePreview() {
  const preview = document.querySelector('#uploadPreview');
  if (preview) { preview.onloadedmetadata = preview.ontimeupdate = preview.onerror = preview.onplay = preview.onpause = null; preview.pause(); preview.removeAttribute('src'); preview.load(); }
  if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }
}

function choose(candidate) {
  if (!candidate) return;
  const error = document.querySelector('#uploadStatus');
  if (!/\.(mp4|mov)$/i.test(candidate.name) || candidate.size > maxUploadBytes) { error.textContent = `Choose an MP4/MOV under ${hosted?'4':'250'} MB.`; return; }
  error.textContent = ''; file = candidate;
  releasePreview();
  previewUrl = URL.createObjectURL(file);
  document.querySelector('#drop').hidden = true;
  document.querySelector('#selectedFile').innerHTML = `<video class="upload-preview" id="uploadPreview" controls playsinline preload="metadata"></video><div class="file-selected"><div><strong>${esc(file.name)}</strong><small>${(file.size/1024/1024).toFixed(1)} MB</small></div><button class="subtle" id="changeFile">Replace</button></div>`;
  const preview = document.querySelector('#uploadPreview'); preview.src = previewUrl;
  document.querySelectorAll('.transport button').forEach(button => button.disabled = false);
  const playButton = document.querySelector('#previewPlay');
  playButton.onclick = () => { if (preview.paused) preview.play().catch(() => { error.textContent = 'This video cannot be previewed in your browser. You can still submit it for review.'; }); else preview.pause(); };
  preview.onplay = () => { playButton.textContent = 'Ⅱ'; playButton.setAttribute('aria-label','Pause preview'); };
  preview.onpause = () => { playButton.textContent = '▷'; playButton.setAttribute('aria-label','Play preview'); };
  document.querySelector('#previewBack').onclick = () => { preview.currentTime = Math.max(0, preview.currentTime - 5); };
  document.querySelector('#previewForward').onclick = () => { if(Number.isFinite(preview.duration)) preview.currentTime = Math.min(preview.duration, preview.currentTime + 5); };
  preview.onloadedmetadata = () => {document.querySelector('#previewDuration').textContent = `0:00 / ${clock(preview.duration*1000)}`;document.querySelector('#previewFormat').textContent = `${preview.videoWidth} × ${preview.videoHeight}`;document.querySelectorAll('.timeline-ruler span').forEach((tick,i) => tick.textContent = clock(preview.duration*1000*i/4));};
  preview.ontimeupdate = () => {document.querySelector('#previewDuration').textContent = `${clock(preview.currentTime*1000)} / ${clock(preview.duration*1000)}`;};
  preview.onerror = () => {error.textContent = 'Preview is unavailable for this codec. You can still submit the video for review.';};
  document.querySelector('#projectName').textContent = file.name;
  document.querySelector('#videoTrack').innerHTML = `<span class="track-clip">${esc(file.name)}</span>`;
  document.querySelector('#timelineMessage').textContent = 'Ready for review';
  document.querySelector('#analyze').disabled = document.querySelector('#goal').value.trim().length < 5;
  document.querySelector('#changeFile').onclick = () => document.querySelector('#fileInput').click();
}

async function upload() {
  const goal = document.querySelector('#goal').value.trim(), error = document.querySelector('#uploadStatus');
  if (!file || goal.length < 5) { error.textContent = 'Select a video and describe its goal (at least 5 characters).'; return; }
  const button = document.querySelector('#analyze'); button.disabled = true; button.textContent = 'Uploading…'; error.textContent = '';
  try {
    const result = await api('/api/analysis', {method:'POST', headers:{'Content-Type':'application/octet-stream','X-Filename':encodeURIComponent(file.name),'X-Goal':encodeURIComponent(goal)},body:file});
    if (hosted) {
      await hostedStore('put', {id: result.id, report: result, file});
      report = result;
      releasePreview();
      hostedVideoUrl = URL.createObjectURL(file);
      newButton.hidden = false;
      history.replaceState({}, '', `/?analysis=${result.id}`);
      render();
    } else await openReport(result.id);
  } catch (e) { error.textContent = e.message; button.disabled = false; button.textContent = 'Review video ↑'; }
}

async function openReport(id) {
  releasePreview();
  releaseHostedVideo();
  clearInterval(poll); poll = null; video = null; newButton.hidden = false;
  history.replaceState({}, '', `/?analysis=${id}`);
  if (hosted) {
    try {
      const item = await hostedStore('get', id);
      if (!item) throw new Error('This review is not saved in this browser.');
      report = item.report;
      hostedVideoUrl = URL.createObjectURL(item.file);
      render();
    } catch(e) {app.innerHTML=`<div class="alert">${esc(e.message)}</div>`;}
    return;
  }
  try { report = await api(`/api/analysis/${id}`); render();
    if (!['complete','failed'].includes(report.status)) poll = setInterval(async () => {try { report = await api(`/api/analysis/${id}`); render(); if (['complete','failed'].includes(report.status)) {clearInterval(poll);poll=null;} } catch(e) {clearInterval(poll);poll=null;app.innerHTML=`<div class="alert">${esc(e.message)}</div>`;}}, 1200);
  } catch(e) { app.innerHTML=`<div class="alert">${esc(e.message)}</div>`; }
}

function render() {
  if (!report) return;
  const meta = report.metadata, done = report.status === 'complete', failed = report.status === 'failed';
  const issues = report.issues || [];
  if (!done && !failed) {
    app.innerHTML = `<section class="review"><div class="eyebrow">ANALYSIS / ${esc(report.status.toUpperCase())}</div><h1>${esc(report.filename)}</h1><p class="review-goal">Goal: ${esc(report.goal)}</p><div class="processing"><span class="pulse"></span><h2>${esc(title(report.status))}…</h2><p>${report.checkers.length} checker${report.checkers.length===1?'':'s'} finished. The report updates as actual stages complete.</p>${report.checkers.map(c=>`<div class="stage-row"><span class="state ${esc(c.status)}">${esc(c.status)}</span><div>${esc(title(c.checkerId))}</div></div>`).join('')}</div></section>`;
    return;
  }
  app.innerHTML = `<section class="review"><div class="review-top"><div><div class="eyebrow">${done ? 'ANALYSIS COMPLETE' : failed ? 'ANALYSIS FAILED' : `ANALYSIS / ${esc(report.status.toUpperCase())}`}</div><h1>${esc(report.filename)}</h1><p class="review-goal">Goal: ${esc(report.goal)}</p></div><div class="review-tools"><button class="ghost" id="download">↓ JSON report</button><button class="ghost" id="delete">Delete</button></div></div>${failed ? `<div class="alert">${esc(report.error)}</div>` : ''}${!done && !failed ? `<div class="processing"><span class="pulse"></span><h2>${esc(title(report.status))}…</h2><p>Measured stages only. This report updates as checkers finish.</p></div>` : ''}${meta ? `<div class="metric-strip"><div class="metric"><strong>${clock(meta.durationMs)}</strong><span class="metric-label">Duration</span></div><div class="metric"><strong>${meta.width || '?'}×${meta.height || '?'}</strong><span class="metric-label">Resolution</span></div><div class="metric"><strong>${issues.length}</strong><span class="metric-label">Findings</span></div><div class="metric"><strong>${report.checkers.filter(c=>c.status==='success').length}</strong><span class="metric-label">Checks run</span></div></div><div class="workspace"><div><div class="player-panel"><div class="video-wrap"><video id="video" src="/api/analysis/${report.id}/video" controls playsinline preload="metadata"></video></div><div class="player-meta"><span>${esc(meta.videoCodec)} ${meta.fps ? `· ${meta.fps} fps` : ''}</span><span>${esc(meta.audioCodec || 'No audio stream')}</span></div><div class="timeline-area"><div class="timeline-head"><strong>Issue timeline</strong><span class="time-label">${clock(meta.durationMs)}</span></div><div class="timeline" id="timeline" role="slider" tabindex="0" aria-label="Video timeline" aria-valuemin="0" aria-valuemax="${meta.durationMs}" aria-valuenow="0"><div class="playhead" id="playhead"></div>${issues.map(i=>`<button class="marker ${esc(i.severity)} ${selected===i.id?'selected':''}" style="left:${Math.min(99,Math.max(1,i.startMs/meta.durationMs*100))}%" data-issue="${esc(i.id)}" title="${esc(i.title)} at ${clock(i.startMs)}" aria-label="${esc(i.title)} at ${clock(i.startMs)}">${i.severity==='error'?'◆':i.severity==='warning'?'▲':'●'}</button>`).join('')}</div><div class="ticks"><span>00:00</span><span>${clock(meta.durationMs/4)}</span><span>${clock(meta.durationMs/2)}</span><span>${clock(meta.durationMs*3/4)}</span><span>${clock(meta.durationMs)}</span></div><div class="legend"><span><b class="error">◆</b> Error</span><span><b class="warning">▲</b> Warning</span><span>Click a marker to inspect</span></div></div></div>${checkersHtml()}</div><div class="issue-panel"><div class="issue-header"><div class="eyebrow">FINDINGS / ${issues.length}</div><h2>${issues.length ? `${issues.length} issue${issues.length===1?'':'s'} found` : 'No findings yet'}</h2><div class="filters">${['all','technical','audio','cut','repetition','goal'].map(f=>`<button class="filter ${filter===f?'active':''}" data-filter="${f}">${esc(title(f))}</button>`).join('')}</div></div><div class="issue-list" id="issueList"></div></div></div>` : ''}</section>`;
  const playerPanel = document.querySelector('.player-panel');
  if (playerPanel) playerPanel.insertAdjacentHTML('afterbegin', `<div class="player-topbar"><span>PREVIEW / <strong>${esc(report.filename)}</strong></span><span>READ-ONLY REVIEW</span></div>`);
  document.querySelector('#download').onclick = download;
  document.querySelector('#delete').onclick = remove;
  if (meta) {
    video = document.querySelector('#video');
    if (hostedVideoUrl) video.src = hostedVideoUrl;
    video.ontimeupdate = () => { const head = document.querySelector('#playhead'), timeline = document.querySelector('#timeline'); if(head) head.style.left = `${video.currentTime*1000/meta.durationMs*100}%`; if(timeline)timeline.setAttribute('aria-valuenow',Math.round(video.currentTime*1000)); };
    const timeline = document.querySelector('#timeline');
    timeline.onclick = e => {if(e.target.closest('.marker'))return;const rect=timeline.getBoundingClientRect();video.currentTime=Math.max(0,Math.min(meta.durationMs/1000,(e.clientX-rect.left)/rect.width*meta.durationMs/1000));};
    timeline.onkeydown = e => {if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();video.currentTime=Math.max(0,Math.min(meta.durationMs/1000,video.currentTime+(e.key==='ArrowRight'?5:-5)));}};
    document.querySelectorAll('.marker').forEach(b=>b.onclick=e=>{e.stopPropagation();selectIssue(b.dataset.issue)});
    document.querySelectorAll('.filter').forEach(b=>b.onclick=()=>{filter=b.dataset.filter;document.querySelectorAll('.filter').forEach(item=>item.classList.toggle('active',item.dataset.filter===filter));renderIssues();});
    renderIssues();
  }
}

function checkersHtml() {
  const debug = report.debug || {};
  const decisions = Array.isArray(debug.judgments) ? debug.judgments : [];
  return `<div class="analysis-panel"><h2>Checker status</h2><div class="pipeline-status"><span>Transcript: ${esc(debug.transcript || 'UNKNOWN')}</span><span>Gemini: ${esc(debug.gemini || 'UNKNOWN')}</span><span>Jev: ${esc(debug.jev || 'UNKNOWN')}</span></div>${report.checkers.map(c=>`<div class="stage-row"><span class="state ${esc(c.status)}">${esc(c.status)}</span><div>${esc(title(c.checkerId))}<small>${esc(c.reason||c.error||(c.latencyMs!=null?`${c.issues.length} findings · ${c.latencyMs} ms`:''))}</small></div></div>`).join('')}${decisions.length ? `<div class="judgment-log"><div class="detail-label">Development judgment log</div>${decisions.map(j=>`<div class="judgment-row"><span>${esc(title(j.checker))} · ${j.timestampMs == null ? '—' : clock(j.timestampMs)}</span><span>${esc(j.provider)} · ${esc(j.verdict || 'failed')} · ${j.confidence == null ? '—' : Math.round(j.confidence*100)+'%'} · ${esc(j.latencyMs)} ms</span></div>`).join('')}</div>` : ''}<div class="technical-note">Speech-dependent checks are not applicable when no speech is detected. Skipped checks never imply a clean result.</div></div>`;
}

function renderIssues() {
  const list = document.querySelector('#issueList'); if(!list)return;
  const items = report.issues.filter(i=>filter==='all'||i.category===filter||filter==='technical'&&['technical','audio'].includes(i.category));
  list.innerHTML = items.length ? items.map(i=>`<article class="issue-card ${selected===i.id?'selected':''}" data-id="${esc(i.id)}" tabindex="0"><div class="issue-meta"><span class="severity ${esc(i.severity)}">${i.severity==='error'?'◆':i.severity==='warning'?'▲':'●'} ${esc(i.severity.toUpperCase())} · ${esc(i.category.toUpperCase())}</span><span>${clock(i.startMs)}${i.endMs>i.startMs+1000?'–'+clock(i.endMs):''}</span></div><h3>${esc(i.title)}</h3><p>${esc(i.explanation)}</p>${selected===i.id?`<div class="issue-detail"><div class="detail-label">Evidence · ${esc(i.source)} · ${Math.round(i.confidence*100)}% confidence</div><div class="evidence">${esc(i.evidence?.transcript||JSON.stringify(i.evidence?.metrics||{},null,2))}</div>${i.suggestedAction?`<div class="detail-label">Suggested action</div><p>${esc(i.suggestedAction)}</p>`:''}<div class="feedback"><button class="ghost ${report.feedback[i.id]===true?'picked':''}" data-feedback="yes">Useful</button><button class="ghost ${report.feedback[i.id]===false?'picked':''}" data-feedback="no">Not an issue</button></div></div>`:''}</article>`).join('') : `<div class="empty">${report.issues.length?'No findings in this category.':'No findings from the checks that ran. Review skipped checks below before treating this as a clean export.'}</div>`;
  list.querySelectorAll('.issue-card').forEach(card=>{card.onclick=e=>{if(e.target.closest('[data-feedback]'))return;selectIssue(card.dataset.id)};card.onkeydown=e=>{if(e.key==='Enter')selectIssue(card.dataset.id)}});
  list.querySelectorAll('[data-feedback]').forEach(button=>button.onclick=async()=>{try{const accepted=button.dataset.feedback==='yes';if(hosted){const item=await hostedStore('get',report.id);item.report.feedback[selected]=accepted;await hostedStore('put',item)}else await api(`/api/analysis/${report.id}/feedback`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({issueId:selected,accepted})});report.feedback[selected]=accepted;renderIssues();}catch(e){alert(e.message)}});
}

function selectIssue(id) { selected=id;const issue=report.issues.find(i=>i.id===id);if(issue&&video)video.currentTime=issue.startMs/1000;document.querySelectorAll('.marker').forEach(m=>m.classList.toggle('selected',m.dataset.issue===id));renderIssues();document.querySelector('.issue-card.selected')?.scrollIntoView({block:'nearest',behavior:'smooth'}); }
function download() {const blob=new Blob([JSON.stringify(report,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;link.download=`videolint-${report.id}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
async function remove() {if(!confirm(`Delete ${report.filename} and its analysis from this browser?`))return;try{if(hosted)await hostedStore('delete',report.id);else await api(`/api/analysis/${report.id}`,{method:'DELETE'});home();}catch(e){alert(e.message)}}

document.querySelector('#homeButton').onclick = home;
newButton.onclick = home;
document.addEventListener('keydown',e=>{if(!video||['INPUT','TEXTAREA','BUTTON'].includes(document.activeElement.tagName))return;if(e.code==='Space'){e.preventDefault();video.paused?video.play():video.pause()}else if(e.key==='j'){video.currentTime=Math.max(0,video.currentTime-10)}else if(e.key==='l'){video.currentTime+=10}else if(e.key==='k'){video.paused?video.play():video.pause()}else if(e.key==='n'||e.key==='p'){const issues=report.issues;if(!issues.length)return;let at=issues.findIndex(i=>i.id===selected);at=(at+(e.key==='n'?1:-1)+issues.length)%issues.length;selectIssue(issues[at].id)}else if(e.key==='Escape'){selected=null;renderIssues()}});
(async()=>{await health();const initial = new URLSearchParams(location.search).get('analysis');if(initial&&/^[0-9a-f]{32}$/.test(initial))openReport(initial);else home();})();
