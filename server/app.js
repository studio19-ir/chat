import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import OpenAI from 'openai';
import multer from 'multer';
import pdfParse from 'pdf-parse';
import { JSDOM } from 'jsdom';
import { fileURLToPath } from 'url';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const app = express();
app.use(cors());
app.use(express.json({ limit: '4mb' }));

const PORT = process.env.PORT || 3000;
const MIN_TOP_SIM = parseFloat(process.env.MIN_TOP_SIM || '0.78');
const MIN_AVG_TOP3 = parseFloat(process.env.MIN_AVG_TOP3 || '0.72');
const ADMIN_TOKEN = process.env.ADMIN_TOKEN || '';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const ROOT = path.join(__dirname, '..');
const DATA_PATH = path.join(ROOT, 'data', 'index.json');
const FILES_DIR = path.join(ROOT, 'ingest', 'files');
const URLS_FILE = path.join(ROOT, 'ingest', 'urls.txt');

// --------- Utilities ---------
function ensureDirs(){
  if (!fs.existsSync(path.join(ROOT, 'data'))) fs.mkdirSync(path.join(ROOT, 'data'));
  if (!fs.existsSync(path.join(ROOT, 'ingest'))) fs.mkdirSync(path.join(ROOT, 'ingest'));
  if (!fs.existsSync(FILES_DIR)) fs.mkdirSync(FILES_DIR);
}
ensureDirs();

let INDEX = { items: [] };
if (fs.existsSync(DATA_PATH)) {
  try { INDEX = JSON.parse(fs.readFileSync(DATA_PATH, 'utf-8')); }
  catch { INDEX = { items: [] }; }
}
function saveIndex(){
  fs.writeFileSync(DATA_PATH, JSON.stringify(INDEX, null, 2), 'utf-8');
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
}
async function embed(texts){
  const inputs = Array.isArray(texts) ? texts : [texts];
  const resp = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: inputs
  });
  return resp.data.map(d => d.embedding);
}
function chunkText(text, size = 800, overlap = 200) {
  const chunks = [];
  let i = 0;
  while (i < text.length) {
    const end = Math.min(text.length, i + size);
    const chunk = text.slice(i, end);
    chunks.push(chunk);
    i += (size - overlap);
  }
  return chunks;
}

async function readTextFromBuffer(buf, name){
  const ext = path.extname(name).toLowerCase();
  if (ext === '.pdf') {
    const parsed = await pdfParse(buf);
    return parsed.text.replace(/\s+/g, ' ').trim();
  }
  // assume text-like
  return buf.toString('utf-8').replace(/\s+/g, ' ').trim();
}

async function fetchHtmlText(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Fetch failed ${res.status} for ${url}`);
  const html = await res.text();
  const dom = new JSDOM(html);
  const doc = dom.window.document;
  doc.querySelectorAll('script, style, noscript').forEach(el => el.remove());
  const text = doc.body?.textContent?.replace(/\s+/g, ' ').trim() || '';
  return text;
}
async function fetchPdfText(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Fetch failed ${res.status} for ${url}`);
  const buf = Buffer.from(await res.arrayBuffer());
  const parsed = await pdfParse(buf);
  return parsed.text.replace(/\s+/g, ' ').trim();
}

async function embedAndAppend(chunks, meta){
  if (!chunks.length) return 0;
  const embs = await embed(chunks);
  for (let i = 0; i < chunks.length; i++) {
    INDEX.items.push({
      id: `${meta.idPrefix}::${i}`,
      source: meta.source,
      type: meta.type,
      text: chunks[i],
      embedding: embs[i]
    });
  }
  saveIndex();
  return chunks.length;
}

async function rebuildFromDisk(){
  const items = [];
  // from urls.txt
  if (fs.existsSync(URLS_FILE)) {
    const lines = fs.readFileSync(URLS_FILE, 'utf-8').split(/\r?\n/).map(s => s.trim()).filter(Boolean).filter(x => !x.startsWith('#'));
    for (const url of lines) {
      try {
        const isPdf = url.toLowerCase().endsWith('.pdf');
        const text = isPdf ? await fetchPdfText(url) : await fetchHtmlText(url);
        const chunks = chunkText(text);
        const embs = await embed(chunks);
        for (let i = 0; i < chunks.length; i++) {
          items.push({ id: `url:${url}::${i}`, source: url, type: 'url', text: chunks[i], embedding: embs[i] });
        }
      } catch (e) {
        console.error('URL rebuild failed', url, e.message);
      }
    }
  }
  // from files dir
  if (fs.existsSync(FILES_DIR)) {
    const files = fs.readdirSync(FILES_DIR);
    for (const name of files) {
      const fp = path.join(FILES_DIR, name);
      if (!fs.statSync(fp).isFile()) continue;
      try {
        const buf = fs.readFileSync(fp);
        const text = await readTextFromBuffer(buf, name);
        const chunks = chunkText(text);
        const embs = await embed(chunks);
        for (let i = 0; i < chunks.length; i++) {
          items.push({ id: `file:${name}::${i}`, source: name, type: 'file', text: chunks[i], embedding: embs[i] });
        }
      } catch (e) {
        console.error('File rebuild failed', name, e.message);
      }
    }
  }
  INDEX = { items };
  saveIndex();
  return items.length;
}

// --------- Auth ---------
function requireAdmin(req, res, next){
  if (!ADMIN_TOKEN) return res.status(500).json({ error: 'ADMIN_TOKEN not set on server' });
  const auth = req.headers['authorization'] || '';
  const token = auth.startsWith('Bearer ') ? auth.slice(7) : '';
  if (token !== ADMIN_TOKEN) return res.status(401).json({ error: 'unauthorized' });
  next();
}

// --------- Chat core ---------
async function embedQuery(q) {
  const resp = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: q
  });
  return resp.data[0].embedding;
}
function topKByCosine(qEmb, k = 5) {
  const scored = INDEX.items.map(it => ({
    ...it,
    score: cosine(qEmb, it.embedding)
  }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, Math.min(k, scored.length));
}
function buildSystemPrompt() {
  return [
    "شما یک دستیار پرسش‌و‌پاسخ هستید که **فقط** از روی «کانتکست» ارائه‌شده پاسخ می‌دهد.",
    "قواعد سخت‌گیرانه:",
    "1) اگر پاسخ در کانتکست نبود یا اطمینان ندارید، صریح بگویید: «برای این موضوع آموزش ندیده‌ام و نمی‌توانم پاسخ بدهم.»",
    "2) از دانسته‌های عمومی یا حدس استفاده نکنید. هرگز اطلاعات تازه تولید نکنید.",
    "3) اگر کانتکست کافی بود، پاسخ را کوتاه و دقیق بدهید و در صورت امکان در پایان منابع را با شماره‌گذاری [[1]]، [[2]] ذکر کنید."
  ].join("\n");
}
function formatContext(chunks) {
  return chunks.map((c, i) => `[[${i+1}]] (${c.type === 'url' ? c.source : 'file: ' + c.source})\n${c.text}`).join("\n\n---\n\n");
}
function refusalMessage() {
  return { answer: "برای این موضوع آموزش ندیده‌ام و نمی‌توانم پاسخ بدهم.", sources: [] };
}

// --------- Routes ---------
app.get('/health', (req, res) => {
  res.json({ ok: true, items: INDEX.items?.length || 0, minTopSim: MIN_TOP_SIM, minAvgTop3: MIN_AVG_TOP3 });
});

// Serve static (widget + admin)
app.use('/', express.static(path.join(ROOT, 'public')));

app.post('/api/chat', async (req, res) => {
  try {
    const { message } = req.body || {};
    if (!message || typeof message !== 'string') {
      return res.status(400).json({ error: 'message is required' });
    }
    if (!INDEX.items || INDEX.items.length === 0) {
      return res.status(400).json({ error: 'No index found. Use admin panel to add resources.' });
    }
    const qEmb = await embedQuery(message);
    const top = topKByCosine(qEmb, 5);
    const topSim = top[0]?.score ?? 0;
    const avgTop3 = top.slice(0,3).reduce((s, x) => s + x.score, 0) / Math.max(1, Math.min(3, top.length));
    if (topSim < MIN_TOP_SIM || avgTop3 < MIN_AVG_TOP3) {
      return res.json(refusalMessage());
    }
    const ctx = formatContext(top);
    const sys = buildSystemPrompt();
    const user = ["پرسش کاربر:", message, "", "کانتکست:", ctx].join("\n");
    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: sys },
        { role: "user", content: user }
      ],
      temperature: 0.2
    });
    const answer = completion.choices[0].message.content || "";
    const sources = top.map((t, i) => ({ id: i+1, source: t.source, score: Number(t.score.toFixed(3)) }));
    res.json({ answer, sources });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message || 'server error' });
  }
});

// --------- Admin APIs ---------
const upload = multer({ storage: multer.memoryStorage() });

app.get('/api/admin/items', requireAdmin, (req, res) => {
  const groupsMap = new Map();
  for (const it of (INDEX.items || [])) {
    const key = it.source;
    const g = groupsMap.get(key) || { source: key, type: it.type, count: 0 };
    g.count++;
    groupsMap.set(key, g);
  }
  const groups = Array.from(groupsMap.values()).sort((a,b)=>a.source.localeCompare(b.source));
  res.json({ total: INDEX.items?.length || 0, groups });
});

app.delete('/api/admin/source', requireAdmin, async (req, res) => {
  const source = req.query.source || '';
  if (!source) return res.status(400).json({ error: 'source query required' });
  const before = INDEX.items.length;
  INDEX.items = INDEX.items.filter(it => it.source !== source);
  saveIndex();
  res.json({ removed: before - INDEX.items.length });
});

app.post('/api/admin/upload', requireAdmin, upload.array('files', 10), async (req, res) => {
  try {
    if (!req.files || !req.files.length) return res.status(400).json({ error: 'no files' });
    let total = 0;
    for (const f of req.files) {
      // persist original file into ingest/files for backup
      const safeName = Date.now() + '-' + f.originalname.replace(/[^\w\.\-\u0600-\u06FF]/g, '_');
      fs.writeFileSync(path.join(FILES_DIR, safeName), f.buffer);
      // parse and embed
      const text = await readTextFromBuffer(f.buffer, f.originalname);
      const chunks = chunkText(text);
      const added = await embedAndAppend(chunks, { idPrefix: `file:${safeName}`, source: safeName, type: 'file' });
      total += added;
    }
    res.json({ ok: true, indexed: total });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message });
  }
});

app.post('/api/admin/add-url', requireAdmin, async (req, res) => {
  try {
    const url = (req.body?.url || '').trim();
    if (!url) return res.status(400).json({ error: 'url required' });
    let text = '';
    if (url.toLowerCase().endsWith('.pdf')) text = await fetchPdfText(url);
    else text = await fetchHtmlText(url);
    const chunks = chunkText(text);
    const added = await embedAndAppend(chunks, { idPrefix: `url:${url}`, source: url, type: 'url' });
    // store to urls.txt for future rebuilds
    try {
      const line = url + "\n";
      const exists = fs.existsSync(URLS_FILE) ? fs.readFileSync(URLS_FILE, 'utf-8').includes(url) : false;
      if (!exists) fs.appendFileSync(URLS_FILE, line);
    } catch {}
    res.json({ ok: true, indexed: added });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message });
  }
});

app.post('/api/admin/rebuild', requireAdmin, async (req, res) => {
  try {
    const n = await rebuildFromDisk();
    res.json({ ok: true, items: n });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Polyfill fetch for Node < 18
if (typeof fetch === 'undefined') {
  const nodeFetch = await import('node-fetch');
  globalThis.fetch = nodeFetch.default;
}

app.listen(PORT, () => {
  console.log(`> Server listening on http://localhost:${PORT}`);
});
