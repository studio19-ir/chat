import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import pdfParse from 'pdf-parse';
import { JSDOM } from 'jsdom';
import OpenAI from 'openai';

dotenv.config();
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const DATA_DIR = path.join(__dirname, '..', 'data');
const FILES_DIR = path.join(__dirname, 'files');
const URLS_FILE = path.join(__dirname, 'urls.txt');
const INDEX_PATH = path.join(DATA_DIR, 'index.json');

async function readUrls() {
  if (!fs.existsSync(URLS_FILE)) return [];
  const raw = fs.readFileSync(URLS_FILE, 'utf-8');
  return raw.split(/\r?\n/).map(l => l.trim()).filter(l => l && !l.startsWith('#'));
}

async function fetchHtmlText(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Fetch failed ${res.status} for ${url}`);
  const html = await res.text();
  const dom = new JSDOM(html);
  const doc = dom.window.document;
  // remove scripts/styles
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

async function readLocalFileText(fp) {
  const ext = path.extname(fp).toLowerCase();
  if (ext === '.pdf') {
    const buf = fs.readFileSync(fp);
    const parsed = await pdfParse(buf);
    return parsed.text.replace(/\s+/g, ' ').trim();
  }
  // fall back to plain text for others
  return fs.readFileSync(fp, 'utf-8').toString().replace(/\s+/g, ' ').trim();
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

async function embedBatch(texts) {
  const resp = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: texts
  });
  return resp.data.map(d => d.embedding);
}

async function main() {
  console.log('> Ingest started...');
  const items = [];

  // 1) URLs
  const urls = await readUrls();
  for (const url of urls) {
    try {
      const isPdf = url.toLowerCase().endsWith('.pdf');
      const text = isPdf ? await fetchPdfText(url) : await fetchHtmlText(url);
      const chunks = chunkText(text);
      chunks.forEach((c, idx) => items.push({ id: `url:${url}::${idx}`, source: url, type: 'url', text: c }));
      console.log(`  - Indexed ${chunks.length} chunks from ${url}`);
    } catch (e) {
      console.error('  ! URL failed', url, e.message);
    }
  }

  // 2) Local files
  const files = fs.readdirSync(FILES_DIR).map(f => path.join(FILES_DIR, f));
  for (const fp of files) {
    const stat = fs.statSync(fp);
    if (!stat.isFile()) continue;
    try {
      const text = await readLocalFileText(fp);
      const chunks = chunkText(text);
      const rel = path.basename(fp);
      chunks.forEach((c, idx) => items.push({ id: `file:${rel}::${idx}`, source: rel, type: 'file', text: c }));
      console.log(`  - Indexed ${chunks.length} chunks from file ${rel}`);
    } catch (e) {
      console.error('  ! File failed', fp, e.message);
    }
  }

  if (items.length === 0) {
    console.log('No items to index. Add URLs to ingest/urls.txt or files to ingest/files/');
    return;
  }

  // Embed in small batches to avoid token limits
  const BATCH = 64;
  for (let i = 0; i < items.length; i += BATCH) {
    const slice = items.slice(i, i + BATCH);
    const embeddings = await embedBatch(slice.map(s => s.text));
    for (let j = 0; j < slice.length; j++) {
      slice[j].embedding = embeddings[j];
    }
    console.log(`  - Embedded ${Math.min(BATCH, items.length - i)} chunks (${i + Math.min(BATCH, items.length - i)}/${items.length})`);
  }

  const index = {
    createdAt: new Date().toISOString(),
    model: 'text-embedding-3-small',
    items: items
  };
  fs.writeFileSync(INDEX_PATH, JSON.stringify(index, null, 2), 'utf-8');
  console.log('> Index written to', INDEX_PATH);
}

// Polyfill fetch for Node < 18
if (typeof fetch === 'undefined') {
  globalThis.fetch = (await import('node-fetch')).default;
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
