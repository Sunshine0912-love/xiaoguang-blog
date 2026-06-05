#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

const ROOT = process.cwd();
const POST_DIR = path.join(ROOT, 'source', '_posts');

const ALLOWED_CODE_IDENTIFIERS = new Set([
  'absorb',
  'naive',
  'q_nope',
  'q_pe',
  'kv',
  'k_pe',
  'kv_cache',
  'pe_cache',
  'k_cache',
  'v_cache',
  'kv_lora_rank',
  'q_lora_rank',
  'qk_rope_head_dim',
  'cache.kv',
  'cache.pe',
  'kv_cache_dtype',
  'kvarn_k4v2_g128',
  'block_size',
  'k4v2',
  'kv_cache_dtype="kvarn_k4v2_g128"',
  'block_size=128',
]);

function read(file) {
  return fs.readFileSync(file, 'utf8');
}

function frontMatter(markdown) {
  if (!markdown.startsWith('---\n')) return {};
  const end = markdown.indexOf('\n---\n', 4);
  if (end === -1) return {};
  const raw = markdown.slice(4, end);
  const result = {};
  for (const line of raw.split('\n')) {
    const match = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/);
    if (match) result[match[1]] = match[2].trim().replace(/^["']|["']$/g, '');
  }
  return result;
}

function withoutCodeFences(markdown) {
  return markdown.replace(/```[\s\S]*?```/g, '');
}

function htmlDecode(text) {
  return text
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&')
    .replace(/&#x2F;/g, '/')
    .replace(/&quot;/g, '"');
}

function allAllowedCodeIdentifiers(text) {
  return text
    .split(',')
    .map((part) => part.trim())
    .every((part) => ALLOWED_CODE_IDENTIFIERS.has(part));
}

function looksLikeMathCode(text) {
  const value = htmlDecode(text).trim();
  if (!value) return false;
  if (allAllowedCodeIdentifiers(value)) return false;

  if (/[\^{}]|<<|<=|>=|=|\\/.test(value)) return true;
  if (/^[A-Za-z]$/.test(value)) return true;
  if (/^[A-Za-z]_[A-Za-z0-9]+$/.test(value)) return true;
  if (/^[A-Za-z]+_[A-Za-z0-9]+$/.test(value) && !/^[a-z]{2,}_[a-z0-9]{2,}$/.test(value)) return true;
  if (/\b(?:d|n|q|k|v|h|c|W|B|L|N|score|softmax)_[A-Za-z0-9]+\b/.test(value)) return true;
  if (/\b\d+\s+[A-Za-z_]+\b|\b[A-Za-z_]+\s+[A-Za-z_]+\b/.test(value)) return true;

  return false;
}

function sourceInlineCodeIssues(markdown) {
  const body = withoutCodeFences(markdown);
  const issues = [];
  for (const match of body.matchAll(/`([^`\n]+)`/g)) {
    if (looksLikeMathCode(match[1])) issues.push(match[1]);
  }
  return issues;
}

function htmlInlineCodeIssues(html) {
  const issues = [];
  for (const match of html.matchAll(/<code>([\s\S]*?)<\/code>/g)) {
    if (looksLikeMathCode(match[1])) issues.push(htmlDecode(match[1]));
  }
  return issues;
}

function publicHtmlPath(postFile) {
  const base = path.basename(postFile, '.md');
  const match = base.match(/^(\d{4})-(\d{2})-(\d{2})-(.+)$/);
  if (!match) return null;
  return path.join(ROOT, 'public', match[1], match[2], match[3], base, 'index.html');
}

function postFilesFromArgs() {
  const args = process.argv.slice(2);
  if (args.length) return args.map((arg) => path.resolve(ROOT, arg));
  return fs
    .readdirSync(POST_DIR)
    .filter((name) => name.endsWith('.md'))
    .map((name) => path.join(POST_DIR, name));
}

const failures = [];

for (const file of postFilesFromArgs()) {
  const rel = path.relative(ROOT, file);
  const markdown = read(file);
  const meta = frontMatter(markdown);
  const isTechLike = /TECH-|技术点讲解|算法推导|架构解析|论文解读/.test(markdown);
  const hasMath = /\$[^$]+\$|\$\$[\s\S]*?\$\$/.test(withoutCodeFences(markdown));

  if (/[\x00-\x08\x0B\x0C\x0E-\x1F]/.test(markdown)) {
    failures.push(`${rel}: contains unexpected control characters`);
  }

  if (hasMath && meta.mathjax !== 'true') {
    failures.push(`${rel}: contains math but front matter is missing mathjax: true`);
  }

  if (isTechLike) {
    const sourceIssues = [...new Set(sourceInlineCodeIssues(markdown))];
    if (sourceIssues.length) {
      failures.push(`${rel}: math-like inline code should use $...$: ${sourceIssues.join(', ')}`);
    }
  }

  if (hasMath || meta.mathjax === 'true') {
    const htmlPath = publicHtmlPath(file);
    if (!htmlPath || !fs.existsSync(htmlPath)) {
      failures.push(`${rel}: generated HTML not found; run npx hexo generate first`);
      continue;
    }

    const html = read(htmlPath);
    const katexCount = (html.match(/katex-display/g) || []).length;
    if (katexCount === 0 && hasMath) {
      failures.push(`${rel}: generated HTML has no katex-display output (KaTeX formulas not rendered)`);
    }
    if (hasMath && !/href="[^"]*\/lib\/katex\/katex\.min\.css"/.test(html)) {
      failures.push(`${rel}: generated HTML is missing local KaTeX CSS (/lib/katex/katex.min.css)`);
    }
    if (/cdnjs\.cloudflare\.com\/ajax\/libs\/KaTeX/i.test(html)) {
      failures.push(`${rel}: generated HTML still depends on external cdnjs KaTeX CSS`);
    }
    // Check for mangled formulas (<em> containing LaTeX)
    const htmlNoKatex = html.replace(/<section>[\s\S]*?<\/section>/g, '').replace(/<span class="katex[\s\S]*?<\/span>/g, '');
    const badEms = (htmlNoKatex.match(/<em>[^<]*?\\[^<]*?<\/em>/g) || []);
    if (badEms.length) {
      failures.push(`${rel}: mangled formulas found (${badEms.length} LaTeX fragments in <em> tags)`);
    }
    if (/nav-text">\$\$/.test(html)) {
      failures.push(`${rel}: generated TOC contains formula fragments`);
    }
    const htmlIssues = [...new Set(htmlInlineCodeIssues(html))];
    if (htmlIssues.length) {
      failures.push(`${rel}: generated HTML still has math-like <code>: ${htmlIssues.join(', ')}`);
    }
  }
}

if (failures.length) {
  console.error('TECH article validation failed:');
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log('TECH article validation passed.');
