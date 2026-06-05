/* global hexo */
'use strict';

hexo.extend.filter.register('theme_inject', injects => {
  const base = '/xiaoguang-blog/categories';
  
  injects.sidebar.raw('category-accordion', `
<div class="category-accordion" style="margin-top:15px;">
  <div style="font-size:13px;color:#777;margin-bottom:10px;text-align:center;letter-spacing:1px;">分类导航</div>

  <details open style="margin-bottom:4px;">
    <summary style="cursor:pointer;font-size:13px;padding:6px 10px;border-radius:6px;background:rgba(255,255,255,0.05);list-style:none;display:flex;align-items:center;gap:6px;color:#ccc;transition:background 0.15s;">
      <span style="font-size:15px;">📰</span>
      <span>每日选题</span>
      <span style="margin-left:auto;font-size:11px;color:#666;">6</span>
      <span style="font-size:10px;margin-left:2px;color:#777;">▼</span>
    </summary>
    <div class="subcat-list" style="padding:2px 0 2px 20px;font-size:12px;">
      <a href="${base}/Topic/AI/AI-Infra/" style="display:flex;justify-content:space-between;padding:4px 8px;border-radius:4px;color:#999;text-decoration:none;transition:background 0.15s;">AI / AI Infra<span style="color:#666;">3</span></a>
      <a href="${base}/Topic/AI/Agent/" style="display:flex;justify-content:space-between;padding:4px 8px;border-radius:4px;color:#999;text-decoration:none;transition:background 0.15s;">AI / Agent<span style="color:#666;">2</span></a>
      <a href="${base}/Topic/AI/Industry/" style="display:flex;justify-content:space-between;padding:4px 8px;border-radius:4px;color:#999;text-decoration:none;transition:background 0.15s;">AI / Industry<span style="color:#666;">1</span></a>
    </div>
  </details>

  <details style="margin-bottom:4px;">
    <summary style="cursor:pointer;font-size:13px;padding:6px 10px;border-radius:6px;background:rgba(255,255,255,0.05);list-style:none;display:flex;align-items:center;gap:6px;color:#ccc;transition:background 0.15s;">
      <span style="font-size:15px;">🔬</span>
      <span>技术深解</span>
      <span style="margin-left:auto;font-size:11px;color:#666;">4</span>
      <span style="font-size:10px;margin-left:2px;color:#777;">▶</span>
    </summary>
    <div class="subcat-list" style="padding:2px 0 2px 20px;font-size:12px;">
      <a href="${base}/TECH/AI/AI-Infra/" style="display:flex;justify-content:space-between;padding:4px 8px;border-radius:4px;color:#999;text-decoration:none;transition:background 0.15s;">AI / AI Infra<span style="color:#666;">1</span></a>
      <a href="${base}/TECH/AI/LLM/" style="display:flex;justify-content:space-between;padding:4px 8px;border-radius:4px;color:#999;text-decoration:none;transition:background 0.15s;">AI / LLM<span style="color:#666;">2</span></a>
      <a href="${base}/TECH/AI/%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96/" style="display:flex;justify-content:space-between;padding:4px 8px;border-radius:4px;color:#999;text-decoration:none;transition:background 0.15s;">AI / 推理优化<span style="color:#666;">1</span></a>
    </div>
  </details>

  <details style="margin-bottom:4px;">
    <summary style="cursor:pointer;font-size:13px;padding:6px 10px;border-radius:6px;background:rgba(255,255,255,0.05);list-style:none;display:flex;align-items:center;gap:6px;color:#ccc;transition:background 0.15s;">
      <span style="font-size:15px;">🎓</span>
      <span>李宏毅 ML2026</span>
      <span style="margin-left:auto;font-size:11px;color:#666;">3</span>
      <span style="font-size:10px;margin-left:2px;color:#777;">▶</span>
    </summary>
    <div class="subcat-list" style="padding:2px 0 2px 20px;font-size:12px;">
      <a href="${base}/hylee-ML-2026-Spring/" style="display:flex;justify-content:space-between;padding:4px 8px;border-radius:4px;color:#999;text-decoration:none;transition:background 0.15s;">全部课程<span style="color:#666;">3</span></a>
    </div>
  </details>
</div>

<style>
.category-accordion summary:hover { background: rgba(255,255,255,0.08) !important; }
.category-accordion .subcat-list a:hover { background: rgba(255,255,255,0.05); color: #ccc !important; }
.category-accordion details[open] > summary { background: rgba(255,255,255,0.08) !important; }
</style>
  `, {}, {cache: true});
});
