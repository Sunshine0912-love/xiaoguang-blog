/* global hexo */
'use strict';

hexo.extend.filter.register('theme_inject', injects => {
  const base = '/xiaoguang-blog/categories';
  
  injects.sidebar.raw('category-accordion', `
<div class="category-accordion" style="margin-top:15px;">
  <div style="font-size:14px;color:#999;margin-bottom:8px;text-align:center;">分类导航</div>

  <details open style="margin-bottom:6px;">
    <summary style="cursor:pointer;font-size:13px;padding:6px 10px;border-radius:8px;background:#f8f8f8;list-style:none;display:flex;align-items:center;gap:6px;">
      <span style="font-size:16px;">📰</span>
      <span>每日选题</span>
      <span style="margin-left:auto;font-size:11px;color:#999;">6</span>
      <span style="font-size:10px;margin-left:4px;transition:transform 0.2s;">▼</span>
    </summary>
    <div class="subcat-list" style="padding:4px 0 4px 20px;font-size:12px;">
      <a href="${base}/Topic/AI/AI-Infra/" style="display:flex;justify-content:space-between;padding:3px 8px;border-radius:4px;color:#555;text-decoration:none;">AI / AI Infra<span style="color:#999;">3</span></a>
      <a href="${base}/Topic/AI/Agent/" style="display:flex;justify-content:space-between;padding:3px 8px;border-radius:4px;color:#555;text-decoration:none;">AI / Agent<span style="color:#999;">2</span></a>
      <a href="${base}/Topic/AI/Industry/" style="display:flex;justify-content:space-between;padding:3px 8px;border-radius:4px;color:#555;text-decoration:none;">AI / Industry<span style="color:#999;">1</span></a>
    </div>
  </details>

  <details style="margin-bottom:6px;">
    <summary style="cursor:pointer;font-size:13px;padding:6px 10px;border-radius:8px;background:#f8f8f8;list-style:none;display:flex;align-items:center;gap:6px;">
      <span style="font-size:16px;">🔬</span>
      <span>技术深解</span>
      <span style="margin-left:auto;font-size:11px;color:#999;">4</span>
      <span style="font-size:10px;margin-left:4px;transition:transform 0.2s;">▶</span>
    </summary>
    <div class="subcat-list" style="padding:4px 0 4px 20px;font-size:12px;">
      <a href="${base}/TECH/AI/AI-Infra/" style="display:flex;justify-content:space-between;padding:3px 8px;border-radius:4px;color:#555;text-decoration:none;">AI / AI Infra<span style="color:#999;">1</span></a>
      <a href="${base}/TECH/AI/LLM/" style="display:flex;justify-content:space-between;padding:3px 8px;border-radius:4px;color:#555;text-decoration:none;">AI / LLM<span style="color:#999;">2</span></a>
      <a href="${base}/TECH/AI/%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96/" style="display:flex;justify-content:space-between;padding:3px 8px;border-radius:4px;color:#555;text-decoration:none;">AI / 推理优化<span style="color:#999;">1</span></a>
    </div>
  </details>

  <details style="margin-bottom:6px;">
    <summary style="cursor:pointer;font-size:13px;padding:6px 10px;border-radius:8px;background:#f8f8f8;list-style:none;display:flex;align-items:center;gap:6px;">
      <span style="font-size:16px;">🎓</span>
      <span>李宏毅 ML2026</span>
      <span style="margin-left:auto;font-size:11px;color:#999;">3</span>
      <span style="font-size:10px;margin-left:4px;transition:transform 0.2s;">▶</span>
    </summary>
    <div class="subcat-list" style="padding:4px 0 4px 20px;font-size:12px;">
      <a href="${base}/hylee-ML-2026-Spring/" style="display:flex;justify-content:space-between;padding:3px 8px;border-radius:4px;color:#555;text-decoration:none;">全部课程<span style="color:#999;">3</span></a>
    </div>
  </details>
</div>
  `, {}, {cache: true});
});
