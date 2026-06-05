/* global hexo */
'use strict';

hexo.extend.filter.register('theme_inject', injects => {
  const base = '/xiaoguang-blog/categories';
  
  injects.sidebar.raw('category-accordion', `
<div class="category-accordion" style="margin-top:15px;">
  <div style="font-size:13px;color:#777;margin-bottom:10px;text-align:center;letter-spacing:1px;">分类导航</div>

  <div class="cat-group" style="margin-bottom:4px;">
    <div style="display:flex;align-items:center;gap:6px;padding:6px 10px;border-radius:6px;background:rgba(255,255,255,0.08);font-size:13px;color:#ccc;">
      <span style="font-size:15px;">📰</span>
      <a href="${base}/Topic/" style="color:#ccc;text-decoration:none;flex:1;">每日选题</a>
      <span style="font-size:11px;color:#666;">6</span>
      <span class="cat-arrow" data-cat="cat-topic" style="cursor:pointer;font-size:10px;color:#777;user-select:none;padding:0 2px;">▼</span>
    </div>
    <div class="cat-sub" id="cat-topic" style="padding:2px 0 2px 20px;font-size:12px;">
      <a href="${base}/Topic/AI/AI-Infra/" style="display:flex;justify-content:space-between;padding:4px 8px;border-radius:4px;color:#999;text-decoration:none;transition:background 0.15s;">AI / AI Infra<span style="color:#666;">3</span></a>
      <a href="${base}/Topic/AI/Agent/" style="display:flex;justify-content:space-between;padding:4px 8px;border-radius:4px;color:#999;text-decoration:none;transition:background 0.15s;">AI / Agent<span style="color:#666;">2</span></a>
      <a href="${base}/Topic/AI/Industry/" style="display:flex;justify-content:space-between;padding:4px 8px;border-radius:4px;color:#999;text-decoration:none;transition:background 0.15s;">AI / Industry<span style="color:#666;">1</span></a>
    </div>
  </div>

  <div class="cat-group" style="margin-bottom:4px;">
    <div style="display:flex;align-items:center;gap:6px;padding:6px 10px;border-radius:6px;background:rgba(255,255,255,0.05);font-size:13px;color:#ccc;">
      <span style="font-size:15px;">🔬</span>
      <a href="${base}/TECH/" style="color:#ccc;text-decoration:none;flex:1;">技术深解</a>
      <span style="font-size:11px;color:#666;">4</span>
      <span class="cat-arrow" data-cat="cat-tech" style="cursor:pointer;font-size:10px;color:#777;user-select:none;padding:0 2px;">▶</span>
    </div>
    <div class="cat-sub" id="cat-tech" style="display:none;padding:2px 0 2px 20px;font-size:12px;">
      <a href="${base}/TECH/AI/AI-Infra/" style="display:flex;justify-content:space-between;padding:4px 8px;border-radius:4px;color:#999;text-decoration:none;transition:background 0.15s;">AI / AI Infra<span style="color:#666;">1</span></a>
      <a href="${base}/TECH/AI/LLM/" style="display:flex;justify-content:space-between;padding:4px 8px;border-radius:4px;color:#999;text-decoration:none;transition:background 0.15s;">AI / LLM<span style="color:#666;">2</span></a>
      <a href="${base}/TECH/AI/%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96/" style="display:flex;justify-content:space-between;padding:4px 8px;border-radius:4px;color:#999;text-decoration:none;transition:background 0.15s;">AI / 推理优化<span style="color:#666;">1</span></a>
    </div>
  </div>

  <div class="cat-group" style="margin-bottom:4px;">
    <div style="display:flex;align-items:center;gap:6px;padding:6px 10px;border-radius:6px;background:rgba(255,255,255,0.05);font-size:13px;color:#ccc;">
      <span style="font-size:15px;">🎓</span>
      <a href="${base}/hylee-ML-2026-Spring/" style="color:#ccc;text-decoration:none;flex:1;">李宏毅 ML2026</a>
      <span style="font-size:11px;color:#666;">3</span>
      <span class="cat-arrow" data-cat="cat-hylee" style="cursor:pointer;font-size:10px;color:#777;user-select:none;padding:0 2px;">▶</span>
    </div>
    <div class="cat-sub" id="cat-hylee" style="display:none;padding:2px 0 2px 20px;font-size:12px;">
      <a href="${base}/hylee-ML-2026-Spring/" style="display:flex;justify-content:space-between;padding:4px 8px;border-radius:4px;color:#999;text-decoration:none;transition:background 0.15s;">全部课程<span style="color:#666;">3</span></a>
    </div>
  </div>
</div>

<style>
.category-accordion .cat-sub a:hover { background: rgba(255,255,255,0.05); color: #ccc !important; }
.category-accordion .cat-group a:hover { color: #eee !important; }
</style>

<script>
(function(){
  document.querySelectorAll('.cat-arrow').forEach(function(arr){
    arr.addEventListener('click', function(e){
      e.stopPropagation();
      e.preventDefault();
      var id = this.dataset.cat;
      var sub = document.getElementById(id);
      if (!sub) return;
      var isOpen = sub.style.display !== 'none';
      sub.style.display = isOpen ? 'none' : '';
      this.textContent = isOpen ? '▶' : '▼';
      // Update background of parent
      var parent = this.parentElement;
      parent.style.background = isOpen ? 'rgba(255,255,255,0.05)' : 'rgba(255,255,255,0.08)';
    });
  });
})();
</script>
  `, {}, {cache: true});
});
