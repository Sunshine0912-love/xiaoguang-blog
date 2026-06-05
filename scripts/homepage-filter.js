/* global hexo */
'use strict';

hexo.extend.filter.register('theme_inject', injects => {
  injects.bodyEnd.raw('homepage-category-filter', `
<div id="cat-filter-bar" style="display:none;justify-content:center;flex-wrap:wrap;gap:8px;margin:0 0 20px 0;padding:12px 0;border-bottom:1px solid #eee;">
  <a class="cat-filter-tab active" data-cat="all" style="display:inline-block;padding:5px 16px;border-radius:16px;font-size:14px;color:#333;text-decoration:none;background:#f0f0f0;cursor:pointer;">全部 <span style="font-size:11px;color:#999;">14</span></a>
  <a class="cat-filter-tab" data-cat="Topic" style="display:inline-block;padding:5px 16px;border-radius:16px;font-size:14px;color:#555;text-decoration:none;background:#fff;border:1px solid #e0e0e0;cursor:pointer;">📰 每日选题 <span style="font-size:11px;color:#999;">6</span></a>
  <a class="cat-filter-tab" data-cat="TECH" style="display:inline-block;padding:5px 16px;border-radius:16px;font-size:14px;color:#555;text-decoration:none;background:#fff;border:1px solid #e0e0e0;cursor:pointer;">🔬 技术深解 <span style="font-size:11px;color:#999;">4</span></a>
  <a class="cat-filter-tab" data-cat="hylee ML 2026 Spring" style="display:inline-block;padding:5px 16px;border-radius:16px;font-size:14px;color:#555;text-decoration:none;background:#fff;border:1px solid #e0e0e0;cursor:pointer;">🎓 李宏毅 ML2026 <span style="font-size:11px;color:#999;">3</span></a>
</div>

<style>
.cat-filter-tab.active { background:#333 !important; color:#fff !important; border-color:#333 !important; }
.cat-filter-tab.active span { color:#ccc !important; }
.cat-filter-tab:hover { border-color:#999 !important; }
@media (max-width:480px) {
  #cat-filter-bar { gap:4px; }
  .cat-filter-tab { padding:4px 10px !important; font-size:12px !important; }
}
</style>

<script>
(function(){
  // Only on homepage index (not inside a single post)
  if (!document.querySelector('.posts-expand')) return;
  var bar = document.getElementById('cat-filter-bar');
  if (!bar) return;
  bar.style.display = 'flex';

  // Detect if current page is paginated
  var isArchive = !!document.querySelector('.archive');
  if (isArchive) {
    // Place filter bar before the archive post list
    var archivePosts = document.querySelector('.posts-collapse') || document.querySelector('.archive .posts-expand');
    if (archivePosts) {
      archivePosts.parentNode.insertBefore(bar, archivePosts);
    }
  } else {
    // Place before post list on homepage
    var posts = document.querySelector('.posts-expand');
    if (posts) posts.parentNode.insertBefore(bar, posts);
  }

  // Filter click handler
  bar.querySelectorAll('.cat-filter-tab').forEach(function(tab){
    tab.addEventListener('click', function(){
      var cat = this.dataset.cat;
      bar.querySelectorAll('.cat-filter-tab').forEach(function(t){ t.classList.remove('active'); });
      this.classList.add('active');

      var blocks = document.querySelectorAll('.post-block, .post');
      var visible = 0;
      blocks.forEach(function(post){
        if (cat === 'all') { post.style.display = ''; visible++; return; }
        var links = post.querySelectorAll('.post-category a, .category-links a');
        var match = false;
        links.forEach(function(link){
          // Check if category URL contains our target
          if (link.href && link.href.indexOf('/categories/' + encodeURIComponent(cat) + '/') !== -1) match = true;
        });
        post.style.display = match ? '' : 'none';
        if (match) visible++;
      });

      // If no results, show empty state
      var empty = document.getElementById('cat-filter-empty');
      if (!empty && visible === 0 && cat !== 'all') {
        empty = document.createElement('div');
        empty.id = 'cat-filter-empty';
        empty.style.cssText = 'text-align:center;padding:40px 0;color:#999;';
        empty.textContent = '该分类暂无文章';
        var container = document.querySelector('.posts-expand') || document.querySelector('.posts-collapse');
        if (container) container.parentNode.insertBefore(empty, container.nextSibling);
      }
      if (visible > 0) {
        var e = document.getElementById('cat-filter-empty');
        if (e) e.remove();
      }
    });
  });
})();
</script>
  `, {}, {cache: true});
});
