(function() {
  var el = document.getElementById('catTree');
  if (!el) return;
  var icons = {'AI':'🤖','hylee ML 2026 Spring':'🎓'};

  var base = document.querySelector('meta[property="og:url"]');
  var root = base ? new URL(base.content).pathname.replace(/\/categories\/.*$/, '/') : '/xiaoguang-blog/';

  fetch(root + 'categories-data.json')
    .then(function(r) { return r.json(); })
    .then(function(posts) {
      var tree = {};

      posts.forEach(function(post) {
        var rootCat = post.c[0];
        var sub = post.c.length > 1 ? post.c[1] : null;
        if (!rootCat) return;
        if (!tree[rootCat]) tree[rootCat] = {};
        var key = sub || '_posts';
        if (!tree[rootCat][key]) tree[rootCat][key] = [];
        tree[rootCat][key].push(post);
      });

      var html = '';
      Object.keys(tree).sort().forEach(function(rootCat) {
        var icon = icons[rootCat] || '📂';
        var subCats = tree[rootCat];
        var totalPosts = 0;
        Object.keys(subCats).forEach(function(k) { totalPosts += subCats[k].length; });

        html += '<div class="cat-group">';
        html += '<div class="cat-header" onclick="this.parentElement.classList.toggle(\'open\')">';
        html += '<span class="name"><span class="icon">' + icon + '</span>' + rootCat + '</span>';
        html += '<span style="display:flex;align-items:center;gap:12px;">';
        html += '<span class="count">' + totalPosts + ' 篇</span>';
        html += '<span class="arrow">▶</span></span></div>';
        html += '<div class="cat-sub">';

        Object.keys(subCats).sort().forEach(function(subCat) {
          var p = subCats[subCat];
          html += '<div class="sub-item sub-group">';
          if (subCat !== '_posts') {
            html += '<div class="sub-header" onclick="event.stopPropagation();this.parentElement.classList.toggle(\'open\')">';
            html += '<span>' + subCat + '</span>';
            html += '<span style="display:flex;align-items:center;gap:12px;">';
            html += '<span class="count">' + p.length + ' 篇</span>';
            html += '<span class="arrow">▶</span></span></div>';
          }
          html += '<div class="sub-posts">';
          p.forEach(function(post) {
            html += '<a href="' + post.u + '"><span class="post-date">' + post.d + '</span>' + post.t + '</a>';
          });
          html += '</div></div>';
        });

        html += '</div></div>';
      });

      el.innerHTML = html;
    })
    .catch(function() {
      el.innerHTML = '<p style="text-align:center;opacity:0.4;padding:40px 0;">分类数据加载失败，请刷新重试。</p>';
    });
})();
