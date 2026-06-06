// Nunjucks helper: sidebar_cat_tree()
// Renders expandable category tree for sidebar without JS dependency
const { url_for } = require('hexo-util');

hexo.extend.helper.register('sidebar_cat_tree', function() {
  const { site } = this;
  if (!site || !site.posts) return '<p>Loading…</p>';
  
  const posts = site.posts.sort('-date').filter(p => p.published !== false).toArray();
  
  const tree = {};
  for (const post of posts) {
    const cats = post.categories.toArray().map(c => c.name);
    const root = cats[0];
    if (!root) continue;
    const sub = cats.length > 1 ? cats[1] : null;
    if (!tree[root]) tree[root] = {};
    const key = sub || '_direct';
    if (!tree[root][key]) tree[root][key] = [];
    tree[root][key].push({ 
      title: post.title, 
      url: url_for.call(this, post.path),
      date: post.date.format('YYYY-MM-DD') 
    });
  }

  const icons = { 'AI': '🤖', 'hello-xiaoguang': '👋', 'hylee ML 2026 Spring': '🎓' };
  const sortedRoots = Object.keys(tree).sort((a, b) => {
    if (a === 'hello-xiaoguang') return 1;
    if (b === 'hello-xiaoguang') return -1;
    if (a === 'AI') return -1;
    if (b === 'AI') return 1;
    return a.localeCompare(b);
  });
  let html = '';

  for (const rootCat of sortedRoots) {
    const icon = icons[rootCat] || '📂';
    const subCats = tree[rootCat];
    let totalPosts = 0;
    for (const k of Object.keys(subCats)) totalPosts += subCats[k].length;

    html += '<div class="scat-group">';
    html += '<div class="scat-root" onclick="var g=this.parentElement;g.classList.toggle(\'open\')">';
    html += '<span>' + icon + ' ' + rootCat + ' <small>(' + totalPosts + ')</small></span>';
    html += '<span class="scat-arrow">▶</span></div>';
    html += '<div class="scat-children">';

    const sortedSubs = Object.keys(subCats).sort();
    for (const subCat of sortedSubs) {
      const p = subCats[subCat];
      if (subCat !== '_direct') {
        html += '<div class="scat-sub">';
        html += '<div class="scat-sub-h" onclick="event.stopPropagation();var s=this.parentElement;s.classList.toggle(\'open\')">';
        html += '<span>' + subCat + ' (' + p.length + ')</span>';
        html += '<span class="scat-arrow">▶</span></div>';
        html += '<div class="scat-links">';
        for (const post of p) {
          html += '<a href="' + post.url + '" title="' + post.title.replace(/"/g, '&quot;') + '">' + post.title + '</a>';
        }
        html += '</div></div>';
      } else {
        html += '<div class="scat-links scat-links--direct">';
        for (const post of p) {
          html += '<a href="' + post.url + '" title="' + post.title.replace(/"/g, '&quot;') + '">' + post.title + '</a>';
        }
        html += '</div>';
      }
    }

    html += '</div></div>';
  }

  return html;
});
