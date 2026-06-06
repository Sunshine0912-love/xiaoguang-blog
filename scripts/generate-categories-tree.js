// Generate categories-data.json for the categories page
const fs = require('fs');
const path = require('path');

hexo.extend.generator.register('categories_tree', function(locals) {
  const posts = locals.posts.sort('-date').toArray();
  const data = posts.map(post => ({
    t: post.title,
    u: hexo.config.root + post.path.replace(/\.md$/, '.html').replace(/index\.html$/, ''),
    d: post.date.format('YYYY-MM-DD'),
    c: post.categories.toArray().map(c => c.name)
  }));

  return {
    path: 'categories-data.json',
    data: JSON.stringify(data)
  };
});
