/* global hexo */
'use strict';

hexo.extend.filter.register('theme_inject', injects => {
  injects.sidebar.raw('category-links', `
    <div class="site-category-links" style="text-align:center;margin-top:15px;padding:5px 0;">
      <div style="font-size:14px;color:#999;margin-bottom:8px;">分类导航</div>
      <a href="/xiaoguang-blog/categories/Topic/" style="display:inline-block;margin:0 6px;padding:3px 10px;border:1px solid #e0e0e0;border-radius:12px;font-size:13px;color:#555;text-decoration:none;">
        <i class="fa fa-newspaper"></i> 每日选题
      </a>
      <a href="/xiaoguang-blog/categories/TECH/" style="display:inline-block;margin:0 6px;padding:3px 10px;border:1px solid #e0e0e0;border-radius:12px;font-size:13px;color:#555;text-decoration:none;">
        <i class="fa fa-microchip"></i> 技术深解
      </a>
      <a href="/xiaoguang-blog/categories/hylee-ML-2026-Spring/" style="display:inline-block;margin:0 6px;padding:3px 10px;border:1px solid #e0e0e0;border-radius:12px;font-size:13px;color:#555;text-decoration:none;">
        <i class="fa fa-graduation-cap"></i> 李宏毅 ML2026
      </a>
    </div>
  `, {}, {cache: true});
});
