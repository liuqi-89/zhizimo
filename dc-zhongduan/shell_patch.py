from __future__ import annotations

import os
from pathlib import Path

MARK = "DSH-NOTRANSLATE 2026-11-05"

TIP_ID = "dsh-i18n-tip"

TIP_HTML = """\
    <!-- 浏览器自动翻译兜底提示（默认隐藏，页面没挂起来时才显示） -->
    <div id="dsh-i18n-tip" style="display:none;position:fixed;left:50%;bottom:24px;
         transform:translateX(-50%);z-index:2147483647;max-width:min(620px,92vw);
         background:#0b2a5e;border:1px solid #3b82f6;border-radius:12px;
         box-shadow:0 10px 30px rgba(0,0,0,.45);padding:14px 18px;
         color:#e8f0ff;font:14px/1.6 'Microsoft YaHei',Arial,sans-serif;text-align:left;">
      <div style="font-weight:700;font-size:15px;margin-bottom:4px;">⚠️ 页面似乎被浏览器「自动翻译」影响</div>
      <div>试一试 <b>关闭自动翻译</b>，或者在 <b>InPrivate（无痕）</b> 窗口中打开。</div>
      <div style="opacity:.8;font-size:12.5px;margin-top:4px;">中文页面被当成英文翻译时，界面排版会错乱；关闭翻译后刷新即可恢复。</div>
      <div style="margin-top:8px;">
        <a href="javascript:location.reload()" style="color:#bfdbfe;font-weight:700;text-decoration:none;">↻ 刷新页面</a>
      </div>
    </div>
"""

GUARD_JS = """\
    <script>
      (function () {
        try {
          if (!window.__dshDomGuard) {
            var NP = Node.prototype;
            var _rc = NP.removeChild;
            var _ib = NP.insertBefore;
            NP.removeChild = function (child) {
              try { if (child && child.parentNode === this) { return _rc.call(this, child); } } catch (e) {}
              try { console.warn('[dsh] 跳过一次 removeChild：节点已不在父节点下', child); } catch (e2) {}
              return child;
            };
            NP.insertBefore = function (node, ref) {
              try { if (!ref || ref.parentNode === this) { return _ib.call(this, node, ref); } } catch (e) {}
              try { return this.appendChild(node); } catch (e3) { return node; }
            };
            window.__dshDomGuard = true;
            window.__dshDomGuardEarly = true;
          }
        } catch (e) {}
        // 兜底提示：捕获 removeChild 一类错误，或 12 秒后界面仍未挂载 → 显示蓝框
        try {
          var tip = document.getElementById('dsh-i18n-tip');
          function showTip() { if (tip) { tip.style.display = 'block'; } }
          window.addEventListener('error', function (ev) {
            var m = (ev && (ev.message || (ev.error && ev.error.message))) || '';
            if (m.indexOf('removeChild') >= 0 || m.indexOf('NotFoundError') >= 0 || m.indexOf('not a child') >= 0) {
              showTip();
            }
          }, true);
          setTimeout(function () {
            var mounted = document.querySelector('[data-testid="stAppViewContainer"]')
                       || document.querySelector('[data-testid="stSidebar"]')
                       || document.querySelector('#root > div');
            if (!mounted) { showTip(); }
          }, 12000);
        } catch (e2) {}
      })();
    </script>
"""

def _index_path() -> Path | None:
    try:
        import streamlit as _st
        return Path(_st.__file__).resolve().parent / "static" / "index.html"
    except Exception:
        return None

def _apply(html: str) -> str:
    if MARK in html:
        return html
    out = html.replace('<html lang="en">',
                       '<html lang="zh-CN" translate="no" class="notranslate">', 1)
    if 'name="google"' not in out:
        out = out.replace(
            '<meta charset="UTF-8" />',
            '<meta charset="UTF-8" />\n'
            '  <meta name="google" content="notranslate" />\n'
            '  <meta name="microsoft" content="notranslate" />', 1)
    out = out.replace("<body>", '<body class="notranslate">', 1)
    inject = TIP_HTML + GUARD_JS
    anchor = '<script type="module"'
    pos = out.find(anchor)
    if pos >= 0:
        out = out[:pos] + inject + "  " + out[pos:]
    elif "</body>" in out:
        out = out.replace("</body>", inject + "</body>", 1)
    else:
        out = out.replace("</head>", inject + "</head>", 1)
    out = out.replace("</head>", "  <!-- %s -->\n  </head>" % MARK, 1)
    return out

def ensure() -> bool:
    path = _index_path()
    if path is None or not path.exists():
        return False
    try:
        cur = path.read_text(encoding="utf-8")
        new = _apply(cur)
        if new == cur:
            return False
        tmp = path.with_name(path.name + ".dsh_tmp")
        tmp.write_text(new, encoding="utf-8")
        os.replace(tmp, path)
        return True
    except Exception:
        return False

if __name__ == "__main__":
    print("patched:", ensure())
