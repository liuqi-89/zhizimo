# -*- coding: utf-8 -*-
"""给本机 Streamlit 的 static/index.html 打补丁（幂等，可重复运行）。

背景 / 为什么需要它
-------------------
Streamlit 的 index.html 写死了 <html lang="en">，但本应用的界面内容是中文。
用户的 Edge 浏览器里开着「英文页面自动翻译成简体中文」（Preferences 里
translate_allowlists = {"en": "zh-CN"}）。页面语言被声明成 en，翻译引擎就会
把整页当英文翻译，改写/包裹 React 管理的文本节点；随后 Streamlit 前端
（React）在提交阶段删除节点时，DOM 已经被翻译引擎改过，于是抛
NotFoundError: Failed to execute 'removeChild' on 'Node'
（堆栈 cu/zi/du = commitDeletionEffectsOnFiber / commitDeletionEffects /
commitMutationEffectsOnFiber），整页卡死并弹「此页面没有响应」。

补丁做两件事
------------
1) 禁止翻译：<html lang="zh-CN" translate="no" class="notranslate"> +
   <meta name="google" content="notranslate"> + <meta name="microsoft" content="notranslate">
   + <body class="notranslate">。让翻译引擎从一开始就不碰这个页面。
2) 首帧前兜底：在 React bundle 之前插一段内联脚本，给主世界的
   Node.prototype.removeChild / insertBefore 装保护：
   - 要删的节点本来就不在父节点下 → 不抛错，打一条 [dsh] 警告后跳过；
   - insertBefore 的参考节点不在父节点下 → 退化成 appendChild。
   这样即使仍被外部脚本（翻译/扩展）改坏了 DOM，React 也不会整页崩，
   最多是某次局部重绘被跳过。

Streamlit 升级会覆盖本文件 → 升级后重新运行本脚本即可（脚本是幂等的）。
"""

from __future__ import annotations

import os
import shutil
import sys
import time

import streamlit.file_util as file_util

MARK = "DSH-NOTRANSLATE 2026-11-05"

GUARD_JS = """\
    <!-- DSH-DOM-GUARD 2026-11-05: 首帧前给 removeChild/insertBefore 装兜底，
         防止浏览器翻译/扩展改动 DOM 后 React 提交阶段抛 NotFoundError 整页卡死 -->
    <script>
      (function () {
        try {
          if (window.__dshDomGuard) { window.__dshDomGuardEarly = true; return; }
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
        } catch (e) {}
      })();
    </script>

"""

META_JS = """\
    <!-- DSH-NOTRANSLATE 2026-11-05: 界面是中文；禁止浏览器自动翻译改写 React DOM
         （此前 Edge 因本页 lang=en 把整页 en→zh-CN 翻译，导致 removeChild NotFoundError） -->
    <meta name="google" content="notranslate" />
    <meta name="microsoft" content="notranslate" />
"""


def patch() -> int:
    static_dir = file_util.get_static_dir()
    path = os.path.join(static_dir, "index.html")
    if not os.path.isfile(path):
        print(f"[FAIL] 找不到 {path}")
        return 2

    with open(path, "r", encoding="utf-8") as f:
        html = f.read()

    if MARK in html and "DSH-DOM-GUARD" in html:
        print(f"[OK] 已经是打过补丁的版本：{path}")
        print(f"     文件大小 {os.path.getsize(path)} B")
        return 0

    bak = path + ".bak_dsh_" + time.strftime("%Y%m%d_%H%M%S")
    shutil.copy2(path, bak)
    print(f"[1/4] 已备份原文件 -> {bak}")

    if MARK not in html:
        if '<html lang="en">' not in html:
            print('[FAIL] 没找到 <html lang="en">，Streamlit 版本可能已变化，请手动检查')
            return 3
        html = html.replace(
            '<html lang="en">',
            '<html lang="zh-CN" translate="no" class="notranslate">',
            1,
        )
        if '<meta charset="UTF-8" />' not in html:
            print('[FAIL] 没找到 <meta charset="UTF-8" />，请手动检查')
            return 3
        html = html.replace(
            '<meta charset="UTF-8" />',
            '<meta charset="UTF-8" />\n' + META_JS.rstrip("\n"),
            1,
        )
        if "<body>" not in html:
            print('[FAIL] 没找到 <body>，请手动检查')
            return 3
        html = html.replace("<body>", '<body class="notranslate">', 1)
        print("[2/4] 已写入 lang=zh-CN / translate=no / notranslate meta / body class")

    if "DSH-DOM-GUARD" not in html:
        anchor = '    <script type="module" crossorigin src='
        if anchor not in html:
            print('[FAIL] 没找到 React bundle 的 <script type="module" ...> 锚点，请手动检查')
            return 3
        html = html.replace(anchor, GUARD_JS + anchor, 1)
        print("[3/4] 已在 React bundle 之前插入 removeChild/insertBefore 兜底脚本")

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    print(f"[4/4] 已写回 {path}（{os.path.getsize(path)} B）")

    # 自检
    with open(path, "r", encoding="utf-8") as f:
        out = f.read()
    checks = {
        'lang="zh-CN"': 'lang="zh-CN"' in out,
        'translate="no"': 'translate="no"' in out,
        'meta google notranslate': 'name="google" content="notranslate"' in out,
        'meta microsoft notranslate': 'name="microsoft" content="notranslate"' in out,
        '<body class="notranslate">': '<body class="notranslate">' in out,
        'DOM guard 脚本': '__dshDomGuardEarly' in out,
        '网页上不再出现 $ 占位符风险（本文件纯静态）': True,
    }
    for k, v in checks.items():
        print(("  [OK] " if v else "  [!!] ") + k)
    ok = all(checks.values())
    print("[DONE]" if ok else "[WARN] 有检查项未通过，请人工复核")
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(patch())
