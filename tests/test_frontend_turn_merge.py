"""前端：客户连发的消息要"聚合成一句话"（后端一个 turn 只回一条）。

实测日志（2026-09-22）：

    客户: i need a led display
    客户: 3*5
    🤖 … indoors or outdoors?                 ← 只回答第一条
    🤖 A 3 by 5 meter screen … fixed or rental? ← 第二条又回了一轮

根因：前端 `flushTurn()` 里有一条

    if (this.isLoading) { turn.timer = setTimeout(() => this.flushTurn(), 800); return; }

上一轮还在生成时，客户补发的消息被**压在本地**，等上一轮回复到达后才单独发出去 →
后端看到的是两个独立 Turn，于是客户收到两条回复、两个问题。

后端本来就有"生成期间补发 → 并入同一轮、重跑、只回一条"的机制
（见 tests/input/test_turn_engine_concurrency.py::TestFollowUpWhileGenerating），
前端必须**立刻把消息送出去**才能触发它。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

APP_JS = os.path.join(project_root, "static", "app.js")
INDEX_HTML = os.path.join(project_root, "static", "index.html")


def _js() -> str:
    import io

    return io.open(APP_JS, encoding="utf-8").read()


def _html() -> str:
    import io

    return io.open(INDEX_HTML, encoding="utf-8").read()


class TestFrontendMessageAggregation:

    def test_follow_up_is_not_held_until_the_previous_reply(self):
        js = _js()
        assert "setTimeout(() => this.flushTurn(), 800)" not in js, (
            "客户补发的消息不能等上一轮回复完再发 —— 必须立刻发出去，"
            "由后端并入正在生成的那一轮（否则客户会收到两条回复）"
        )

    def test_debounce_still_batches_quick_messages(self):
        """同一个 turn 的静默聚合仍然保留（600ms 静默 / 1800ms 上限）。"""
        js = _js()
        assert "turnDebounceMs" in js
        assert "scheduleTurnFlush" in js
        assert "messages: parts.map" in js, "一次请求要带上这一轮的多条消息"

    def test_tracks_inflight_requests_so_typing_indicator_is_correct(self):
        js = _js()
        assert "inflightCount" in js, "可能同时有两条请求在飞"
        assert "if (!this.inflightCount) this.hideTyping()" in js
        # 打字指示器只能有一个（两条请求在飞时不能叠两个）
        assert "if (document.getElementById('typing-indicator')) return;" in js

    def test_merged_response_still_never_renders_a_second_bubble(self):
        """被并入的那一条响应带 duplicate=true → 前端不再渲染第二个气泡。"""
        js = _js()
        assert "data.duplicate" in js
        assert "if (!data.response_count)" in js

    def test_cache_buster_bumped_for_the_changed_js(self):
        html = _html()
        assert "app.js?v=4" in html, "改了 app.js 要升版本号，否则浏览器用旧 JS"
