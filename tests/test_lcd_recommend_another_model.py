"""回归：LCD/IFP 客户问"还有其他推荐吗"必须**换一个型号**。

客户口径（2026-10 实测）：

    客户: 还有其他开源推荐的吗
    AI  : ...we recommend the Omni T65-K4/K4C...        ← 同一款
    客户: 还有其他可以推荐的吗
    AI  : ...the Omni T65-K4/K4C is the right choice...  ← 还是同一款
    客户: 还有其他可以推荐的吗
    AI  : ...the Omni T65-K4/K4C...                      ← 依然同一款

三个原因叠加，本文件逐一钉住：

  1. ``recommend_node`` 在计算 ``follow_up`` **之前**就把 LCD 分流走了
     → LCD 永远拿不到"客户要换一款"的信号；
  2. ``_locked_lcd_candidate`` 只按需求指纹复用锁定款 → 需求没变就永远是它；
  3. "已经推荐过哪些"没记住：``mark_recommendation_done`` 只认 ``item.model``，
     而 LCD 的推荐结果是 ``Document``（型号在 ``metadata``）→ 记录成空列表。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class _Doc:
    """最小的 Document 替身：LCD 检索结果就是这种（型号在 metadata 里）。"""

    def __init__(self, model: str, **metadata):
        self.metadata = {"model": model, "display_type": "LCD", **metadata}
        self.page_content = f"{model} page"


def _profile():
    from src.models.requirement import RequirementProfile

    profile = RequirementProfile()
    profile.display_type = "LCD"
    profile.lcd_category = "advertising"
    profile.lcd_is_splicing = True
    profile.lcd_splicing_layout = "3x3"
    profile.lcd_screen_count = 9
    profile.lcd_size_inch = 65.0
    profile.lcd_resolution = "4K"
    profile.environment = "indoor"
    profile.sources.update(
        {
            "lcd_is_splicing": "explicit",
            "lcd_splicing_layout": "explicit",
            "lcd_screen_count": "explicit",
            "lcd_size_inch": "explicit",
            "lcd_resolution": "explicit",
            "environment": "explicit",
        }
    )
    return profile


def _products():
    # 两款得分相同（同尺寸 / 同分辨率 / 同为拼接面板）→ 先出现的优先；
    # 排除掉第一款之后，第二款必须顶上来。
    return [
        _Doc("H6530LN-B", display_size_inch="65", resolution="4K",
             is_splicing=True, bazel_mm="3.5", brightness_nit=500),
        _Doc("H6518LN-B", display_size_inch="65", resolution="4K",
             is_splicing=True, bazel_mm="1.7", brightness_nit=500),
    ]


class TestCandidateSelectionCanExclude:

    def test_first_choice_then_the_next_one(self):
        from src.rag.lcd_recommendation import select_lcd_candidate

        profile = _profile()
        first = select_lcd_candidate(profile, _products())
        assert first[1]["model"] == "H6530LN-B"

        second = select_lcd_candidate(profile, _products(), exclude=["H6530LN-B"])
        assert second[1]["model"] == "H6518LN-B", "排除已给过的型号后要换一款"

    def test_all_excluded_falls_back_instead_of_returning_nothing(self):
        """都推荐过了也要给得出东西（宁可重复一次，也不能给空）。"""
        from src.rag.lcd_recommendation import select_lcd_candidate

        picked = select_lcd_candidate(
            _profile(), _products(), exclude=["H6530LN-B", "H6518LN-B"]
        )
        assert picked is not None
        assert picked[1]["model"] in ("H6530LN-B", "H6518LN-B")


class TestRecommendAnotherModel:

    def _state(self, session_id):
        return {
            "session_id": session_id,
            "current_message": "还有其他推荐的吗",
            "already_recommended": True,
        }

    def test_follow_up_switches_to_a_model_not_offered_yet(self):
        from src.agents.solution.nodes.recommend import _recommend_lcd
        from src.memory.store import memory

        session_id = "lcd-another-1"
        memory.clear(session_id)
        try:
            profile = _profile()
            state = self._state(session_id)

            # 第一次推荐 → 锁定首选款
            first = _recommend_lcd(state, profile, _products())
            assert first["recommendation_result"]["selected_model"] == "H6530LN-B"

            # 需求没变、客户没要别的 → 仍然复用锁定款（不能乱换）
            keep = _recommend_lcd(self._state(session_id), profile, _products())
            assert keep["recommendation_result"]["selected_model"] == "H6530LN-B"

            # 客户问"还有其他推荐吗" → 必须换成另一款
            other = _recommend_lcd(
                self._state(session_id),
                profile,
                _products(),
                follow_up=True,
                previous_models=["H6530LN-B"],
            )
            assert other["recommendation_result"]["selected_model"] == "H6518LN-B", (
                "客户要别的推荐，不能再推同一款"
            )
            assert other["products"][0].metadata["model"] == "H6518LN-B"

            # 再问一次 → 轮到还没给过的（这里只剩已给过的两款，允许兜底重复）
            third = _recommend_lcd(
                self._state(session_id),
                profile,
                _products(),
                follow_up=True,
                previous_models=["H6530LN-B", "H6518LN-B"],
            )
            assert third["recommendation_result"]["selected_model"] in (
                "H6530LN-B", "H6518LN-B",
            )
        finally:
            memory.clear(session_id)

    def test_follow_up_does_not_reuse_the_locked_model(self):
        """即使锁定款仍然满足需求，follow_up 也必须绕过锁定。"""
        from src.agents.solution.nodes.recommend import _recommend_lcd
        from src.memory.store import memory

        session_id = "lcd-another-2"
        memory.clear(session_id)
        try:
            profile = _profile()
            _recommend_lcd(self._state(session_id), profile, _products())
            locked = memory.get_lcd_lock(session_id)
            assert locked and locked.get("model") == "H6530LN-B", locked

            again = _recommend_lcd(
                self._state(session_id), profile, _products(),
                follow_up=True, previous_models=["H6530LN-B"],
            )
            assert again["recommendation_result"]["selected_model"] != locked["model"]
        finally:
            memory.clear(session_id)


class TestRecommendedModelsAreRemembered:

    def test_document_metadata_model_is_recorded(self):
        """LCD 的 Document（型号在 metadata）也必须被记进"已推荐"。"""
        from src.memory.store import memory

        session_id = "lcd-memory-1"
        memory.clear(session_id)
        try:
            memory.mark_recommendation_done(session_id, _products())
            record = memory.get_recommendation(session_id)
            assert record.get("delivered") is True
            assert record.get("models") == ["H6530LN-B", "H6518LN-B"], record
        finally:
            memory.clear(session_id)

    def test_dict_products_still_work(self):
        """LED 走的是 dict（model 键）—— 老口径不能被弄坏。"""
        from src.memory.store import memory

        session_id = "led-memory-1"
        memory.clear(session_id)
        try:
            memory.mark_recommendation_done(
                session_id, [{"model": "TW11-3216"}, {"series": "TW21-3216"}]
            )
            assert memory.get_recommendation(session_id)["models"] == [
                "TW11-3216", "TW21-3216",
            ]
        finally:
            memory.clear(session_id)
