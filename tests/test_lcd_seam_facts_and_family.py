"""回归：LCD 推荐话术里的**拼缝事实**与**产品类别**不许编造。

客户口径（2026-10 实测）：

    客户: yes,0.88mm                ← 明确要求 0.88mm 拼缝
    AI  : … H6530LN-B … a 3.5mm bezel, meeting your seam requirement
    客户: 还有其他推荐的吗
    AI  : … P65 … a large, seamless visual surface …

三个硬伤：

  1. **拼缝不达标却说达标**：H6530LN-B 的 bezel 是 3.5mm，客户要 ≤0.88mm，
     话术却说 "meeting your seam requirement" —— 编造事实；
  2. **把单屏商用显示器当拼接墙面板**：P65 的产品数据是 ``is_splicing=False``
     （P 系列单屏显示器，能拼但边框明显），话术却把它说成 video wall；
  3. **把 LCD 拼接说成"无缝"**：产品数据明写 "seam is visible (not seamless)"。

本文件锁的是"事实口径"，不是关键词路由：
  · 拼缝达没达到由 Python 确定性判断，并**明确写进提示词**；
  · 输出侧只做**从句级**改写（"满足…"→"达不到…"）；"无缝"这类词一旦出现
    就退回确定性模板（词级替换会写出病句）；
  · 产品类别由产品数据推导（video wall / single display / IFP）。
"""
import os
import sys
from types import SimpleNamespace

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 真实产品数据的口径（取自 data/lcd_products.json）
H6530 = {
    "model": "H6530LN-B", "display_type": "LCD", "is_splicing": True,
    "splicing_supported": True, "bazel_mm": "3.5mm",
    "splicing_note": "can be spliced into a video wall, but the seam is visible (bezel 3.5mm)",
}
H5510 = {
    "model": "H5510LN/HN-B", "display_type": "LCD", "is_splicing": True,
    "splicing_supported": True, "bazel_mm": "0.88mm",
    "splicing_note": "can be spliced, but the seam is visible (bezel 0.88mm)",
}
P65 = {
    "model": "P65", "display_type": "LCD", "is_splicing": False,
    "splicing_supported": True,
    "splicing_note": "can be tiled, but the frame/seam is visible (not seamless)",
}


def _profile(bezel=0.88):
    from src.models.requirement import RequirementProfile

    profile = RequirementProfile()
    profile.display_type = "LCD"
    profile.purpose = "exhibition"
    profile.lcd_is_splicing = True
    profile.lcd_bezel_mm = bezel
    profile.lcd_size_inch = 65.0
    return profile


class TestSeamTruth:

    def test_bezel_is_read_from_the_product_data(self):
        from src.agents.solution.nodes.recommend import _bezel_mm

        assert _bezel_mm(H6530) == 3.5
        assert _bezel_mm(H5510) == 0.88
        assert _bezel_mm(P65) is None          # 单屏显示器没有拼缝数据

    def test_requirement_met_is_computed_not_guessed(self):
        from src.agents.solution.nodes.recommend import _seam_requirement_met

        profile = _profile(0.88)
        assert _seam_requirement_met(profile, 3.5) is False, "3.5mm 达不到 0.88mm"
        assert _seam_requirement_met(profile, 0.88) is True
        assert _seam_requirement_met(profile, None) is False, "拿不到拼缝就不能说达标"
        # 客户没提拼缝 → 没有可比的要求
        assert _seam_requirement_met(_profile(None), 3.5) is True

    def test_seam_facts_tell_the_model_the_truth(self):
        from src.agents.solution.nodes.recommend import _seam_facts

        profile = _profile(0.88)
        not_met = _seam_facts(profile, H6530, 3.5)
        assert "does NOT meet" in not_met, not_met
        assert "visible" in not_met.lower(), not_met

        met = _seam_facts(profile, H5510, 0.88)
        assert "DOES meet" in met, met

    def test_false_meets_claim_is_rewritten(self):
        from src.agents.solution.nodes.recommend import _guard_lcd_seam_claims

        text = ("The H6530LN-B comes with a 3.5mm bezel, meeting your seam requirement. "
                "Shall I prepare the quotation?")
        fixed = _guard_lcd_seam_claims(
            text, seam_is_visible=True, seam_met=False, bezel_mm=3.5, want_bezel_mm=0.88
        )
        assert "meeting your seam requirement" not in fixed.lower(), fixed
        assert "does not reach" in fixed.lower(), fixed
        assert "0.88mm" in fixed, fixed
        # 型号与其余内容必须保留（不是整句删除）
        assert "H6530LN-B" in fixed, fixed
        assert "quotation" in fixed, fixed

    def test_compliant_text_is_left_alone(self):
        from src.agents.solution.nodes.recommend import _guard_lcd_seam_claims

        good = ("The H5510LN/HN-B has a 0.88mm bezel and meets your seam requirement. "
                "The seam between panels stays visible.")
        assert _guard_lcd_seam_claims(
            good, seam_is_visible=True, seam_met=True, bezel_mm=0.88, want_bezel_mm=0.88
        ) == good


class TestProductFamilyWording:

    def test_video_wall_panel_vs_single_display(self):
        from src.agents.solution.nodes.recommend import _lcd_product_family

        profile = _profile()
        assert _lcd_product_family(profile, H6530).startswith("LCD video wall panel")
        single = _lcd_product_family(profile, P65)
        assert single.startswith("commercial LCD monitor"), single
        assert "not a video wall panel" in single, single
        assert "visible" in single, single

    def test_ifp_is_named_as_an_interactive_flat_panel(self):
        from src.agents.solution.nodes.recommend import _lcd_product_family
        from src.models.requirement import RequirementProfile

        profile = RequirementProfile()
        profile.display_type = "LCD"
        profile.lcd_handwriting_required = True
        assert _lcd_product_family(profile, P65).startswith("interactive flat panel")


class TestNoSeamlessClaimReachesTheCustomer:

    def _express(self, monkeypatch, llm_text, *, seam_met, metadata):
        import importlib

        rec = importlib.import_module("src.agents.solution.nodes.recommend")

        class _FakeLLM:
            def invoke(self, prompt, *args, **kwargs):
                return SimpleNamespace(content=llm_text)

        monkeypatch.setattr(rec, "get_llm", lambda **kwargs: _FakeLLM())
        from src.agents.solution.nodes.recommend import _express_lcd_recommendation

        return _express_lcd_recommendation(
            model=str(metadata["model"]),
            facts=["65\" panels", "3.5mm bezel"],
            reasons=["video-wall panels with 3.5mm bezel"],
            layout="3x3 layout (9 panels)",
            profile=_profile(0.88),
            state={"session_id": "seam-guard"},
            seam_facts=rec._seam_facts(_profile(0.88), metadata, rec._bezel_mm(metadata)),
            seam_is_visible=rec._seam_is_visible(metadata),
            seam_met=seam_met,
            product_family_override=rec._lcd_product_family(_profile(0.88), metadata),
            bezel_mm=rec._bezel_mm(metadata),
        )

    def test_seamless_wording_is_never_returned(self, monkeypatch):
        """模型写出"无缝" → 必须退回确定性模板（模板讲实话，不说无缝）。"""
        text = self._express(
            monkeypatch,
            "For your exhibition, the H6530LN-B gives you a seamless visual surface with a "
            "3.5mm bezel, meeting your seam requirement. Shall I prepare the quotation?",
            seam_met=False,
            metadata=H6530,
        )
        assert "seamless" not in text.lower(), text
        assert "H6530LN-B" in text, text
        # 模板要把"达不到 0.88mm"说出来
        assert "0.88mm" in text, text
        assert "does not reach" in text.lower(), text

    def test_a_good_reply_passes_through_untouched(self, monkeypatch):
        good = ("For your exhibition, the H5510LN/HN-B matches the seam you asked for and "
                "keeps 4K detail. Shall I prepare the quotation?")
        text = self._express(monkeypatch, good, seam_met=True, metadata=H5510)
        assert text == good


class TestNoSeamTalkWithoutASplicingRequirement:
    """客户没说拼接 → 一个字都不许提拼缝 / 多面板（客户口径 2026-10）。

    实测原话（客户只买一块 65" 单屏）：

        P65 is the closest fit for your requirement: 65" panels, 3840*2160 resolution,
        350nit brightness. The seam between panels stays visible. Shall I prepare the quotation?

    客户从没说要拼接，却被聊了一句"面板之间的拼缝是可见的"。
    """

    def _single_profile(self):
        from src.models.requirement import RequirementProfile

        profile = RequirementProfile()
        profile.display_type = "LCD"
        profile.purpose = "exhibition"
        profile.lcd_size_inch = 65.0
        profile.lcd_is_splicing = False        # ← 客户没要拼接
        return profile

    def test_splicing_requirement_flag(self):
        from src.agents.solution.nodes.recommend import _splicing_requirement

        assert _splicing_requirement(self._single_profile()) is False
        assert _splicing_requirement(_profile(0.88)) is True     # _profile 是拼接需求

    def test_seam_facts_tell_the_model_not_to_raise_seams(self):
        from src.agents.solution.nodes.recommend import _seam_facts

        facts = _seam_facts(self._single_profile(), P65, None)
        assert "do not mention seams" in facts, facts
        assert "visible" not in facts.lower(), facts

    def test_template_says_nothing_about_seams(self):
        from src.agents.solution.nodes.recommend import _seam_truth_sentence

        assert _seam_truth_sentence(
            seam_is_visible=True, seam_met=True, bezel_mm=None,
            want_bezel_mm=None, splicing_requirement=False,
        ) == ""

    def test_fallback_template_for_a_single_screen_has_no_seam_sentence(self, monkeypatch):
        """模型调用失败时走确定性模板 —— 单屏模板里不该有拼缝。"""
        import importlib

        rec = importlib.import_module("src.agents.solution.nodes.recommend")

        def _boom(**kwargs):
            raise RuntimeError("no llm in test")

        monkeypatch.setattr(rec, "get_llm", _boom)
        profile = self._single_profile()
        text = rec._express_lcd_recommendation(
            model="P65",
            facts=["65-inch screen", "3840*2160 resolution", "350nit brightness"],
            reasons=[],
            layout="",
            profile=profile,
            state={"session_id": "single-screen"},
            seam_facts=rec._seam_facts(profile, P65, None),
            seam_is_visible=False,
            seam_met=True,
            product_family_override=rec._lcd_product_family(profile, P65),
            bezel_mm=None,
            splicing_requirement=False,
        )
        assert "P65" in text, text
        assert "seam" not in text.lower(), text
        assert "panels" not in text.lower(), text

