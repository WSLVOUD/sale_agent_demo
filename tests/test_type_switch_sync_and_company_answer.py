"""回归：① 改口换产品大类必须同步进档案；② 代理商类提问任何意图都要照公司资料答。

客户 2026-10 实测两个 bug：

  ① 客户先说 LCD，后来 "i change my mind ,i need a led"。
     问句已经按 LED 走（点间距 / 固装租赁），但最后推的是 **LCD** 户外广告屏
     （DS-O-75）—— 因为产品类型有两个存放处：
         state["display_type_decision"]  类型路由的结论（问句 / 链路分流用）
         profile.display_type             需求档案（**检索与推荐**用）
     改口只更新了前者。

  ② 客户问 "你们在印度尼西亚有代理商吗"，
     系统回 "I'm not sure about Indonesia … let me check with our team"。
     可 data/company_profile.txt 里明明写着"只有一个点、**没有代理商**"。
     原因：公司信息只在 objection / product_question / others 三个意图分支里答，
     而这一轮的意图是 need_query（正在采集需求）→ 没有这条分支。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _lcd_profile():
    from src.models.requirement import RequirementProfile

    profile = RequirementProfile()
    profile.display_type = "LCD"
    profile.lcd_category = "advertising"
    profile.lcd_size_inch = 65.0
    profile.lcd_is_splicing = True
    profile.lcd_splicing_layout = "3x3"
    profile.lcd_screen_count = 9
    profile.lcd_bezel_mm = 0.88
    profile.pixel_pitch_mm = 3.0
    profile.sources.update(
        {
            "display_type": "customer_explicit",
            "lcd_category": "understanding",
            "lcd_size_inch": "explicit",
            "lcd_is_splicing": "explicit",
            "lcd_splicing_layout": "explicit",
            "lcd_screen_count": "explicit",
            "lcd_bezel_mm": "explicit",
            "pixel_pitch_mm": "explicit",
        }
    )
    return profile


class TestProductTypeSwitchSyncsIntoTheProfile:

    def _state(self, display_type, status="CONFIRMED", locked=True):
        return {
            "display_type_decision": {
                "display_type": display_type, "status": status, "locked": locked,
            }
        }

    def test_lcd_to_led_switches_the_profile_and_drops_lcd_facts(self):
        from src.agents.sales.nodes.requirement import _sync_profile_display_type

        profile = _lcd_profile()
        cleared = _sync_profile_display_type(self._state("LED"), profile)

        assert profile.display_type == "LED", profile.display_type
        assert profile.sources.get("display_type") == "customer_explicit"
        # LCD 口径的需求事实必须作废（否则会把英寸/拼接带进 LED 链路）
        for name in ("lcd_category", "lcd_size_inch", "lcd_is_splicing",
                     "lcd_splicing_layout", "lcd_screen_count", "lcd_bezel_mm",
                     "pixel_pitch_mm"):
            assert getattr(profile, name, None) is None, (name, getattr(profile, name))
            assert name not in (profile.sources or {}), name
        assert "lcd_size_inch" in cleared and "lcd_is_splicing" in cleared

    def test_led_to_lcd_switches_back(self):
        from src.agents.sales.nodes.requirement import _sync_profile_display_type
        from src.models.requirement import RequirementProfile

        profile = RequirementProfile()
        profile.display_type = "LED"
        profile.pixel_pitch_mm = 4.0
        profile.sources["display_type"] = "customer_explicit"
        profile.sources["pixel_pitch_mm"] = "explicit"

        _sync_profile_display_type(self._state("LCD"), profile)
        assert profile.display_type == "LCD"
        assert profile.pixel_pitch_mm is None

    def test_unconfirmed_inference_never_rewrites_the_profile(self):
        """只是推断（未锁定）时不许写档案 —— 别把猜测当客户事实。"""
        from src.agents.sales.nodes.requirement import _sync_profile_display_type

        profile = _lcd_profile()
        assert _sync_profile_display_type(
            self._state("LED", status="INFERRED", locked=False), profile
        ) == []
        assert profile.display_type == "LCD"
        assert profile.lcd_size_inch == 65.0

    def test_ifp_is_not_downgraded_to_lcd(self):
        """IFP 是 LCD 的子类型，由 LCD 链路自己判 —— 这里不许把它降级。"""
        from src.agents.sales.nodes.requirement import _sync_profile_display_type

        profile = _lcd_profile()
        profile.display_type = "IFP"
        assert _sync_profile_display_type(self._state("LCD"), profile) == []
        assert profile.display_type == "IFP"

    def test_same_type_is_a_no_op(self):
        from src.agents.sales.nodes.requirement import _sync_profile_display_type

        profile = _lcd_profile()
        assert _sync_profile_display_type(self._state("LCD"), profile) == []
        assert profile.lcd_size_inch == 65.0


class TestCompanyQuestionAnsweredFromTheProfile:

    def test_the_chinese_agent_question_is_recognised(self):
        from src.rag.company_info import is_company_question

        assert is_company_question("你们在印度尼西亚有代理商吗") is True
        assert is_company_question("do you have a distributor in Indonesia?") is True
        assert is_company_question("where are you located?") is True
        # 跟公司无关的话不能被当成公司问题
        assert is_company_question("i need a 3x5 led screen") is False

    def test_the_answer_says_there_is_no_agent(self):
        """答案必须来自 company_profile.txt：只有一个点、没有代理商。"""
        from src.rag.company_info import company_answer

        answer = company_answer("你们在印度尼西亚有代理商吗", language="en", seed=0)
        assert answer, "必须给出公司资料的答案"
        lowered = answer.lower()
        assert "distributor" in lowered or "agent" in lowered or "branch" in lowered
        assert "do not" in lowered or "no " in lowered
        # 不能是"我不确定/回头问团队"这种搪塞
        assert "not sure" not in lowered
        assert "check with our team" not in lowered

    def test_company_question_is_answered_even_while_collecting_requirements(self):
        """意图是 need_query（需求采集中）时也要照公司资料回答。"""
        from src.agents.sales.nodes.script_generator import script_generator

        message = "你们在印度尼西亚有代理商吗"
        state = {
            "session_id": "company-mid-collection",
            "intent": "need_query",
            "current_message": message,
            "requirements": {"display_type": "LED"},
            "should_generate_solution": False,
            "response": "",
            "next_action": "",
        }
        out = script_generator(state)
        reply = str(out.get("response") or "")
        assert reply, "必须有回复"
        lowered = reply.lower()
        assert "not sure" not in lowered, reply
        assert "check with our team" not in lowered, reply
        assert ("distributor" in lowered or "agent" in lowered or "branch" in lowered), reply
