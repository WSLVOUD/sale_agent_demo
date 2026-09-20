"""
新增两问（客户口径）回归测试：

  1. 问完场景 → 紧接着问"放视频还是放图片"（含"两者都有"）；只记录，不影响选型；
  2. 推荐前问"最看重价格还是质量"：价格/都看重 → 默认档；只看质量 → 中等价位款；
     客户自己说过预算 → 不问。

注：推荐后"收集联系方式"那一问已按客户口径**删除**（2026-09-18），
相关回归测试同步移除。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.query_understanding import extract_slots  # noqa: E402
from src.rag.readiness import (  # noqa: E402
    EASIER_QUESTIONS,
    MISSING_ORDER,
    QUESTION_VARIANTS,
    check_recommendation_ready,
    question_for,
)


class TestContentTypeQuestion:

    def test_asked_right_after_scenario(self):
        # 客户口径（2026-09-18）：硬性条件（室内外 / 固装租赁 / P值 / 尺寸）先问，
        # 场景、内容类型、价格取向都排在后面且不阻塞推荐。
        assert MISSING_ORDER.index("environment") < MISSING_ORDER.index("installation")
        assert MISSING_ORDER.index("installation") < MISSING_ORDER.index("pixel_pitch")
        assert MISSING_ORDER.index("pixel_pitch") < MISSING_ORDER.index("size")
        assert MISSING_ORDER.index("size") < MISSING_ORDER.index("purpose")
        assert MISSING_ORDER.index("purpose") < MISSING_ORDER.index("content_type")

        profile = RequirementProfile.from_slots(
            {"display_type": "LED", "environment": "indoor", "purpose": "church"},
            explicit_keys={"display_type", "environment", "purpose"},
        )
        decision = check_recommendation_ready(profile)
        # 只说了"教堂" → 缺硬性条件，先问固装租赁（室内外已由场景确定）
        assert "environment" not in decision.missing
        assert decision.missing[0] == "installation"
        assert "rental" in (decision.next_question or "").lower()

    def test_every_variant_offers_both(self):
        for language in ("en", "zh"):
            variants = QUESTION_VARIANTS["content_type"][language]
            assert len(variants) >= 4
            for text in variants:
                assert ("both" in text.lower()) or ("都有" in text), text
        for text in EASIER_QUESTIONS["content_type"]["zh"]:
            assert "都有" in text, text

    def test_variants_rotate(self):
        texts = {question_for("content_type", "en", seed) for seed in range(8)}
        assert len(texts) >= 3

    def test_answer_parsing(self):
        assert extract_slots("放视频").get("content_type") == "video"
        assert extract_slots("主要显示图片").get("content_type") == "image"
        assert extract_slots("两种都有").get("content_type") == "mixed"
        assert extract_slots("视频和图片都要").get("content_type") == "mixed"
        assert extract_slots("a mix of both").get("content_type") == "mixed"

    def test_does_not_change_selection(self):
        """内容类型只记录，不影响选型结果。"""
        from src.rag.recommendation_service import RecommendationService

        base = {
            "display_type": "LED", "environment": "indoor", "purpose": "conference",
            "content_type": "mixed", "installation": "fixed", "viewing_distance_m": 10,
            "price_preference": "price",
            "target_width_mm": 5000, "target_height_mm": 3000,
        }
        service = RecommendationService()
        without = service.recommend(
            RequirementProfile.from_slots(base, explicit_keys=set(base))
        )
        profile = RequirementProfile.from_slots(
            {**base, "content_type": "video"}, explicit_keys=set(base) | {"content_type"}
        )
        with_video = service.recommend(profile)

        assert with_video["recommendation_status"] == "RECOMMENDED"
        assert [r["model"] for r in with_video["recommendations"]] == [
            r["model"] for r in without["recommendations"]
        ]


class TestPricePreferenceQuestion:

    def _profile(self, **extra):
        slots = {
            "display_type": "LED", "environment": "indoor", "purpose": "conference",
            "content_type": "mixed", "installation": "fixed", "viewing_distance_m": 10,
            # 客户口径：尺寸 / P值 / 室内外 / 固装租赁 是硬性条件
            "target_width_mm": 5000, "target_height_mm": 3000,
        }
        slots.update(extra)
        return RequirementProfile.from_slots(slots, explicit_keys=set(slots))

    def test_price_preference_does_not_block_recommendation(self):
        """客户口径（2026-09-18）：价格取向是非硬性项，不再阻塞推荐。"""
        assert MISSING_ORDER[-1] == "price_preference"
        decision = check_recommendation_ready(self._profile())
        assert decision.ready is True
        assert "price_preference" not in decision.missing

    def test_not_asked_when_budget_already_stated(self):
        decision = check_recommendation_ready(self._profile(budget_level="high"))
        assert decision.ready is True
        assert "price_preference" not in decision.missing

    def test_quality_preference_maps_to_middle_tier(self):
        profile = self._profile(price_preference="quality")
        assert profile.budget_level == "medium"
        assert check_recommendation_ready(profile).ready is True

    def test_price_and_both_map_to_default_tier(self):
        for preference in ("price", "both"):
            profile = self._profile(price_preference=preference)
            assert profile.budget_level == "low", preference

    def test_quality_recommends_middle_tier_model(self):
        from src.rag.recommendation_service import RecommendationService

        profile = self._profile(price_preference="quality")
        assert profile.budget_level == "medium"
        result = RecommendationService().recommend(profile)
        assert result["recommendation_status"] == "RECOMMENDED"
        assert result["recommendations"]

    def test_price_preference_parsing(self):
        assert extract_slots("我们更看重质量").get("price_preference") == "quality"
        assert extract_slots("价格优先").get("price_preference") == "price"
        assert extract_slots("价格和质量都看重").get("price_preference") == "both"
        # "视频和图片都要" 说的是内容类型，不能被当成"价格+质量都要"
        assert extract_slots("视频和图片都要").get("price_preference") is None

    def test_unambiguous_both_phrases_are_parsed(self):
        """客户口径：both are fine / either works 这类说法要算"两者都行"。"""
        for text in (
            "both are fine",
            "both is fine",
            "either is fine",
            "either works",
            "两者都行",
            "两个都可以",
        ):
            assert extract_slots(text).get("price_preference") == "both", text

    def test_bare_both_answer_is_recognised_for_price_question(self):
        """实测 bug：问"最看重价格还是质量"客户只回 "both" → 系统当成没回答。

        修复：结合"上一轮问的就是这一项"把 bare both 落到 price_preference=both，
        预算按默认档（最便宜优先）走，Gate 直接放行、不再重复追问。
        """
        from src.core.requirement_extractor import get_requirement_extractor

        slots = {
            "display_type": "LED", "environment": "indoor", "purpose": "church",
            "content_type": "mixed", "installation": "fixed",
            "pixel_pitch_mm": 3.0, "viewing_distance_m": 5,
            "target_width_mm": 5000, "target_height_mm": 3000,
        }
        previous = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        previous.last_asked_slot = "price_preference"
        # 客户口径：价格取向不再阻塞推荐（硬性条件已经齐）
        assert check_recommendation_ready(previous).ready is True

        profile = get_requirement_extractor().extract(
            "both", previous_profile=previous, use_llm=False, session_id=""
        )
        assert profile.price_preference == "both"
        assert profile.budget_level == "low", "两者都行 → 按默认档（最便宜优先）"
        assert check_recommendation_ready(profile).ready is True

    def test_bare_both_answer_follows_the_asked_slot(self):
        """"both" 落在哪个槽位由"上一轮问的是什么"决定，不能互相污染。"""
        from src.core.requirement_extractor import get_requirement_extractor

        previous = RequirementProfile.from_slots({}, explicit_keys=set())
        previous.last_asked_slot = "content_type"
        profile = get_requirement_extractor().extract(
            "both", previous_profile=previous, use_llm=False, session_id=""
        )
        assert profile.content_type == "mixed"
        assert profile.price_preference is None

    def test_wording_does_not_quote_prices(self):
        for language in ("en", "zh"):
            for text in QUESTION_VARIANTS["price_preference"][language]:
                assert "$" not in text and "美元" not in text and "元" not in text
                assert ("price" in text.lower()) or ("价格" in text), text


class TestOffTopicAckUsesHigherTemperature:
    """客户说了与需求无关的话时：

    "接住这句话"的话术单独用 config.ACK_TEMPERATURE（默认 0.7）生成，措辞更发散；
    需求抽取仍用 temperature=0；并且**绝不允许**回答知识性问题。
    """

    @pytest.fixture
    def recording_llm(self, monkeypatch):
        import importlib

        sales_req = importlib.import_module("src.agents.sales.nodes.requirement")
        extractor_mod = importlib.import_module("src.core.requirement_extractor")
        calls = []

        class _Response:
            def __init__(self, content):
                self.content = content

        class _FakeChat:
            def __init__(self, *args, **kwargs):
                self.temperature = kwargs.get("temperature")

            def invoke(self, messages, *args, **kwargs):
                system = str(getattr(messages[0], "content", ""))
                calls.append({"temperature": self.temperature, "system": system})
                if "与产品需求无关" in system:
                    return _Response("Haha, sounds good — let's take it one step at a time.")
                return _Response('{"usage": null, "additional_requirements": [], "ack": ""}')

        monkeypatch.setattr(sales_req, "ChatOpenAI", _FakeChat)
        monkeypatch.setattr(
            extractor_mod.RequirementExtractor,
            "_llm_semantic_extract",
            lambda self, message, rule_slots, session_id="": {},
        )
        extractor_mod.RequirementExtractor._semantic_cache.clear()
        return sales_req, calls

    def _turn(self, module, message, profile=None):
        state = {
            "messages": [{"role": "user", "content": message}],
            "current_message": message,
            "session_id": "ack-temperature",
            "requirements": {},
            "additional_requirements": [],
            "intent": "need_query",
            "next_action": "ask",
            "should_generate_solution": False,
            "response": "",
            "pending_question": "",
            "pending_slot": "",
            "acknowledgement": "",
        }
        if profile is not None:
            state["requirement_profile"] = profile
        return module.requirement_mining(state)

    def test_offtopic_message_uses_ack_temperature(self, recording_llm):
        module, calls = recording_llm
        turn = self._turn(module, "haha nice weather today")

        temperatures = [call["temperature"] for call in calls]
        assert 0 in temperatures, "需求抽取仍然是 temperature=0"
        assert 0.7 in temperatures, "接话话术要用 0.7"
        assert turn["acknowledgement"].startswith("Haha")

    def test_requirement_turn_does_not_call_the_ack_model(self, recording_llm):
        module, calls = recording_llm
        turn = self._turn(module, "I need an indoor LED screen for a church")

        assert all(call["temperature"] != 0.7 for call in calls), "正常回答需求时不额外调用"
        assert "与产品需求无关" not in " ".join(call["system"] for call in calls)

    def test_ack_prompt_forbids_knowledge_answers(self, recording_llm):
        module, calls = recording_llm
        self._turn(module, "haha nice weather today")

        ack_systems = [call["system"] for call in calls if "与产品需求无关" in call["system"]]
        assert ack_systems, "应当有专门的接话调用"
        prompt = ack_systems[0]
        assert "绝对不要" in prompt and "知识性" in prompt, prompt
        assert "不要提问" in prompt
        assert "不要给参数、型号、价格、方案或建议" in prompt

    def test_ack_temperature_is_configurable(self):
        from src.config import config

        assert float(config.ACK_TEMPERATURE) == 0.7


class TestOffTopicVsBusinessQuestion:
    """闲聊式问句（"do u like watching TV series?"）也要算"无关话"：

    只接住这句话 + 继续问需求，**不能**走自由问答去倒产品/参数。
    但真正的业务问题（产品/规格/价格/交期/公司/质保）仍要正面回答。
    """

    def test_offtopic_questions_are_not_business(self):
        from src.agents.sales.nodes.requirement import _is_product_or_business_question

        for message in (
            "do u like watching TV serises?",
            "haha nice weather today",
            "my cousin also runs a shop there",
            "how are you doing today?",
        ):
            assert _is_product_or_business_question(message) is False, message

        for message in (
            "do you have P1.2 COB LED?",
            "what is the price?",
            "how long does delivery take?",
            "where are you located?",
            "tell me about your warranty",
            "i need an indoor led screen for a church",
        ):
            assert _is_product_or_business_question(message) is True, message

    def test_offtopic_question_only_acks_and_asks_requirement(self):
        """节点级：闲聊问句 → 回复里不能出现型号/参数，且必须继续问需求。"""
        import importlib

        sales_req = importlib.import_module("src.agents.sales.nodes.requirement")
        extractor_mod = importlib.import_module("src.core.requirement_extractor")

        class _Response:
            def __init__(self, content):
                self.content = content

        class _FakeChat:
            def __init__(self, *args, **kwargs):
                self.temperature = kwargs.get("temperature")

            def invoke(self, messages, *args, **kwargs):
                system = str(getattr(messages[0], "content", ""))
                if "与产品需求无关" in system:
                    return _Response("Haha, I do enjoy a good series now and then.")
                return _Response('{"usage": null, "additional_requirements": [], "ack": ""}')

        import pytest as _pytest

        monkey = _pytest.MonkeyPatch()
        try:
            monkey.setattr(sales_req, "ChatOpenAI", _FakeChat)
            monkey.setattr(
                extractor_mod.RequirementExtractor,
                "_llm_semantic_extract",
                lambda self, message, rule_slots, session_id="": {},
            )
            extractor_mod.RequirementExtractor._semantic_cache.clear()

            state = {
                "messages": [{"role": "user", "content": "do u like watching TV serises?"}],
                "current_message": "do u like watching TV serises?",
                "session_id": "offtopic-question",
                "requirements": {},
                "additional_requirements": [],
                "intent": "others",
                "next_action": "others",
                "should_generate_solution": False,
                "response": "",
                "pending_question": "",
                "pending_slot": "",
                "acknowledgement": "",
                "requirement_profile": RequirementProfile.from_slots(
                    {"display_type": "LED", "environment": "indoor", "purpose": "church"},
                    explicit_keys={"display_type", "environment", "purpose"},
                ),
            }
            out = sales_req.requirement_mining(state)

            assert out["intent"] == "need_query", "闲聊不要说成 others（否则会绕到自由问答）"
            assert out["offtopic_turn"] is True
            assert out["pending_question"], "必须继续问需求"
            assert out["acknowledgement"].startswith("Haha")
        finally:
            monkey.undo()


class TestTwoAskLimitForNewQuestions:
    """客户口径：新问题同样"最多问两次、第二次换问法"，两次都没有就跳过。"""

    @pytest.fixture
    def sales_node(self, monkeypatch):
        import importlib

        sales_req = importlib.import_module("src.agents.sales.nodes.requirement")
        extractor_mod = importlib.import_module("src.core.requirement_extractor")

        class _Response:
            content = '{"usage": "conference", "additional_requirements": [], "ack": ""}'

        class _FakeChat:
            def __init__(self, *args, **kwargs):
                pass

            def invoke(self, *args, **kwargs):
                return _Response()

        monkeypatch.setattr(sales_req, "ChatOpenAI", _FakeChat)
        monkeypatch.setattr(
            extractor_mod.RequirementExtractor,
            "_llm_semantic_extract",
            lambda self, message, rule_slots, session_id="": {},
        )
        extractor_mod.RequirementExtractor._semantic_cache.clear()
        return sales_req

    def _turn(self, module, message, profile, last=""):
        if profile is not None and last:
            profile.last_asked_slot = last
        state = {
            "messages": [{"role": "user", "content": message}],
            "current_message": message,
            "session_id": "two-ask-limit",
            "requirements": {},
            "additional_requirements": [],
            "intent": "need_query",
            "next_action": "ask",
            "should_generate_solution": False,
            "response": "",
            "pending_question": "",
            "pending_slot": "",
        }
        if profile is not None:
            state["requirement_profile"] = profile
        return module.requirement_mining(state)

    def _profile(self, **extra):
        slots = {
            "display_type": "LED", "environment": "indoor", "purpose": "conference",
            "installation": "fixed", "price_preference": "price",
            "pixel_pitch_mm": 3,
            # 硬性条件：尺寸（P值/室内外/固装租赁上面都已给）
            "target_width_mm": 5000, "target_height_mm": 3000,
        }
        slots.update(extra)
        return RequirementProfile.from_slots(slots, explicit_keys=set(slots))

    def test_content_type_is_not_hard_and_never_blocks(self, sales_node):
        """客户口径：内容类型是非硬性项，客户不知道也不影响推荐。"""
        profile = self._profile(content_type=None)
        decision = check_recommendation_ready(profile)
        assert decision.ready is True
        assert "content_type" not in decision.missing

        # 硬性条件齐 → 这一轮直接推荐，不会再问内容类型
        turn = self._turn(sales_node, "hello", profile)
        assert turn["should_generate_solution"] is True
        assert turn["pending_slot"] != "content_type"

    def test_bare_both_answers_price_question_and_recommends(self, sales_node):
        """客户口径：问"最看重价格还是质量"，客户回 "both" 要直接按默认档进入推荐。

        实测 bug：bare "both" 没被识别成价格取向，系统又问了一遍同一个问题。
        """
        profile = self._profile(
            content_type="mixed", pixel_pitch_mm=3,
            price_preference=None,
        )
        turn = self._turn(sales_node, "hello", profile)
        # 硬性条件齐 → 这一轮就应该推荐（不会再去问价格取向）
        assert turn["should_generate_solution"] is True

        turn = self._turn(
            sales_node, "both", turn["requirement_profile"], last="price_preference"
        )
        assert turn["requirement_profile"].price_preference == "both"
        assert turn["requirement_profile"].budget_level == "low"
        assert turn["should_generate_solution"] is True

    def test_price_preference_never_blocks_recommendation(self, sales_node):
        """客户口径：价格取向不问也不阻塞；客户主动说了就记录（both → 默认档）。"""
        profile = self._profile(content_type="mixed", pixel_pitch_mm=3, price_preference=None)
        turn = self._turn(sales_node, "hello", profile)
        assert turn["recommendation_gate"]["ready"] is True
        assert turn["should_generate_solution"] is True
        assert turn["requirement_profile"].price_preference is None
