"""多屏链路两个实测问题（2026-09-28 客户会话）。

问题 1（室内固装那块被推了租赁款 / 室外租赁那块没有型号）

    客户: permanent for indoor and rental for outdoor
    AI  : Screen 1 (indoor / other): … the quick-install rental design covers the
          outdoor rental while the same cabinet works permanently indoors …
          Screen 2 (outdoor / other): this one is still being confirmed, I'll
          come back with the model shortly.

    多屏是"每块屏各跑一次推荐"，但每块屏那一轮仍然把**客户整句**
    （"permanent for indoor and rental for outdoor"）交给 Solution。
    ``query_understanding.understand_query`` 把**句子级**槽位当成客户明说的事实
    （explicit）并覆盖这块屏自己的档案：

        extract_slots("permanent for indoor and rental for outdoor")
        = {'environment': 'indoor', 'purpose': 'rental', 'installation': 'rental'}

    → 室内固装那块变成 indoor + rental → TW11-IR-P3.9（租赁款）；
      室外租赁那块变成 indoor → 室内型号又被 ``_model_matches_screen`` 拦下
      → "still being confirmed"。

    修法：每块屏那一轮只把**这块屏自己的档案**拼成一句需求（``_screen_query_text``），
    并且 ``_model_matches_screen`` 除了环境也校验安装方式（fixed / rental）。

问题 2（推荐话术被句数上限截断）

    推荐链路的提示词里写死了长度上限（Solution 的 "(2-3 sentences max)" /
    "max 90 words"、NATIVE 的 "3 to 4 short sentences"）。多屏要"有几块屏就
    给几块屏"的完整推荐，这些上限会把内容砍掉。
"""
import os
import re
import sys
from types import SimpleNamespace

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _profile(**slots):
    from src.models.requirement import RequirementProfile

    # 与多屏链路一致：直接按字段名建档案，并把客户明说的字段标成 explicit
    # （`_split_and_apply_screen_specs` 就是这么写的）。
    profile = RequirementProfile.model_validate(slots)
    for field in slots:
        profile.sources[field] = "explicit"
    return profile


def _indoor_fixed_profile():
    return _profile(
        display_type="LED",
        environment="indoor",
        installation="fixed",
        pixel_pitch_mm=4.0,
        target_width_m=3.0,
        target_height_m=5.0,
        viewing_distance_m=5.0,
        purpose="church",
    )


def _outdoor_rental_profile():
    return _profile(
        display_type="LED",
        environment="outdoor",
        installation="rental",
        pixel_pitch_mm=4.0,
        target_width_m=3.0,
        target_height_m=5.0,
        viewing_distance_m=5.0,
        purpose="church",
    )


def _canonical_record(model: str):
    from src.config import config
    from src.rag.json_loader import canonical_model_index

    return canonical_model_index(config.DATA_DIR).get(model)


class TestRootCauseDocumented:
    """问题 1 的根因留痕：整句交给 understand_query 会把句子级槽位当客户明说的事实。"""

    def test_project_sentence_flips_the_screen_profile(self):
        from src.rag.query_understanding import understand_query

        understanding = understand_query(
            "permanent for indoor and rental for outdoor",
            history=[],
            profile=_indoor_fixed_profile(),
        )

        assert understanding.profile.installation == "rental", (
            "（根因）室内固装这块档案被整句里的 rental 覆盖 —— 所以不能把整句交给单块屏的推荐"
        )


class TestScreenQueryTextKeepsOwnFacts:
    """修法：单块屏只拿"这一块屏自己的需求句"去推荐。"""

    def test_indoor_fixed_screen_text_keeps_indoor_fixed(self):
        from src.rag.multi_screen import MultiScreenManager
        from src.rag.query_understanding import extract_slots

        text = MultiScreenManager._screen_query_text(_indoor_fixed_profile().model_dump())
        slots = extract_slots(text)

        assert slots.get("environment") == "indoor", slots
        assert slots.get("installation") == "fixed", slots
        assert slots.get("pixel_pitch_mm") == 4.0, slots
        assert slots.get("target_width_mm") == 3000.0, slots
        assert slots.get("target_height_mm") == 5000.0, slots

    def test_outdoor_rental_screen_text_keeps_outdoor_rental(self):
        from src.rag.multi_screen import MultiScreenManager
        from src.rag.query_understanding import extract_slots

        text = MultiScreenManager._screen_query_text(_outdoor_rental_profile().model_dump())
        slots = extract_slots(text)

        assert slots.get("environment") == "outdoor", slots
        assert slots.get("installation") == "rental", slots

    def test_hard_constraints_follow_the_screen_not_the_project_sentence(self):
        from src.rag.hard_filter import build_hard_constraints
        from src.rag.multi_screen import MultiScreenManager
        from src.rag.query_understanding import understand_query

        history = [{"role": "user", "content": "permanent for indoor and rental for outdoor"}]
        for profile, environment, installation in (
            (_indoor_fixed_profile(), "indoor", "fixed"),
            (_outdoor_rental_profile(), "outdoor", "rental"),
        ):
            text = MultiScreenManager._screen_query_text(profile.model_dump())
            understanding = understand_query(text, history=history, profile=profile)
            constraints = build_hard_constraints(understanding.profile, message=text)

            assert constraints.environment == environment, (text, constraints.describe())
            assert constraints.installation == installation, (text, constraints.describe())


class TestModelMatchesScreenChecksInstallation:
    """型号校验不能只看环境：室内固装不能推租赁款，室外租赁不能推固装款。"""

    def test_indoor_fixed_rejects_a_rental_model(self):
        from src.rag.multi_screen import MultiScreenManager

        profile = {"environment": "indoor", "installation": "fixed"}
        assert MultiScreenManager._model_matches_screen("TW11-3216-P4.0", profile) is True
        assert MultiScreenManager._model_matches_screen("TW11-IR-P3.9", profile) is False

    def test_outdoor_rental_rejects_a_fixed_model(self):
        from src.rag.multi_screen import MultiScreenManager

        profile = {"environment": "outdoor", "installation": "rental"}
        assert MultiScreenManager._model_matches_screen("TW11-OR-P3.9", profile) is True
        assert MultiScreenManager._model_matches_screen("TW11-OD-P4", profile) is False

    def test_unknown_installation_keeps_the_environment_only_check(self):
        from src.rag.multi_screen import MultiScreenManager

        assert MultiScreenManager._model_matches_screen(
            "TW11-IR-P3.9", {"environment": "indoor"}
        ) is True


class _CanonicalStubSolution:
    """把真实链路里的 LLM 表达换成"按硬约束从产品索引里挑一个型号"。

    只保留**硬约束**这条关键路径 —— 问题 1 就出在这里：整句 message 覆盖了
    这块屏的档案，硬约束跟着错，选出来的型号自然错。
    """

    def __init__(self) -> None:
        self.seen_messages: list = []
        self.models: list = []

    def run(self, message="", history=None, session_id="", profile=None, intent="", **_kwargs):
        from src.rag.hard_filter import build_hard_constraints
        from src.rag.query_understanding import understand_query

        self.seen_messages.append(message)
        understanding = understand_query(message, history=history or [], profile=profile)
        constraints = build_hard_constraints(understanding.profile, message=message)
        model = self._pick(constraints)
        if not model:
            return {"answer": "", "products": [], "route": "agent"}
        self.models.append(model)
        return {
            "answer": f"For this screen I'd go with {model}, it matches the brief.",
            "products": [{"model": model}],
            "route": "agent",
        }

    @staticmethod
    def _pick(constraints) -> str:
        from src.config import config
        from src.rag.json_loader import canonical_model_index

        where = constraints.chroma_where()
        pitch_min = constraints.pixel_pitch_min
        pitch_max = constraints.pixel_pitch_max
        best = None
        for name, record in canonical_model_index(config.DATA_DIR).items():
            if where.get("display_type") and getattr(record, "display_type", None) != where["display_type"]:
                continue
            if where.get("indoor") is True and not getattr(record, "indoor", False):
                continue
            if where.get("outdoor") is True and not getattr(record, "outdoor", False):
                continue
            want_rental = where.get("is_rental")
            is_rental = getattr(record, "installation", "") == "rental"
            if want_rental is not None and is_rental != bool(want_rental):
                continue
            pitch = getattr(record, "pixel_pitch_mm", None)
            if pitch is None:
                continue
            if pitch_min is not None and pitch < pitch_min:
                continue
            if pitch_max is not None and pitch > pitch_max:
                continue
            target = pitch_min if pitch_min is not None else pitch
            score = abs(pitch - target)
            if best is None or score < best[0]:
                best = (score, name)
        return best[1] if best else ""


class TestMultiScreenRecommendationPerScreen:
    """问题 1 端到端：室内固装 → 固装型号；室外租赁 → 户外租赁型号。"""

    def _run(self, session_id: str = "multi-screen-install-mix"):
        from src.memory.store import memory
        from src.models.requirement import RequirementProfile
        from src.rag.multi_screen import MultiScreenManager

        memory.clear(session_id)
        indoor = _indoor_fixed_profile()
        outdoor = _outdoor_rental_profile()
        memory.set_project_items(
            session_id,
            [{"profile": indoor.model_dump()}, {"profile": outdoor.model_dump()}],
        )
        memory.set_active_item_index(session_id, 1)
        memory.set_requirement_profile(session_id, outdoor)

        stub = _CanonicalStubSolution()
        manager = MultiScreenManager(
            store=memory,
            profile_lookup=lambda sid: RequirementProfile.model_validate(
                memory.get_requirement_profile(sid)
            ),
            history_lookup=lambda sid: [
                {"role": "user", "content": "permanent for indoor and rental for outdoor"}
            ],
            solution_agent=stub,
        )
        reply = manager._recommend_all_screens(
            session_id, "permanent for indoor and rental for outdoor"
        )
        return stub, reply

    def test_both_screens_get_a_model(self):
        stub, reply = self._run()

        assert reply, "多屏回复不能为空"
        assert len(stub.models) == 2, stub.models
        assert "still being confirmed" not in reply, reply

    def test_indoor_screen_gets_a_fixed_model(self):
        stub, reply = self._run("multi-screen-install-mix-indoor")

        first = _canonical_record(stub.models[0])
        assert getattr(first, "indoor", False) is True, stub.models[0]
        assert getattr(first, "installation", "") == "fixed", stub.models[0]
        assert reply.split("\n\n")[0].find(stub.models[0]) >= 0, reply

    def test_outdoor_screen_gets_a_rental_model(self):
        stub, reply = self._run("multi-screen-install-mix-outdoor")

        second = _canonical_record(stub.models[1])
        assert getattr(second, "outdoor", False) is True, stub.models[1]
        assert getattr(second, "installation", "") == "rental", stub.models[1]

    def test_each_screen_only_sees_its_own_facts(self):
        stub, _reply = self._run("multi-screen-install-mix-messages")

        first_message, second_message = stub.seen_messages[0], stub.seen_messages[1]
        assert "rental" not in first_message.lower(), first_message
        assert "fixed" in first_message.lower(), first_message
        assert "outdoor" in second_message.lower(), second_message
        assert "rental" in second_message.lower(), second_message


class TestRecommendationLengthLimitLifted:
    """问题 2：推荐轮不该被句数 / 字数上限砍掉内容。"""

    def test_recommend_prompt_has_no_sentence_cap(self):
        from src.agents.solution.nodes.recommend import RECOMMEND_PROMPT

        assert "2-3 sentences" not in RECOMMEND_PROMPT
        assert "complete" in RECOMMEND_PROMPT.lower()

    def test_recommendation_prompt_has_no_word_cap(self, monkeypatch):
        import src.agents.solution.nodes.recommend as recommend_mod

        captured: dict = {}

        class _LLM:
            def invoke(self, prompt):
                captured["prompt"] = prompt
                return SimpleNamespace(content="TW11-3216-P4.0 is the closest match.")

        monkeypatch.setattr(recommend_mod, "get_llm", lambda **kwargs: _LLM())
        recommend_mod._express_recommendation(
            recommendations=[
                {
                    "model": "TW11-3216-P4.0",
                    "series_id": "TW11-3216",
                    "pixel_pitch_mm": 4.0,
                    "brightness_nit": 500,
                    "cabinet_size_mm": "640x480",
                    "modules_per_cabinet": 6,
                    "reasons": ["4mm pitch suits a 5m viewing distance"],
                }
            ],
            profile=None,
            calculation=None,
            additional_requirements=[],
            customer_text="permanent for indoor and rental for outdoor",
        )

        prompt = captured["prompt"]
        assert "max 90 words" not in prompt
        assert "2-3 sentences" not in prompt

    def test_native_prompt_drops_the_sentence_cap_when_delivering(self):
        from src.dialogue.response_context import ResponseContext
        from src.dialogue.response_generator import native_length_rule

        delivering = ResponseContext(action="RECOMMEND")
        asking = ResponseContext(action="ASK")

        delivery_rule = native_length_rule(delivering).lower()
        ask_rule = native_length_rule(asking).lower()

        assert "3 to 4 short sentences" not in delivery_rule
        assert "no sentence limit" in delivery_rule
        assert "3 to 4 short sentences" in ask_rule

    def test_native_prompt_uses_the_shape_specific_rule(self, monkeypatch):
        import src.dialogue.response_generator as generator

        captured: dict = {}

        class _LLM:
            def invoke(self, prompt):
                captured["prompt"] = prompt
                return SimpleNamespace(content="TW11-3216-P4.0 is the closest match.")

        generator._generate_native(ResponseContextProxy("RECOMMEND"), _LLM())
        assert "no sentence limit" in captured["prompt"].lower()

    def test_recommendation_answer_is_not_cut_mid_sentence(self):
        from src.agents.solution.runner import cap_answer_length

        first = (
            "Screen 1 (indoor / church): TW11-3216-P4.0 is the closest match, "
            "with 4.0mm pixel pitch. " + "It suits the 5m viewing distance well. " * 40
        )
        second = (
            "Screen 2 (outdoor / church): TW11-OR-P3.9 is the closest match. "
            + "The rental cabinet keeps the install quick. " * 40
        )
        full = first + second
        assert len(full) > 2000, "用例要真的超过上限才有意义"
        capped = cap_answer_length(full)

        assert capped.startswith("Screen 1 (indoor / church): TW11-3216-P4.0"), capped[:80]
        assert first.strip() in capped, "第一块屏的完整内容不能被砍"
        assert capped.rstrip().endswith((".", "!", "?")), capped[-60:]
        assert not capped.rstrip().endswith("..."), capped[-60:]

    def test_short_recommendation_is_untouched(self):
        from src.agents.solution.runner import cap_answer_length

        text = "TW11-OR-P3.9 is the closest match for your outdoor rental screen."
        assert cap_answer_length(text) == text


def ResponseContextProxy(action: str):
    """最小可用的 ResponseContext（只关心 response_shape / 语言规则）。"""
    from src.dialogue.response_context import ResponseContext

    return ResponseContext(action=action, customer_message="permanent for indoor and rental for outdoor")


class TestMultiScreenCompositionIsNotTemplated:
    """客户实测（2026-09-28 第二次）：两块屏各打一次招呼、两块各问一遍、读起来一个模板。

        客户可见原文：
        Screen 1 (indoor / stage): Hi Jack, thanks for reaching out. For your indoor
            3m x 5m screen we recommend the TW11-3216-P3.0. … Shall I go ahead and
            prepare the quotation for the TW11-3216-P3.0 indoor screen …?
        Screen 2 (outdoor / stage): Hi Jack, thanks for reaching out. For your indoor
            3x5 screen … please share the exact dimensions … and I'll prepare the
            full quotation for both screens.

    → 招呼必须只出现一次；整条回复只能留一个"要客户做的事"（问句或请求），
      并且要落在整条回复的最后；第二块不能再重复第一块的招呼。
    """

    def _blocks(self):
        indoor = _indoor_fixed_profile().model_dump()
        outdoor = _outdoor_rental_profile().model_dump()
        first = (
            "Hi Jack, thanks for reaching out. For your indoor 3m x 5m screen we "
            "recommend the TW11-3216-P3.0, with a 3.076mm pixel pitch that suits the "
            "viewing distance. Horizontal tiling: 5 x 11 = 55 cabinets. "
            "Shall I go ahead and prepare the quotation for the indoor screen?"
        )
        second = (
            "Hi Jack, thanks for reaching out. For your outdoor 5m x 10m screen the "
            "TW11-OR-P3.9 is the closest match, with a 3.91mm pixel pitch and a "
            "waterproof rental cabinet. Horizontal tiling: 10 x 20 = 200 cabinets. "
            "For the outdoor screen, please share the exact dimensions so I can "
            "confirm the layout and prepare the full quotation."
        )
        return [(0, indoor, first), (1, outdoor, second)]

    def _reply(self, language: str = "en") -> str:
        from src.rag.multi_screen import MultiScreenManager

        return MultiScreenManager._compose_screen_blocks(self._blocks(), language)

    def test_greeting_appears_only_once(self):
        reply = self._reply()

        assert reply.lower().count("hi jack") == 1, reply
        assert reply.lower().count("thanks for reaching out") == 1, reply

    def test_second_block_does_not_open_with_the_same_opener(self):
        reply = self._reply()

        second = reply.split("\n\n")[1]
        body = second.split(":", 1)[1].strip().lower()
        assert not body.startswith(("hi", "hello", "hey", "thanks", "thank you")), body[:60]

    def test_only_one_ask_survives_and_it_closes_the_reply(self):
        reply = self._reply()

        questions = reply.count("?") + reply.count("？")
        requests = len(re.findall(r"\bplease\b|\bshall i\b|\bcould you\b", reply, re.IGNORECASE))
        assert questions + requests == 1, reply
        assert reply.rstrip().endswith((".", "?", "!", "。", "？", "！")), reply[-80:]
        assert "please share" in reply, "保留下来的应该是最后那一个收尾请求"

    def test_each_block_keeps_its_own_model_and_facts(self):
        reply = self._reply()
        first, second = reply.split("\n\n")

        assert "TW11-3216-P3.0" in first and "5 x 11 = 55" in first, first
        assert "TW11-OR-P3.9" in second and "10 x 20 = 200" in second, second
        assert "TW11-3216-P3.0" not in second, second

    def test_single_block_keeps_its_own_greeting(self):
        from src.rag.multi_screen import MultiScreenManager

        one = self._blocks()[:1]
        reply = MultiScreenManager._compose_screen_blocks(one, "en")

        assert reply.lower().count("hi jack") == 1, reply


class TestScreenBriefTellsTheModelWhichScreenThisIs:
    """第二块不能再借用第一块的场景/环境（实测：室外那块写成 "your indoor 3x5 screen"）。"""

    def test_brief_names_this_screen_and_forbids_the_other(self):
        from src.rag.multi_screen import MultiScreenManager

        brief = MultiScreenManager._screen_brief(
            1, _outdoor_rental_profile().model_dump(), 2
        )
        lowered = brief.lower()

        assert "screen 2 of 2" in lowered, brief
        assert "outdoor" in lowered, brief
        assert "indoor" in lowered, "要点明这块屏是室外的（不要写成室内、不要提另一块屏）: " + brief
        assert "greet" in lowered, brief

    def test_per_screen_recommend_passes_the_brief(self):
        from src.memory.store import memory
        from src.models.requirement import RequirementProfile
        from src.rag.multi_screen import MultiScreenManager

        session_id = "multi-screen-brief"
        memory.clear(session_id)
        indoor = _indoor_fixed_profile()
        outdoor = _outdoor_rental_profile()
        memory.set_project_items(
            session_id,
            [{"profile": indoor.model_dump()}, {"profile": outdoor.model_dump()}],
        )
        memory.set_active_item_index(session_id, 1)
        memory.set_requirement_profile(session_id, outdoor)

        seen: list = []

        class _Stub:
            def run(self, message="", history=None, session_id="", profile=None, **_kwargs):
                seen.append(_kwargs.get("multi_screen_brief", ""))
                return {
                    "answer": f"Hi Jack, {message}",
                    "products": [{"model": "TW11-3216-P4.0" if len(seen) == 1 else "TW11-OR-P3.9"}],
                    "route": "agent",
                }

        manager = MultiScreenManager(
            store=memory,
            profile_lookup=lambda sid: RequirementProfile.model_validate(
                memory.get_requirement_profile(sid)
            ),
            history_lookup=lambda sid: [],
            solution_agent=_Stub(),
        )
        manager._recommend_all_screens(session_id, "permanent for indoor and rental for outdoor")

        assert len(seen) == 2, seen
        assert "screen 1 of 2" in seen[0].lower(), seen[0]
        assert "screen 2 of 2" in seen[1].lower(), seen[1]


class TestPhantomThirdScreen:
    """客户实测（2026-09-28 第三次）：客户只要两块屏，回复里多出一块 "Screen 3"。

        客户: i change my mind, i need two screens   ← 只是说"要两块屏"
        客户: i need a outdoor and indoor / p4 for indoor and p5 for outdoor / …
        AI  : Screen 1 (indoor) … Screen 2 (outdoor) … Screen 3: TW31-COB-P0.7H …

    根因：`detect_new_item` 把 "two screens" 当成"另一块屏"的信号，而当时当前这块
    还没有任何屏体需求（没有环境/用途/尺寸/点间距）—— 归档出一块空需求，
    后来被 `_share_common_facts` 填上安装方式/点间距（共有项不含环境），于是变成
    客户看到的第三块屏（标签连环境都没有）。
    """

    def test_screen_facts_definition(self):
        from src.rag.multi_screen import MultiScreenManager

        assert MultiScreenManager._describes_a_screen({"environment": "indoor"}) is True
        assert MultiScreenManager._describes_a_screen({"target_width_m": 3.0}) is True
        assert MultiScreenManager._describes_a_screen({"pixel_pitch_mm": 4.0}) is True
        assert MultiScreenManager._describes_a_screen(
            {"display_type": "LED", "installation": "fixed", "pixel_pitch_mm": None}
        ) is False
        assert MultiScreenManager._describes_a_screen({}) is False

    def test_two_screens_intent_does_not_open_a_new_item(self):
        from src.memory.store import memory
        from src.models.requirement import RequirementProfile
        from src.rag.multi_screen import MultiScreenManager

        session_id = "phantom-two-screens"
        memory.clear(session_id)
        # 当前这块只有"共有项"（客户还没说环境/尺寸）—— 正是实测踩到的那种状态
        profile = RequirementProfile.model_validate(
            {"display_type": "LED", "has_recommendation": False}
        )
        memory.set_requirement_profile(session_id, profile)
        memory.set_project_items(session_id, [{"profile": profile.model_dump()}])
        memory.set_active_item_index(session_id, 0)

        manager = MultiScreenManager(
            store=memory,
            profile_lookup=lambda sid: RequirementProfile.model_validate(
                memory.get_requirement_profile(sid)
            ),
            solution_agent=None,
        )
        reason = manager._maybe_start_new_item(
            session_id, "i change my mind, i need two screens"
        )

        assert reason == "", "客户只是说'要两块屏'，不该新开条目"
        assert len(memory.get_project_items(session_id)) == 1
        assert memory.get_active_item_index(session_id) == 0

    def test_quantity_statement_is_not_a_new_item_even_with_a_full_profile(self):
        """整块档案都齐了，客户说"i need two screens"仍然只是在报数量。"""
        from src.rag.project_items import detect_new_item

        for message in (
            "i change my mind, i need two screens",
            "we need 2 screens",
            "我要两块屏",
        ):
            assert detect_new_item(message, _indoor_fixed_profile()) == (False, ""), message

    def test_a_second_screen_statement_still_opens_a_new_item(self):
        from src.rag.project_items import detect_new_item

        should_start, reason = detect_new_item(
            "another screen for the entrance", _indoor_fixed_profile()
        )
        assert should_start is True and reason, (should_start, reason)

    def test_new_item_still_opens_once_a_real_screen_exists(self):
        from src.memory.store import memory
        from src.models.requirement import RequirementProfile
        from src.rag.multi_screen import MultiScreenManager

        session_id = "phantom-real-second-screen"
        memory.clear(session_id)
        profile = _indoor_fixed_profile()
        memory.set_requirement_profile(session_id, profile)
        memory.set_project_items(session_id, [{"profile": profile.model_dump()}])
        memory.set_active_item_index(session_id, 0)

        manager = MultiScreenManager(
            store=memory,
            profile_lookup=lambda sid: RequirementProfile.model_validate(
                memory.get_requirement_profile(sid)
            ),
            solution_agent=None,
        )
        reason = manager._maybe_start_new_item(
            session_id, "another screen for the entrance"
        )

        assert reason, "第一块屏已经描述清楚了 → 客户说'再来一块'要能开新条目"
        assert len(memory.get_project_items(session_id)) == 2

    def test_recommend_skips_an_item_with_no_screen_facts(self):
        from src.memory.store import memory
        from src.models.requirement import RequirementProfile
        from src.rag.multi_screen import MultiScreenManager

        session_id = "phantom-skip"
        memory.clear(session_id)
        indoor = _indoor_fixed_profile()
        outdoor = _outdoor_rental_profile()
        phantom = {
            "display_type": "LED",
            "installation": "fixed",
            "sources": {"display_type": "default", "installation": "explicit"},
        }
        memory.set_project_items(
            session_id,
            [
                {"profile": indoor.model_dump()},
                {"profile": outdoor.model_dump()},
                {"profile": phantom},
            ],
        )
        memory.set_active_item_index(session_id, 2)
        memory.set_requirement_profile(session_id, phantom)

        stub = _CanonicalStubSolution()
        manager = MultiScreenManager(
            store=memory,
            profile_lookup=lambda sid: RequirementProfile.model_validate(
                memory.get_requirement_profile(sid)
            ),
            history_lookup=lambda sid: [],
            solution_agent=stub,
        )
        reply = manager._recommend_all_screens(session_id, "can u recommend to me")

        assert reply, reply
        assert len(stub.models) == 2, stub.models
        assert "Screen 3" not in reply, reply
        assert "0.78" not in reply, reply


class TestMultiScreenReplyStaysShort:
    """客户口径（2026-09-28 第三次）：两块屏的推荐要短，但不能短到写不完。"""

    def test_brief_sets_a_word_budget_and_format(self):
        from src.rag.multi_screen import MultiScreenManager

        brief = MultiScreenManager._screen_brief(
            0, _indoor_fixed_profile().model_dump(), 2
        ).lower()

        assert "50-70 words" in brief, brief
        assert "no closing commentary" in brief, brief
        assert "tiling options" in brief, brief

    def test_composer_drops_fact_free_closing_commentary(self):
        from src.rag.multi_screen import MultiScreenManager

        indoor = _indoor_fixed_profile().model_dump()
        outdoor = _outdoor_rental_profile().model_dump()
        first = (
            "For your indoor wall the TW11-3216-P4.0 is the closest fit. "
            "Horizontal: 5 x 11 = 55 cabinets, actual 3.2m x 5.28m, 330 modules. "
            "Both options fit your 3m x 5m indoor wall, so you can choose based on "
            "which proportions work better for your installation."
        )
        second = (
            "For your outdoor wall the TW11-OR-P3.9 is the closest fit. "
            "Horizontal: 10 x 20 = 200 cabinets, actual 5.0m x 10.0m, 800 modules. "
            "Both layouts work for this wall, so the choice comes down to taste."
        )
        reply = MultiScreenManager._compose_screen_blocks(
            [(0, indoor, first), (1, outdoor, second)], "en"
        )

        assert "Both options fit" not in reply, reply
        assert "Both layouts work" not in reply, reply
        assert "5 x 11 = 55 cabinets" in reply, reply
        assert "10 x 20 = 200 cabinets" in reply, reply

    def test_filler_trim_never_empties_a_block(self):
        from src.rag.multi_screen import MultiScreenManager

        indoor = _indoor_fixed_profile().model_dump()
        only_filler = "Both options work well for this wall, so you can choose either one."
        reply = MultiScreenManager._compose_screen_blocks(
            [(0, indoor, only_filler)], "en"
        )

        assert "Both options work well" in reply, "整段只有这一句时不能删空"


class TestInstallationTypoTolerance:
    """实测笔误："fixed for indoor and reantal for outdoor" —— 室外那块要租赁。"""

    def test_reantal_is_read_as_rental(self):
        from src.rag.query_understanding import extract_slots

        slots = extract_slots("fixed for indoor and reantal for outdoor")
        assert slots.get("installation") == "rental", slots

    def test_splitter_assigns_rental_to_the_outdoor_screen(self):
        from src.rag.project_items import split_multi_screen_specs

        specs = split_multi_screen_specs("fixed for indoor and reantal for outdoor")
        by_env = {spec["environment"]: spec.get("installation") for spec in specs}
        assert by_env == {"indoor": "fixed", "outdoor": "rental"}, specs
