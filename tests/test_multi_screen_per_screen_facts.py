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
