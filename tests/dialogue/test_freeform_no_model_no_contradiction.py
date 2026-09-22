"""客户口径（2026-09-21）：自由问答 / 异议回答不许"乱说型号、乱说环境"。

实测 bug：客户是「室内 / 教堂 / 固装」，问"不能10天到吗"，AI 却回答
    "…an outdoor fixed cabinet setup like this. The TW21-OD-P10 needs production…"
——型号是检索到的（产品语料里有 TW21-OD-P10 这个户外型号），
而这条路径既没有客户需求上下文，也没有硬约束。
"""
import os
import sys
from types import SimpleNamespace

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.delivery_info import (  # noqa: E402
    delivery_answer,
    is_delivery_question,
    requested_window_days,
    wants_faster_delivery,
)
from src.rag.model_guard import (  # noqa: E402
    chunk_conflicts,
    drop_conflicting_chunks,
    requirement_lines,
    retrieval_filters,
    strip_environment_contradictions,
    strip_model_mentions,
)

CHURCH = RequirementProfile.from_slots(
    {
        "display_type": "LED",
        "environment": "indoor",
        "purpose": "church",
        "installation": "fixed",
        "target_width_mm": 3000,
        "target_height_mm": 5000,
        "viewing_distance_m": 5.0,
    },
    explicit_keys={"display_type", "environment", "purpose"},
)


class TestExpediteQuestions:

    def test_short_window_counts_as_a_delivery_question(self):
        assert is_delivery_question("不能10天到吗") is True
        assert wants_faster_delivery("不能10天到吗") is True
        assert requested_window_days("不能10天到吗") == 10

    def test_normal_window_is_not_an_expedite_request(self):
        assert wants_faster_delivery("3周内能发货吗") is False
        assert requested_window_days("3周内能发货吗") == 21

    def test_answer_covers_lead_time_and_air_freight(self):
        answer = delivery_answer("不能10天到吗", language="en", seed=0) or ""
        assert "15" in answer and "30" in answer
        assert "air" in answer.lower(), answer   # 加急口径：可空运、成本增加


class TestModelGuard:

    OUTDOOR_CHUNK = {
        "text": "Model TW21-OD-P10, pixel pitch 10mm, 6500nit, outdoor cabinet",
        "metadata": {"outdoor": True, "indoor": False},
    }
    INDOOR_CHUNK = {
        "text": "Model TW11-3216-P3.0, 1200nit, indoor cabinet, 640x480mm",
        "metadata": {"indoor": True, "outdoor": False, "is_rental": False},
    }

    def test_confirmed_requirements_are_rendered(self):
        lines = "\n".join(requirement_lines(CHURCH))
        assert "indoor" in lines and "fixed" in lines and "church" in lines

    def test_retrieval_filters_follow_requirements(self):
        filters = retrieval_filters(CHURCH)
        assert filters.get("indoor") is True
        assert filters.get("is_rental") is False

    def test_outdoor_chunk_is_dropped_for_an_indoor_customer(self):
        assert chunk_conflicts(self.OUTDOOR_CHUNK, CHURCH) is True
        kept, dropped = drop_conflicting_chunks(
            [self.OUTDOOR_CHUNK, self.INDOOR_CHUNK], CHURCH
        )
        assert dropped == 1
        assert kept == [self.INDOOR_CHUNK]

    def test_model_sentence_is_removed(self):
        text = (
            "That is tight for your setup. "
            "The TW21-OD-P10 needs production plus waterproofing and testing. "
            "We can also ship by air if you need it sooner."
        )
        cleaned, removed = strip_model_mentions(text)
        assert removed == ["TW21-OD-P10"]
        assert "TW21-OD-P10" not in cleaned
        assert "ship by air" in cleaned

    def test_opposite_environment_claim_is_removed(self):
        text = (
            "That is really tight for an outdoor fixed cabinet setup like this. "
            "Screens indoors usually need less sealing."
        )
        cleaned, removed = strip_environment_contradictions(text, CHURCH)
        assert removed, "与室内需求矛盾的 outdoor 配置句必须删掉"
        assert "outdoor fixed cabinet setup" not in cleaned


class TestObjectionPathUsesRequirements:

    class _StubSearch:
        def __init__(self):
            self.calls = []

        def similarity_search(self, query, k=2, filter=None):
            self.calls.append({"query": query, "k": k, "filter": filter})
            return [
                SimpleNamespace(
                    page_content="Model TW21-OD-P10 outdoor cabinet 6500nit"
                ),
                SimpleNamespace(
                    page_content="Model TW11-3216-P3.0 indoor cabinet 1200nit"
                ),
            ]

    def test_prompt_carries_requirements_and_answer_has_no_model(self, monkeypatch):
        import importlib

        module = importlib.import_module("src.agents.sales.nodes.script_generator")
        captured = {}

        class _FakeLLM:
            def invoke(self, prompt):
                captured["prompt"] = "\n".join(
                    str(getattr(item, "content", item)) for item in prompt
                )
                return SimpleNamespace(
                    content=(
                        "That is tight for an outdoor fixed cabinet setup like this. "
                        "The TW21-OD-P10 needs production plus waterproofing and testing."
                    )
                )

        monkeypatch.setattr(module, "get_llm", lambda **_kwargs: _FakeLLM())
        search = self._StubSearch()
        state = {
            "current_message": "不能10天到吗",
            "messages": [{"role": "user", "content": "不能10天到吗"}],
            "session_id": "objection-guard",
            "requirements": {},
            "requirement_profile": CHURCH,
            "sales_search": search,
            "intent": "objection",
            "next_action": "ask",
            "should_generate_solution": False,
            "response": "",
            "pending_question": "",
            "pending_slot": "",
        }
        text = module._answer_objection_like(state, "不能10天到吗")

        # ① prompt 里必须带上客户已确认的需求，并明确"不许提型号/不许矛盾"
        prompt = captured["prompt"]
        assert "indoor" in prompt and "church" in prompt
        assert "Never mention any product model code" in prompt
        assert "must NOT be contradicted" in prompt
        # ② 检索已经按需求过滤（室内 + 固装）
        assert search.calls and search.calls[0]["filter"].get("indoor") is True
        assert search.calls[0]["filter"].get("is_rental") is False
        # ③ 最终答复里既没有型号，也没有"outdoor 配置"这种矛盾说法
        assert "TW21-OD-P10" not in text
        assert "outdoor fixed cabinet setup" not in text
