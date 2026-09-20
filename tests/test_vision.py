"""
智谱视觉需求提取单元测试（计划第三章~第十一章 / 第十九~二十一章）。

覆盖：
  1. 模型输出的容错解析（```json 代码块 / 前后废话 / 坏 JSON）
  2. 单位标准化（mm / cm / ft / inch → 米；点间距 → 毫米）
  3. explicit / inferred 区分（裸值一律按 inferred，绝不当成"看到的"）
  4. 图片缓存（同一 session + 同一图片只调一次；不同 session 不共享）
  5. 合并进 RequirementProfile：来源优先级 / 冲突 / 尺寸只当提示
  6. Gate 行为：有图片 ≠ 可推荐；视觉尺寸不进入工程计算
  7. 失败降级：视觉链路出错不影响主流程
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.readiness import (  # noqa: E402
    check_calculation_ready,
    check_recommendation_ready,
)
from src.vision.client import VisionError, check_image_payload  # noqa: E402
from src.vision.extractor import (  # noqa: E402
    VisionExtractor,
    merge_vision_results,
    parse_meters,
    parse_mm,
    parse_nit,
    parse_vision_json,
)
from src.vision.integration import (  # noqa: E402
    apply_vision_to_profile,
    extract_vision_for_turn,
)
from src.vision.schema import VISION_EXPLICIT, VISION_INFERRED, VisionRequirement  # noqa: E402


@pytest.fixture(autouse=True)
def clear_vision_cache():
    VisionExtractor._cache.clear()
    yield
    VisionExtractor._cache.clear()


class _FakeVisionClient:
    """假客户端：只返回预设文本，用来测解析与缓存逻辑。"""

    def __init__(self, payload: str, model: str = "glm-4v-plus"):
        self.payload = payload
        self.model = model
        self.calls = 0

    def analyze_image(self, image, prompt, system_prompt=None, mime_type=""):
        self.calls += 1
        return self.payload


INDOOR_ROOM_JSON = """
```json
{
  "display_type": {"value": "LED", "confidence": 0.95, "source": "vision_explicit", "evidence": "large LED video wall"},
  "environment": {"value": "indoor", "confidence": 0.9, "source": "vision_explicit", "evidence": "ceiling, carpet, chairs"},
  "purpose": {"value": "conference", "confidence": 0.8, "source": "vision_explicit", "evidence": "conference table"},
  "installation": null,
  "target_width_m": {"value": "5 m", "confidence": 0.4, "source": "vision_inferred", "evidence": "screen spans about 1/4 of the wall"},
  "target_height_m": {"value": "3000 mm", "confidence": 0.4, "source": "vision_inferred"},
  "viewing_distance_m": null,
  "pixel_pitch_mm": null,
  "brightness_min_nit": null,
  "brightness_max_nit": null,
  "special_requirements": [],
  "notes": "indoor meeting room with an LED wall"
}
```
"""


class TestInstallationPrompt:
    """客户口径（2026-09-18）：图片识别还要判断"固定安装（长期不动）还是租赁（快拆快装）"。"""

    def test_prompt_covers_fixed_vs_rental_cues(self):
        from src.vision.prompts import VISION_SYSTEM_PROMPT

        text = VISION_SYSTEM_PROMPT
        assert "installation" in text
        # 租赁（快拆快装）的判断线索
        for token in ("rental", "快锁", "桁架", "航空箱"):
            assert token in text, token
        # 固定安装（长期不动）的判断线索
        for token in ("fixed", "嵌入墙体", "钢结构"):
            assert token in text, token
        # 看不出来必须 null / 只能写 inferred（后面还会跟客户核对）
        assert "vision_inferred" in text
        assert "null" in text

    def test_user_prompt_also_reminds_installation(self):
        """实测：模型最容易漏 installation（倾向直接 null）→ 用户提示词末尾再提醒一次。"""
        from src.vision.prompts import build_user_prompt

        prompt = build_user_prompt("看看这个屏")
        assert "installation" in prompt
        assert "rental" in prompt and "fixed" in prompt
        assert "null" in prompt
        # 没带文字时也要带上提醒
        assert "installation" in build_user_prompt("")


class TestInstallationRecognition:
    """实测反馈：图片识别不给"永久固定安装 / 快装快拆租赁"。

    两层原因都修了：
      1. 提示词把 installation 明确成"必须判断"，并给了两边的判断线索；
      2. 解析层的线索表补上了"快装/快拆/快锁/桁架/航空箱/钢结构…"，
         且英文按整词匹配（避免 "installation" 里的 "install" 误判成 fixed）。
    """

    @pytest.mark.parametrize("value", [
        "rental", "temporarily installed", "quick-lock cabinet", "quick release panels",
        "mounted on truss", "flight case next to it", "portable event panel",
        "快装快拆", "快锁箱体", "桁架吊挂", "旁边有航空箱", "临时搭建",
    ])
    def test_rental_cues(self, value):
        result = VisionExtractor.from_payload({"installation": value})
        assert result.installation is not None, value
        assert result.installation.value == "rental", value

    @pytest.mark.parametrize("value", [
        "fixed", "permanent", "fixed installation", "permanently installed",
        "embedded in the wall", "flush mounted in wall", "steel structure below the screen",
        "wall mounted bracket", "永久固定安装", "固定安装", "嵌入墙体", "钢结构立柱",
    ])
    def test_fixed_cues(self, value):
        result = VisionExtractor.from_payload({"installation": value})
        assert result.installation is not None, value
        assert result.installation.value == "fixed", value

    def test_install_word_alone_does_not_mean_fixed(self):
        """"installation: rental" 不能被 "install" 抢成 fixed（最长键 + 整词匹配）。"""
        result = VisionExtractor.from_payload(
            {"installation": {"value": "installation: rental", "source": "vision_explicit"}}
        )
        assert result.installation.value == "rental"
        assert VisionExtractor.from_payload({"installation": "installation"}).installation is None

    def test_evidence_is_used_when_value_is_the_cue_text(self):
        """模型把线索原文当 value 返回时，仍然要能落成 fixed/rental。"""
        result = VisionExtractor.from_payload({
            "installation": {
                "value": "steel structure below the screen",
                "confidence": 0.6,
                "source": "vision_inferred",
                "evidence": "steel structure below the screen",
            }
        })
        assert result.installation.value == "fixed"
        assert result.installation.source == "vision_inferred"

    def test_no_cue_stays_null(self):
        assert VisionExtractor.from_payload({"installation": None}).installation is None
        assert VisionExtractor.from_payload(
            {"installation": {"value": None, "evidence": "no mounting structure visible"}}
        ).installation is None

    @pytest.mark.parametrize("notes,expected", [
        ("LED wall mounted on a steel frame above the stage", "fixed"),
        ("rental panels with quick locks and flight cases beside them", "rental"),
        ("a big screen on the wall", None),
    ])
    def test_notes_can_decide_installation(self, notes, expected):
        """模型把安装结构写在一句话描述（notes）里时，也要能用上。

        从描述里推出来的只能算"推测"，不能算图片明确可见。
        """
        result = VisionExtractor.from_payload({"installation": None, "notes": notes})
        if expected is None:
            assert result.installation is None, notes
        else:
            assert result.installation.value == expected, notes
            assert result.installation.source == "vision_inferred", notes

    def test_low_confidence_explicit_is_downgraded(self):
        """低置信度的"明确可见"要降级为推测 → Gate 会再问客户一次（图片判断可能看错）。"""
        result = VisionExtractor.from_payload({
            "environment": {"value": "indoor", "confidence": 0.9, "source": "vision_explicit",
                            "evidence": "ceiling"},
            "installation": {"value": "rental", "confidence": 0.4, "source": "vision_explicit",
                             "evidence": "truss"},
        })
        assert result.installation.value == "rental"
        assert result.installation.source == "vision_inferred"
        merged, _ = apply_vision_to_profile(RequirementProfile(), result)
        assert "installation" in check_recommendation_ready(merged).missing

    def test_high_confidence_explicit_is_kept(self):
        result = VisionExtractor.from_payload({
            "installation": {"value": "fixed", "confidence": 0.95, "source": "vision_explicit",
                             "evidence": "flush mounted in wall"},
        })
        assert result.installation.source == "vision_explicit"

    @pytest.mark.parametrize("source", ["vision_explicit", "vision_inferred"])
    def test_profile_merge_and_customer_confirmation(self, source):
        """图片判定的安装方式要进档案，并在"跟客户核对"那一句里说出来。"""
        from src.rag.reply_composer import vision_confirmation_sentence

        result = VisionExtractor.from_payload({
            "environment": {"value": "indoor", "confidence": 0.9, "source": "vision_explicit",
                            "evidence": "ceiling"},
            "installation": {"value": "快装快拆", "confidence": 0.8, "source": source,
                             "evidence": "快锁箱体"},
        })
        merged, _ = apply_vision_to_profile(RequirementProfile(), result)
        assert merged.installation == "rental"
        assert merged.sources["installation"] == source
        assert "installation" in merged.vision_confirmation_pending
        sentence = vision_confirmation_sentence(merged, "en", 0)
        assert "rental setup" in sentence, sentence

    def test_inferred_installation_still_asked_by_gate(self):
        """只是"推测"（vision_inferred）→ 不能算客户确认，Gate 仍要问固装/租赁。"""
        result = VisionExtractor.from_payload({
            "environment": {"value": "indoor", "confidence": 0.9, "source": "vision_explicit",
                            "evidence": "ceiling"},
            "installation": {"value": "rental", "confidence": 0.6, "source": "vision_inferred",
                             "evidence": "quick-lock edges"},
        })
        merged, _ = apply_vision_to_profile(RequirementProfile(), result)
        assert "installation" in check_recommendation_ready(merged).missing

    def test_visually_explicit_installation_is_settled(self):
        result = VisionExtractor.from_payload({
            "environment": {"value": "indoor", "confidence": 0.9, "source": "vision_explicit",
                            "evidence": "ceiling"},
            "installation": {"value": "fixed", "confidence": 0.9, "source": "vision_explicit",
                             "evidence": "flush mounted in wall"},
        })
        merged, _ = apply_vision_to_profile(RequirementProfile(), result)
        # 图片明确可见 → 不用再单独问安装方式（但会跟客户核对一次）
        assert "installation" not in check_recommendation_ready(merged).missing


class TestParseResponse:
    def test_parses_fenced_json_with_prefix(self):
        payload = parse_vision_json("Here is the result:\n```json\n{\"a\": 1}\n```")
        assert payload == {"a": 1}

    def test_parses_json_with_trailing_text(self):
        payload = parse_vision_json('{"a": 2}\nHope this helps!')
        assert payload == {"a": 2}

    def test_tolerates_trailing_comma(self):
        payload = parse_vision_json('{"a": 3,}')
        assert payload == {"a": 3}

    def test_bad_json_raises_vision_error(self):
        with pytest.raises(VisionError):
            parse_vision_json("sorry, I cannot analyse this image")

    def test_non_object_raises(self):
        with pytest.raises(VisionError):
            parse_vision_json("[1, 2, 3]")


class TestUnitNormalization:
    @pytest.mark.parametrize("value,expected", [
        ("5000 mm", 5.0), ("5 m", 5.0), ("5米", 5.0), ("500 cm", 5.0),
        ("16 ft", 4.8768), ("200 inch", 5.08), (5, 5.0), (5000, 5.0),
        ("about 8 meters", 8.0),
    ])
    def test_parse_meters(self, value, expected):
        assert parse_meters(value) == pytest.approx(expected, rel=1e-3)

    def test_parse_meters_rejects_junk(self):
        assert parse_meters("") is None
        assert parse_meters("not a number") is None

    @pytest.mark.parametrize("value,expected", [
        (2.5, 2.5), ("2.5mm", 2.5), ("P3.0", 3.0), ("0.0025 m", 2.5),
    ])
    def test_parse_mm(self, value, expected):
        assert parse_mm(value) == pytest.approx(expected)

    def test_parse_mm_rejects_out_of_range(self):
        assert parse_mm(50) is None      # 50mm 不是点间距
        assert parse_mm(0.01) is None

    def test_parse_nit(self):
        assert parse_nit("600 nit") == 600
        assert parse_nit(0) is None
        assert parse_nit(99999) is None


class TestSchemaCoercion:
    def test_explicit_and_inferred_are_kept_apart(self):
        result = VisionExtractor.from_payload(parse_vision_json(INDOOR_ROOM_JSON))
        assert result.display_type.source == VISION_EXPLICIT
        assert result.environment.value == "indoor"
        assert result.target_width_m.source == VISION_INFERRED
        assert result.target_width_m.value == pytest.approx(5.0)
        assert result.target_height_m.value == pytest.approx(3.0)
        assert result.viewing_distance_m is None

    def test_bare_value_is_treated_as_inferred(self):
        """模型没按格式返回时，绝不能把"猜的"当成"看到的"。"""
        result = VisionExtractor.from_payload({"environment": "indoor"})
        assert result.environment.value == "indoor"
        assert result.environment.source == VISION_INFERRED

    def test_unknown_source_is_treated_as_inferred(self):
        result = VisionExtractor.from_payload(
            {"environment": {"value": "outdoor", "source": "guess", "confidence": 0.9}}
        )
        assert result.environment.source == VISION_INFERRED

    def test_invalid_enum_is_dropped(self):
        result = VisionExtractor.from_payload(
            {"environment": {"value": "somewhere", "source": "vision_explicit"}}
        )
        assert result.environment is None

    def test_chinese_environment_and_specials(self):
        result = VisionExtractor.from_payload(
            {
                "environment": {"value": "室外", "source": "vision_explicit"},
                "installation": {"value": "租赁", "source": "vision_explicit"},
                "special_requirements": ["防水", "cob"],
            }
        )
        assert result.environment.value == "outdoor"
        assert result.installation.value == "rental"
        assert result.special_requirements == ["waterproof", "cob"]

    def test_metrics_counts(self):
        result = VisionExtractor.from_payload(parse_vision_json(INDOOR_ROOM_JSON))
        metrics = result.metrics()
        assert metrics["fields_extracted"] == 3       # LED / indoor / conference
        assert metrics["fields_inferred"] == 2        # 宽 / 高
        assert metrics["fields_null"] == 5


class TestExtractorCache:
    def test_same_image_same_session_calls_once(self):
        client = _FakeVisionClient(INDOOR_ROOM_JSON)
        extractor = VisionExtractor(client=client)

        first = extractor.extract(b"fake-image-bytes", session_id="s1")
        second = extractor.extract(b"fake-image-bytes", session_id="s1")

        assert client.calls == 1
        assert first is second
        assert first.image_hash == second.image_hash

    def test_different_session_does_not_reuse_cache(self):
        client = _FakeVisionClient(INDOOR_ROOM_JSON)
        extractor = VisionExtractor(client=client)

        extractor.extract(b"fake-image-bytes", session_id="s1")
        extractor.extract(b"fake-image-bytes", session_id="s2")

        assert client.calls == 2

    def test_different_image_calls_again(self):
        client = _FakeVisionClient(INDOOR_ROOM_JSON)
        extractor = VisionExtractor(client=client)

        extractor.extract(b"image-one", session_id="s1")
        extractor.extract(b"image-two", session_id="s1")

        assert client.calls == 2

    def test_extract_many_skips_broken_image(self):
        class _FlakyClient(_FakeVisionClient):
            def analyze_image(self, image, prompt, system_prompt=None, mime_type=""):
                self.calls += 1
                if image == b"bad":
                    raise VisionError("boom", kind="api_error")
                return self.payload

        extractor = VisionExtractor(client=_FlakyClient(INDOOR_ROOM_JSON))
        results = extractor.extract_many([b"good", b"bad", b"good2"], session_id="s1")

        assert len(results) == 2

    def test_merge_prefers_explicit_over_inferred(self):
        inferred = VisionExtractor.from_payload(
            {"environment": {"value": "indoor", "source": "vision_inferred"}}
        )
        explicit = VisionExtractor.from_payload(
            {"environment": {"value": "outdoor", "source": "vision_explicit"}}
        )
        merged = merge_vision_results([inferred, explicit])
        assert merged.environment.value == "outdoor"
        assert merged.environment.source == VISION_EXPLICIT


class TestImagePayloadChecks:
    def test_rejects_unsupported_mime(self):
        with pytest.raises(VisionError):
            check_image_payload(b"xx", mime_type="application/pdf")

    def test_rejects_oversized_image(self):
        big = b"0" * (2 * 1024 * 1024)
        with pytest.raises(VisionError):
            check_image_payload(big, mime_type="image/jpeg", max_mb=1)

    def test_accepts_normal_image(self):
        info = check_image_payload(b"0" * 1024, mime_type="image/png")
        assert info["mime_type"] == "image/png"


class TestApplyVisionToProfile:
    def test_empty_profile_is_filled_from_image(self):
        profile = RequirementProfile()
        vision = VisionExtractor.from_payload(parse_vision_json(INDOOR_ROOM_JSON))

        merged, stats = apply_vision_to_profile(profile, vision)

        assert merged.display_type == "LED"
        assert merged.environment == "indoor"
        assert merged.purpose == "conference"
        assert merged.sources["environment"] == VISION_EXPLICIT
        assert stats["merged_fields"] >= 3

    def test_customer_value_is_never_overwritten(self):
        slots = {"environment": "outdoor", "display_type": "LED"}
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        vision = VisionExtractor.from_payload(parse_vision_json(INDOOR_ROOM_JSON))

        merged, stats = apply_vision_to_profile(profile, vision)

        assert merged.environment == "outdoor", "客户明确说的不能被图片覆盖"
        assert merged.sources["environment"] == "explicit"
        assert stats["conflict_count"] == 1
        assert any("environment" in item for item in merged.conflicts)
        assert "environment" in merged.conflict_slots

    def test_vision_inferred_does_not_beat_explicit(self):
        slots = {"purpose": "church"}
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        vision = VisionExtractor.from_payload(
            {"purpose": {"value": "conference", "source": "vision_inferred"}}
        )

        merged, _ = apply_vision_to_profile(profile, vision)

        assert merged.purpose == "church"

    def test_vision_inferred_fills_missing_field(self):
        profile = RequirementProfile()
        vision = VisionExtractor.from_payload(
            {"environment": {"value": "outdoor", "source": "vision_inferred"}}
        )
        merged, _ = apply_vision_to_profile(profile, vision)

        assert merged.environment == "outdoor"
        assert merged.sources["environment"] == VISION_INFERRED

    def test_vision_size_only_becomes_hint(self):
        """计划第十八 / 二十阶段：图片尺寸绝不能变成工程计算输入。"""
        profile = RequirementProfile()
        vision = VisionExtractor.from_payload(parse_vision_json(INDOOR_ROOM_JSON))

        merged, stats = apply_vision_to_profile(profile, vision)

        assert merged.target_width_m is None
        assert merged.target_height_m is None
        assert merged.vision_size_hint_mm == pytest.approx([5000.0, 3000.0])
        assert stats["size_hint"] is True

    def test_special_requirements_recorded(self):
        profile = RequirementProfile()
        vision = VisionExtractor.from_payload(
            {"special_requirements": ["waterproof", "hdr"]}
        )
        merged, _ = apply_vision_to_profile(profile, vision)
        assert merged.special_requirements == ["waterproof", "hdr"]


class TestGateWithVision:
    def _profile_with_vision(self):
        profile = RequirementProfile()
        vision = VisionExtractor.from_payload(parse_vision_json(INDOOR_ROOM_JSON))
        merged, _ = apply_vision_to_profile(profile, vision)
        return merged

    def test_image_alone_is_not_recommendation_ready(self):
        decision = check_recommendation_ready(self._profile_with_vision())
        assert decision.ready is False
        assert decision.status == "CONTINUE_ASKING"
        # 客户口径（2026-09-18）：硬性条件 = 室内外 / 固装租赁 / P值 / 尺寸。
        # 图片能确定室内外，但"固装租赁 / P值 / 尺寸"仍要跟客户确认。
        assert "installation" in decision.missing
        assert {"pixel_pitch", "size"} & set(decision.missing)

    def test_clearly_visible_fields_are_not_asked_again(self):
        profile = self._profile_with_vision()
        assert profile.environment == "indoor"
        decision = check_recommendation_ready(profile)
        assert "environment" not in decision.missing
        assert "purpose" not in decision.missing

    def test_size_question_mentions_image_hint(self):
        profile = self._profile_with_vision()
        decision = check_calculation_ready(profile)
        assert decision.ready is False
        assert "5m x 3m" in (decision.next_question or "")

    def test_vision_size_does_not_open_calculation_gate(self):
        profile = self._profile_with_vision()
        assert check_calculation_ready(profile).ready is False

    def test_viewing_distance_from_image_still_asked(self):
        """观看距离图片推不准 → 不能被图片结果顶掉。"""
        profile = RequirementProfile()
        vision = VisionExtractor.from_payload(
            {"viewing_distance_m": {"value": 5.0, "source": "vision_explicit"}}
        )
        merged, _ = apply_vision_to_profile(profile, vision)
        assert merged.viewing_distance_m == pytest.approx(5.0)
        assert check_recommendation_ready(merged).ready is False


class TestVisionFailSafe:
    def test_api_failure_returns_empty_result(self, monkeypatch):
        import src.vision.integration as integration_module

        class _Boom:
            def extract_many(self, *args, **kwargs):
                raise VisionError("api down", kind="api_error")

        monkeypatch.setattr(integration_module, "get_vision_extractor", lambda: _Boom())

        results, metrics = extract_vision_for_turn([b"image"], session_id="s1")

        assert results == []
        assert metrics["vision_success"] is False
        assert metrics["error"]

    def test_single_image_failure_is_tolerated(self, monkeypatch):
        import src.vision.integration as integration_module

        class _Partial:
            def extract_many(self, images, **kwargs):
                from src.vision.extractor import VisionExtractor as VE

                return [VE.from_payload({"environment": {"value": "indoor", "source": "vision_explicit"}})]

        monkeypatch.setattr(integration_module, "get_vision_extractor", lambda: _Partial())
        results, metrics = extract_vision_for_turn([b"image"], session_id="s1")

        assert len(results) == 1
        assert metrics["vision_success"] is True
        assert metrics["fields_extracted"] == 1

    def test_no_images_is_noop(self):
        results, metrics = extract_vision_for_turn([], session_id="s1")
        assert results == []
        assert metrics["images"] == 0
        assert metrics["vision_success"] is False

    def test_broken_extractor_never_raises(self, monkeypatch):
        import src.vision.integration as integration_module

        monkeypatch.setattr(
            integration_module, "get_vision_extractor",
            lambda: (_ for _ in ()).throw(RuntimeError("nope")),
        )
        results, metrics = extract_vision_for_turn([b"image"], session_id="s1")
        assert results == []
        assert "nope" in metrics["error"]
