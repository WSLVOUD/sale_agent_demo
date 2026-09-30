"""LCD 识图分流（客户口径 2026-09-30）。

客户要求：

    1. 没确认 LED/LCD 时发图 → 先判类型；判成 LED 就用**现有 LED 那套**字段，
       判成 LCD 就只取"是否拼接 / 有没有摄像头"。
    2. 已经确认过类型再发图 → 直接按那个类型提取，不再判类型。
    3. 每次识别完都要把识别出来的内容拿给客户确认。
    4. 图文同时来 → 以客户文字为准；客户说"你定"或没正面回答 → 用识别结果；
       客户说"不对"但没说哪里不对 → **问他哪里不对**（不能当成没反对）。
    5. 拼接和摄像头是冲突的：两个都识别出来 → 只问"是否拼接"。
    6. **不要直接识别出 IFP**（归到 LCD）。
    7. 现有 LED 视觉识别不许改动。
"""
import json
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue.image_confirmation import prompt_from_profile  # noqa: E402
from src.memory.store import memory  # noqa: E402
from src.models.requirement import RequirementProfile  # noqa: E402
from src.vision import extractor as vex  # noqa: E402
from src.vision import integration as vi  # noqa: E402
from src.vision import lcd_extractor as lx  # noqa: E402
from src.vision.integration import apply_vision_to_profile, extract_vision_for_turn  # noqa: E402
from src.vision.lcd_prompt import LCD_VISION_SYSTEM_PROMPT  # noqa: E402
from src.vision.schema import VISION_EXPLICIT, VisionField, VisionRequirement  # noqa: E402

LED_IMAGE = b"image-of-an-led-wall"
LCD_IMAGE = b"image-of-an-lcd-wall"


def _field(value, confidence=0.9, source=VISION_EXPLICIT, evidence="seen in the photo"):
    return {"value": value, "confidence": confidence, "source": source, "evidence": evidence}


LED_PAYLOAD = {
    "display_type": _field("LED"),
    "environment": _field("indoor"),
    "installation": _field("fixed"),
    "pixel_pitch_mm": _field(4.0),
}
LCD_PAYLOAD = {
    "display_type": _field("LCD"),
    "is_splicing": _field(True, evidence="three panels lined up as one wall"),
    "camera_observed": None,
}


class FakeVisionClient:
    """按 system_prompt 分辨这一路是 LED 还是 LCD，各回一份假 JSON。"""

    def __init__(self, led_payload, lcd_payload):
        self.led_payload = led_payload
        self.lcd_payload = lcd_payload
        self.calls = []
        self.model = "fake-vision"

    def analyze_image(self, image, prompt, system_prompt=None, mime_type=""):
        if system_prompt == LCD_VISION_SYSTEM_PROMPT:
            self.calls.append("lcd")
            return json.dumps(self.lcd_payload)
        self.calls.append("led")
        return json.dumps(self.led_payload)


@pytest.fixture
def install_vision(monkeypatch):
    def _install(led_payload=None, lcd_payload=None):
        client = FakeVisionClient(led_payload or LED_PAYLOAD, lcd_payload or LCD_PAYLOAD)
        monkeypatch.setattr(vex, "_extractor", vex.VisionExtractor(client=client), raising=False)
        monkeypatch.setattr(
            lx, "_lcd_extractor", lx.LcdVisionExtractor(client=client), raising=False
        )
        return client

    vex.VisionExtractor._cache.clear()
    lx.LcdVisionExtractor._cache.clear()
    memory.clear_all()
    yield _install
    vex.VisionExtractor._cache.clear()
    lx.LcdVisionExtractor._cache.clear()
    memory.clear_all()


def _settle(session_id, display_type):
    memory.set_display_type_decision(
        session_id,
        {
            "display_type": display_type,
            "status": "CONFIRMED",
            "locked": True,
            "source": "customer",
            "confidence": 1.0,
        },
    )


class TestTypeRouting:
    def test_unknown_type_with_led_image_uses_the_existing_led_extractor(self, install_vision):
        # 这次客户发的确实是 LED 的图 → 类型探针也回 LED
        client = install_vision(led_payload=LED_PAYLOAD, lcd_payload=LED_PAYLOAD)

        results, metrics = extract_vision_for_turn([LED_IMAGE], "s-led-auto")

        # 先用带判定标准的提示词判类型（lcd 那路），判成 LED 再走现有 LED 提取器
        assert client.calls == ["lcd", "led"], client.calls
        assert metrics["vision_route"] == "led(auto)"
        assert results[0].display_type.value == "LED"
        # 现有 LED 字段照旧（没被动过）
        assert results[0].pixel_pitch_mm.value == 4.0

    def test_unknown_type_with_lcd_image_switches_to_the_lcd_extractor(self, install_vision):
        client = install_vision(led_payload=LCD_PAYLOAD, lcd_payload=LCD_PAYLOAD)

        results, metrics = extract_vision_for_turn([LCD_IMAGE], "s-lcd-auto")

        assert client.calls == ["lcd"], "判成 LCD 就直接用这一路的结果，不必再跑 LED 那一路"
        assert metrics["vision_route"] == "lcd(auto)"
        assert results[0].display_type.value == "LCD"
        assert results[0].is_splicing.value is True

    def test_the_type_probe_uses_the_criteria_bearing_prompt(self, install_vision):
        """实测 2026-09-30：图片有明显拼缝/边框却被判成 LED —— 类型判断必须用带标准的提示词。"""
        seen = {}

        client = install_vision()
        original = client.analyze_image

        def spy(image, prompt, system_prompt=None, mime_type=""):
            seen.setdefault("prompts", []).append(system_prompt)
            return original(image, prompt, system_prompt=system_prompt, mime_type=mime_type)

        client.analyze_image = spy

        extract_vision_for_turn([LED_IMAGE], "s-probe")

        assert seen["prompts"][0] == LCD_VISION_SYSTEM_PROMPT, "判类型必须先走带标准的提示词"

    def test_settled_lcd_session_never_calls_the_led_extractor(self, install_vision):
        client = install_vision()
        _settle("s-lcd-known", "LCD")

        results, metrics = extract_vision_for_turn([LCD_IMAGE], "s-lcd-known")

        assert client.calls == ["lcd"], "类型已经确认是 LCD → 不需要再判类型"
        assert metrics["vision_route"] == "lcd(confirmed)"
        assert results[0].display_type.value == "LCD"

    def test_settled_led_session_never_calls_the_lcd_extractor(self, install_vision):
        client = install_vision()
        _settle("s-led-known", "LED")

        results, metrics = extract_vision_for_turn([LED_IMAGE], "s-led-known")

        assert client.calls == ["led"]
        assert metrics["vision_route"] == "led(confirmed)"

    def test_ifp_is_never_returned_as_a_type(self, install_vision):
        install_vision(
            led_payload={
                "display_type": _field("IFP"),
                "is_splicing": _field(True),
                "camera_observed": None,
            },
            lcd_payload={
                "display_type": _field("IFP"),
                "is_splicing": _field(True),
                "camera_observed": None,
            },
        )

        results, _metrics = extract_vision_for_turn([LCD_IMAGE], "s-ifp")

        assert results[0].display_type.value == "LCD", "IFP 必须归到 LCD，不能直接产出 IFP"


class TestLcdFactsMerge:
    def _lcd_vision(self, *, splicing=True, camera=None):
        payload = {
            "display_type": VisionField(value="LCD", confidence=0.9, source=VISION_EXPLICIT),
        }
        if splicing is not None:
            payload["is_splicing"] = VisionField(
                value=splicing, confidence=0.8, source=VISION_EXPLICIT
            )
        if camera is not None:
            payload["camera_observed"] = VisionField(
                value=camera, confidence=0.7, source=VISION_EXPLICIT
            )
        return VisionRequirement(**payload)

    def test_lcd_facts_land_in_the_profile_and_are_queued_for_confirmation(self):
        profile = RequirementProfile()
        profile, stats = apply_vision_to_profile(profile, self._lcd_vision(splicing=True))

        assert profile.lcd_is_splicing is True
        assert "lcd_is_splicing" in profile.vision_confirmation_pending
        assert profile.vision_assertions.get("lcd_is_splicing") is True
        assert "lcd_is_splicing" in stats.get("lcd_fields", [])

    def test_splicing_and_camera_conflict_keeps_only_splicing(self):
        """客户口径：拼接和摄像头不会同时成立 → 只问"是否拼接"。"""
        profile = RequirementProfile()
        profile, _stats = apply_vision_to_profile(
            profile, self._lcd_vision(splicing=True, camera=True)
        )

        assert profile.lcd_is_splicing is True
        assert profile.lcd_camera_observed is None, "冲突时丢掉摄像头"
        assert "lcd_camera_observed" not in profile.vision_confirmation_pending

    def test_camera_is_kept_when_it_is_not_a_spliced_wall(self):
        profile = RequirementProfile()
        profile, _stats = apply_vision_to_profile(
            profile, self._lcd_vision(splicing=False, camera=True)
        )

        assert profile.lcd_camera_observed is True
        assert "lcd_camera_observed" in profile.vision_confirmation_pending

    def test_customer_text_beats_the_image(self):
        """图文同时来 → 以客户文字为准（客户说单体屏，图片看成拼接 → 留客户的）。"""
        profile = RequirementProfile(
            display_type="LCD", lcd_is_splicing=False, sources={"lcd_is_splicing": "explicit"}
        )
        profile, stats = apply_vision_to_profile(profile, self._lcd_vision(splicing=True))

        assert profile.lcd_is_splicing is False, "客户明说的值不能被图片改掉"
        assert stats["conflict_count"] >= 1

    def test_led_vision_result_does_not_touch_lcd_fields(self):
        """LED 那一路的结果里没有这两个字段 → LCD 字段保持为空（行为不变）。"""
        led = VisionRequirement(
            display_type=VisionField(value="LED", confidence=0.9, source=VISION_EXPLICIT),
            environment=VisionField(value="indoor", confidence=0.9, source=VISION_EXPLICIT),
        )
        profile, _stats = apply_vision_to_profile(RequirementProfile(), led)

        assert profile.lcd_is_splicing is None
        assert profile.lcd_camera_observed is None
        assert "environment" in profile.vision_confirmation_pending


class TestCustomerConfirmation:
    def _merged(self):
        payload = {
            "display_type": VisionField(value="LCD", confidence=0.9, source=VISION_EXPLICIT),
            "is_splicing": VisionField(value=True, confidence=0.8, source=VISION_EXPLICIT),
        }
        profile, _stats = apply_vision_to_profile(
            RequirementProfile(display_type="LCD", sources={"display_type": "explicit"}),
            VisionRequirement(**payload),
        )
        return profile

    def test_confirmation_asks_about_what_was_recognised(self):
        prompt = prompt_from_profile(self._merged(), language="en")

        assert "spliced" in prompt.lower() or "video wall" in prompt.lower(), prompt

    def test_customer_confirms(self):
        profile = self._merged()

        stats = vi.resolve_vision_confirmation(profile, "yes, that's right")

        assert stats["confirmed"], stats
        assert profile.vision_confirmation_pending == []
        assert (profile.sources or {}).get("lcd_is_splicing") == "confirmed"

    def test_customer_delegates_to_the_ai(self):
        """客户说"你定" → 按识别结果走。"""
        profile = self._merged()

        stats = vi.resolve_vision_confirmation(profile, "you decide")

        assert stats["accepted"], stats
        assert profile.vision_confirmation_pending == []
        assert (profile.sources or {}).get("lcd_is_splicing") == "vision_accepted"

    def test_customer_silent_also_adopts_the_recognition(self):
        """客户没正面回答 → 也按识别结果走。"""
        profile = self._merged()

        stats = vi.resolve_vision_confirmation(profile, "how much does it cost?")

        assert stats["accepted"], stats
        assert profile.vision_confirmation_pending == []

    def test_customer_says_no_without_detail_asks_what_is_wrong(self):
        """客户说"不对"但没说哪里不对 → 不能当成没反对，要问他哪里不对。"""
        profile = self._merged()

        stats = vi.resolve_vision_confirmation(profile, "no, that's wrong")

        assert stats["denied"], stats
        assert profile.vision_confirmation_pending == ["lcd_is_splicing"]
        assert profile.vision_confirmation_denied == ["lcd_is_splicing"]

        prompt = prompt_from_profile(profile, language="en")
        assert "which part" in prompt.lower(), prompt
        assert "spliced" in prompt.lower() or "video wall" in prompt.lower(), prompt

    def test_correction_is_recorded_and_wins(self):
        """客户给出正确的值 → 记一条纠正，客户值优先。"""
        profile = self._merged()
        profile.lcd_is_splicing = False
        profile.sources = {**(profile.sources or {}), "lcd_is_splicing": "explicit"}

        stats = vi.resolve_vision_confirmation(profile, "no, they are single displays")

        assert stats["corrected"] == ["lcd_is_splicing"], stats
        assert any("image said True" in note for note in profile.vision_corrections)

    def test_no_with_a_specific_correction_does_not_hold_up_the_other_fields(self):
        """客户一边说 no 一边指出了哪里不对 → 只有没被反对的那项算"已核对过"。

        回归用例：不能因为句子里有 "no" 就把**所有**图片字段都挂起来反复问 ——
        客户已经把不对的那项纠正了，其余项就按识别结果继续。
        """
        profile = RequirementProfile(
            display_type="LCD",
            sources={"display_type": "vision_explicit"},
        )
        profile.lcd_is_splicing = True
        profile.sources["lcd_is_splicing"] = "vision_explicit"
        profile.vision_confirmation_pending = ["display_type", "lcd_is_splicing"]
        profile.vision_assertions = {"display_type": "LCD", "lcd_is_splicing": True}
        # 客户这句话已经把正确的值说出来了（Extractor 会写成 explicit）
        profile.lcd_is_splicing = False
        profile.sources["lcd_is_splicing"] = "explicit"

        stats = vi.resolve_vision_confirmation(profile, "no, they are single displays")

        assert stats["corrected"] == ["lcd_is_splicing"], stats
        assert profile.vision_confirmation_pending == [], profile.vision_confirmation_pending
        assert stats["accepted"] == ["display_type"], stats
