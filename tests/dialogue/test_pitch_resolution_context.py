"""v2.6 §14/§15：P 值 / 视距推导与 requested→resolved 的透明口径。"""
import os
import sys
from types import SimpleNamespace

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.pitch_resolution import EXACT, resolve_pitch  # noqa: E402


class TestPitchResolutionContext:

    def _profile(self, **slots):
        return RequirementProfile.from_slots(slots, explicit_keys=set(slots))

    def test_requested_pitch_is_kept_when_catalog_has_it(self):
        profile = self._profile(pixel_pitch_mm=3.0)
        model = SimpleNamespace(model="TW11-3216-P3.0", pixel_pitch_mm=3.0)
        resolution = resolve_pitch(profile, model, available_pitches=[2.5, 3.0, 4.0])
        assert resolution.match_type == EXACT
        assert resolution.requested_pitch == 3.0
        assert resolution.resolved_pitch == 3.0
        assert resolution.needs_explanation is False

    def test_nearest_available_is_explained_not_invented(self):
        """§15：P3 不在目录里 → resolved=P2.9 + 说明原因，数据全部来自计算。"""
        profile = self._profile(pixel_pitch_mm=3.0)
        model = SimpleNamespace(model="TW21-IRHD-P2.9", pixel_pitch_mm=2.9)
        resolution = resolve_pitch(
            profile, model, available_pitches=[2.5, 2.9, 4.0], band_min=1.5, band_max=4.0
        )
        payload = resolution.to_dict()
        assert payload["requested_pitch"] == 3.0
        assert payload["resolved_pitch"] == 2.9
        assert payload["pitch_match_type"]
        assert payload["pitch_resolution_reason"]
        assert payload["needs_explanation"] is True
        assert "P2.9" in resolution.explain()

    def test_resolution_feeds_pitch_window_when_customer_gave_distance(self):
        """§14：视距能推导点间距时，走结构化窗口，而不是让 LLM 自己猜。"""
        profile = self._profile(viewing_distance_m=5.0)
        model = SimpleNamespace(model="TW11-3216-P3.0", pixel_pitch_mm=3.0)
        resolution = resolve_pitch(profile, model, available_pitches=[2.5, 3.0, 4.0])
        assert resolution.requested_pitch is None
        assert resolution.resolved_pitch == 3.0
        assert resolution.to_dict()["pitch_match_type"]
