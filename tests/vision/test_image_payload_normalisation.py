"""客户实测（2026-09-21）：上传图片后 AI 没识别出室内外 / 固定安装 / 租赁。

根因：前端把图片发成**对象**（``{data, mime_type}`` / ``{url}``），
而这条链路只认字符串 → 视觉模型**根本没被调用**：

    WARNING:src.vision.extractor:Vision extract failed (image #1): unsupported image payload
    Vision metrics: images=1 latency_ms=1 … success=False

修好之后：所有形态的图片负载都会被归一化成 URL / data URL，
再交给视觉模型（这样室内外、固装/租赁才能从照片里读出来）。
"""
import base64
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.input.message import collect_messages, normalize_image_payloads  # noqa: E402
from src.vision.client import (  # noqa: E402
    VisionError,
    _to_data_url,
    check_image_payload,
)

B64 = base64.b64encode(b"fake-image-bytes" * 4).decode()
DATA_URL = f"data:image/png;base64,{B64}"


class TestNormalizeImagePayloads:

    def test_frontend_object_with_data_and_mime(self):
        assert normalize_image_payloads([{"data": B64, "mime_type": "image/png"}]) == [DATA_URL]

    def test_frontend_object_with_url(self):
        assert normalize_image_payloads([{"url": "https://example.com/a.jpg"}]) == [
            "https://example.com/a.jpg"
        ]

    def test_plain_string_and_data_url(self):
        assert normalize_image_payloads([DATA_URL]) == [DATA_URL]
        assert normalize_image_payloads(["data:image/jpeg;base64," + B64]) == [
            "data:image/jpeg;base64," + B64
        ]

    def test_truncated_data_url_prefix_is_repaired(self):
        assert normalize_image_payloads([f"image/png;base64,{B64}"]) == [DATA_URL]

    def test_duplicates_are_collapsed_and_empty_dropped(self):
        # 没给 mime_type 时按 jpeg 兜底
        assert normalize_image_payloads([{"data": B64}, {"data": B64}, None, ""]) == [
            f"data:image/jpeg;base64,{B64}"
        ]

    def test_object_with_attributes(self):
        class _Image:
            url = "https://example.com/b.png"
            data = ""
            mime_type = ""

        assert normalize_image_payloads([_Image()]) == ["https://example.com/b.png"]

    def test_messages_carry_normalised_images(self):
        """前端在 messages[].images 里发的对象，也要在入口层变成字符串。"""
        messages = collect_messages(
            session_id="img-1",
            messages=[
                {
                    "text": "i need this one",
                    "message_id": "m1",
                    "images": [{"data": B64, "mime_type": "image/png"}],
                }
            ],
        )
        assert messages[0].images == [DATA_URL]


class TestVisionClientAcceptsObjects:

    @pytest.mark.parametrize(
        "payload",
        [
            {"data": B64, "mime_type": "image/png"},
            {"url": "https://example.com/a.jpg"},
            DATA_URL,
            "https://example.com/a.jpg",
            B64,
            b"raw",
        ],
    )
    def test_payload_shapes_become_data_urls_or_urls(self, payload):
        out = _to_data_url(payload)
        assert out.startswith(("data:image/", "http"))

    def test_check_image_payload_accepts_objects(self):
        info = check_image_payload({"data": B64, "mime_type": "image/png"})
        assert info["mime_type"] == "image/png"

    def test_empty_object_still_rejected(self):
        with pytest.raises(VisionError):
            _to_data_url({})
