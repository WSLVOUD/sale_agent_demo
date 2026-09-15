"""关键词依赖测试集（《关键词依赖优化计划 v1.0》Phase 9/19/20/21）的公共设施。

设计要点：
  - Golden Cases 里 `requires_semantic: true` 的用例需要 LLM 语义理解，
    默认**跳过**（离线环境无网络）；设置 RUN_LLM_EXTRACTION_TESTS=1 后才会跑。
  - 其余用例离线可跑：规则解析 + PurposeNormalizer 的短语表即可覆盖。
"""
import json
import os
import sys
from pathlib import Path

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@pytest.fixture(scope="session")
def golden_cases():
    path = Path(__file__).parent / "golden_cases.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def extractor():
    from src.core.requirement_extractor import RequirementExtractor

    return RequirementExtractor()


def run_llm_tests() -> bool:
    return os.getenv("RUN_LLM_EXTRACTION_TESTS", "").strip() in ("1", "true", "yes")


skip_without_llm = pytest.mark.skipif(
    not run_llm_tests(),
    reason="需要 LLM 语义理解（设置 RUN_LLM_EXTRACTION_TESTS=1 且具备网络/API Key 时运行）",
)
