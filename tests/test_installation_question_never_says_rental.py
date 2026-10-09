"""回归：LED 链路**问客户**安装方式时，不许出现 "rental"。

客户口径（2026-10）：

    「led 链路里在问固定安装还是租赁这个问题的时候，不要再用 rental 这个词去问客户，
      这个只需要需求路由里记录是 rental 就行了，问客户的只需要问客户是要固定安装，
      还是需要块状快拆的哪种」

也就是：`installation=rental` / `is_rental` 仍是**内部口径**（检索、硬过滤照用），
但对客户只说**安装形态** —— 固定安装 vs 块状快拆（可以一块块拆装搬走）。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 这些词都不该出现在"问客户"的句子里
# · rental / 租赁 / 租用：内部商业口径，不许问客户
# · 块状快拆 / block-by-block / modular quick-release：我一度用错的词 ——
#   客户口径是"**快装快拆、可以灵活搬动的**"
_FORBIDDEN = (
    "rental", "租赁", "租用", "租的",
    "块状快拆", "block-by-block", "modular quick-release",
)


def _is_clean(text: str) -> bool:
    lowered = str(text or "").lower()
    return not any(word.lower() in lowered for word in _FORBIDDEN)


class TestOneCanonicalWordingPerLanguage:
    """客户口径：话术只用**一种**能表达意思的说法，措辞交给 AI 润色。"""

    def test_installation_questions_keep_exactly_one_wording_each(self):
        from src.rag import readiness as R

        for name, table in (
            ("QUESTION_VARIANTS", R.QUESTION_VARIANTS),
            ("EASIER_QUESTIONS", R.EASIER_QUESTIONS),
        ):
            for lang, variants in (table.get("installation") or {}).items():
                assert len(variants) == 1, (name, lang, variants)
                assert variants[0].strip(), (name, lang)

    def test_the_canonical_wording_says_fixed_vs_quick_release_and_movable(self):
        from src.rag import readiness as R

        en = R.QUESTION_VARIANTS["installation"]["en"][0].lower()
        zh = R.QUESTION_VARIANTS["installation"]["zh"][0]
        assert "fixed installation" in en and "quick-release" in en
        assert "move around" in en or "movable" in en
        assert "固定安装" in zh and "快装快拆" in zh and "搬动" in zh

    def test_the_polisher_is_told_to_vary_the_wording_and_never_invent(self):
        from src.dialogue.response_generator import POLISH_SYSTEM_PROMPT

        text = POLISH_SYSTEM_PROMPT
        assert "different" in text.lower(), "没要求换句式"
        assert "not reuse" in text.lower() or "do not reuse" in text.lower(), text
        assert "invent" in text.lower(), "没强调不许编造"


class TestCustomerFacingInstallationQuestions:

    def test_readiness_variants_never_say_rental(self):
        from src.rag import readiness as R

        tables = {
            "QUESTION_VARIANTS": R.QUESTION_VARIANTS,
            "EASIER_QUESTIONS": R.EASIER_QUESTIONS,
            "CONFIRM_QUESTION_VARIANTS": R.CONFIRM_QUESTION_VARIANTS,
        }
        checked = 0
        for table_name, table in tables.items():
            for lang, variants in (table.get("installation") or {}).items():
                assert variants, f"{table_name} 安装方式问句为空"
                for text in variants:
                    assert _is_clean(text), f"{table_name}[{lang}] 仍对客户说 rental: {text}"
                    checked += 1
        assert checked >= 2, checked

    def test_labels_used_to_render_questions_are_clean(self):
        from src.rag import readiness as R

        assert _is_clean(R.MISSING_LABELS["installation"]), R.MISSING_LABELS["installation"]
        assert _is_clean(R._HUMAN_SLOT_LABELS["installation_or_distance"])

    def test_the_sales_question_planner_is_clean(self):
        source = open(
            os.path.join(project_root, "src", "agents", "sales", "question_planner.py"),
            encoding="utf-8",
        ).read()
        # 安装方式那一问的英文/中文问句都不能含 rental / 租赁
        block = source[source.index('"installation"') : source.index('"installation"') + 400]
        for line in block.splitlines():
            if line.strip().startswith(('"', "“")):
                assert _is_clean(line), line

    def test_image_confirmation_prompt_is_clean(self):
        from src.dialogue import image_confirmation as IC

        table = None
        for name in dir(IC):
            value = getattr(IC, name)
            if isinstance(value, dict) and "installation" in value and "environment" in value:
                table = value
                break
        assert table is not None, "找不到图片确认的字段问句表"
        en, zh = table["installation"]
        assert _is_clean(en) and _is_clean(zh), (en, zh)

    def test_the_internal_vocabulary_is_untouched(self):
        """内部口径照旧：路由、检索仍认 rental / is_rental。"""
        from src.models.requirement import RequirementProfile

        profile = RequirementProfile()
        assert profile.installation in (None, "fixed", "rental")
        from src.rag import hard_filter

        source = open(hard_filter.__file__, encoding="utf-8").read()
        assert "is_rental" in source, "内部检索仍必须能按租赁过滤"
