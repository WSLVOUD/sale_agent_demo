"""回归：客户明说"室外固装"时，就按客户说的做 —— 不记冲突、不追问、更不许泄露内部串。

客户口径（2026-10 实测）：

    客户: permanent
    AI  : A quick-install quick-release setup makes sense for a stage screen you'll be
          moving between events …            ← 与客户明说的 permanent 相反
    AI  : Just to make sure I have it right, could you confirm the environment?
          (installation_conflict: rule=fixed vs semantic=rental)   ← 内部串泄露

两个缺陷：
  ① 客户明说 permanent（固装），语义侧从 "play video" 推成 rental，系统记了
     installation_conflict 并**推翻了客户明说的事实**；
  ② conflict_message 把内部冲突串原样贴给了客户。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestExplicitCustomerFactWins:

    def test_rule_value_from_the_customer_is_not_flagged_as_a_conflict(self):
        """规则值来自客户明说时，直接以客户为准，不再记 semantic conflict。"""
        source = open(
            os.path.join(project_root, "src", "core", "requirement_extractor.py"),
            encoding="utf-8",
        ).read()
        # 旧的"记冲突"写法必须消失
        assert "semantic_conflicts.append" not in source or (
            "_conflict: rule=" not in source
        ), "客户明说的事实仍被记成冲突"
        assert "客户明说" in source and "不记冲突" in source

    def test_installation_written_by_the_customer_beats_purpose_inference(self):
        """可执行验证：客户说 permanent，不该因为场景被推成 rental 而反悔。"""
        from src.core.environment_installation_resolver import ConflictDetector

        # 明确值与推断值不同 → 确认冲突检测本身仍能识别（这是底层能力，保留）
        conflicts = ConflictDetector.detect_installation_conflicts(
            explicit="fixed", inferred="rental"
        )
        assert conflicts, "底层冲突检测被误删"

        # 但真正该拦住的是"把冲突抛给客户"这一步：冲突不得含内部串
        from src.engineering import conflicts as C

        class _P:
            conflicts = ["installation_conflict: rule=fixed vs semantic=rental"]
            conflict_slots = ["installation"]

        message = C.conflict_message(_P()) or ""
        assert "installation_conflict" not in message, message
        assert "rule=" not in message and "semantic=" not in message, message
        assert "confirm" in message.lower(), message


class TestConflictMessageNeverLeaksInternals:

    def _profile(self, conflicts, slots):
        class _P:
            pass

        p = _P()
        p.conflicts = conflicts
        p.conflict_slots = slots
        return p

    def test_no_internal_token_survives_into_the_customer_text(self):
        from src.engineering.conflicts import conflict_message

        for raw in (
            "installation_conflict: rule=fixed vs semantic=rental",
            "environment_conflict: rule=indoor vs semantic=outdoor",
        ):
            message = conflict_message(self._profile([raw], ["installation"])) or ""
            for token in ("_conflict", "rule=", "semantic=", "vs "):
                assert token not in message, (raw, message)

    def test_it_still_asks_a_plain_language_question(self):
        from src.engineering.conflicts import conflict_message

        msg = conflict_message(
            self._profile(["installation_conflict: x"], ["installation"])
        ) or ""
        assert "fixed install" in msg and "quick-release" in msg, msg
        # 问客户安装方式时同样不许出现 rental
        assert "rental" not in msg.lower(), msg
