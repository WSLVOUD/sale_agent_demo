"""评测包（客户口径 2026-09-22：只保留还在用的评测）。

保留的三条评测入口（都能离线跑，结果写 eval/reports/，该目录已 gitignore）：

    python -m eval.calculator_eval        # 箱体/模组计算（14 用例）
    python -m eval.recommendation_eval    # 需求理解 / 路由 / 推荐引擎
    python -m eval.retrieval_eval         # 检索 Recall@K / MRR / 硬约束违规
    python -m eval.dialogue.run_dialogue_eval   # 对话链路行为约束

数据：eval/golden_dataset.json（82 条标注用例）、eval/dialogue_chain_golden.json、
eval/naturalness_golden.json、eval/dialogue/golden_dialogue.json。
（Phase 0 时代的 run_eval.py / dataset.json 已被上面的脚本取代，2026-09-22 删除。）

注意：这里**不导出**任何符号（旧版 `from .run_eval import ...` 会让
`import eval.metrics` 连带失败）。
"""

__all__: list = []
