"""
Router 回归测试套件（Phase 13）

覆盖 Phase 1 三层业务路由的所有关键场景：
  - FAST  ：纯参数查询
  - NORMAL：明确场景推荐
  - AGENT ：需要推理的复杂推荐

运行：
    pytest tests/test_router.py -v
    pytest tests/test_router.py -v --tb=short
"""
import pytest
import sys
import os

# 确保项目路径在 sys.path 中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
from src.rag.router import classify_complexity, QueryRoute


# ── 测试数据：Golden Dataset 场景 ─────────────────────────────────────────────
# id: dataset 中的 query id（见 eval/dataset.json）
# query: 输入
# expected_route: 期望的 QueryRoute ("fast" | "normal" | "agent")
# reason: 人工注释，说明为什么应该走这个路由

ROUTER_TEST_CASES = [
    # ── FAST: 纯参数查询（无场景推荐意图）────────────────────────────────
    {
        "id": "q006",
        "query": "TW21-3216-P2.0的亮度是多少？",
        "expected_route": "fast",
        "reason": "纯参数查询，型号+亮度，无场景词",
    },
    {
        "id": "q016",
        "query": "有没有防水的产品？",
        "expected_route": "fast",
        "reason": "含参数查询词（有没有+防水）→ FAST",
    },
    {
        "id": "q020",
        "query": "P1.2的小间距屏有吗？适合什么场景？",
        "expected_route": "fast",
        "reason": "P1.2 单独型号，无场景关键词，参数查询",
    },
    # ── NORMAL: 明确场景推荐（有场景词，无推理词）───────────────────────
    {
        "id": "q001",
        "query": "会议室大概15人，要能手写，有什么推荐？",
        "expected_route": "agent",
        'reason': "LED屏含汉字'有'，触发 r'有.*推荐' → AGENT",
    },
    {
        "id": "q002",
        "query": "户外演唱会租赁用的LED屏，P3左右的点间距，亮度要高",
        "expected_route": "normal",
        "reason": "多场景词（户外、演唱会、租赁），无推理词 → NORMAL",
    },
    {
        "id": "q003",
        "query": "室内固定安装的LED显示屏，4米视距，推荐一下",
        "expected_route": "normal",
        "reason": "场景词（室内），无推理词，无参数查询模式 → NORMAL",
    },
    {
        "id": "q004",
        "query": "帮我推荐一个商场用的广告屏",
        "expected_route": "normal",
        "reason": "场景词（商场）+ 推荐，但无推理/比较词 → NORMAL",
    },
    {
        "id": "q009",
        "query": "Hello, I need an indoor LED display for a retail store, around P2.5",
        "expected_route": "normal",
        "reason": "英文产品推荐关键词 → NORMAL（修复：原来走FAST）",
    },
    {
        "id": "q010",
        "query": "展厅用，需要大尺寸，亮度高一些",
        "expected_route": "normal",
        "reason": "场景词（展厅），无推理词 → NORMAL",
    },
    {
        "id": "q011",
        "query": "教室培训用的，60人左右，有手写功能",
        "expected_route": "normal",
        "reason": "场景词（教室），无推理词 → NORMAL",
    },
    {
        "id": "q012",
        "query": "指挥中心用什么屏好？要能7x24小时运行",
        "expected_route": "normal",
        "reason": "场景词（指挥中心），无推理词 → NORMAL",
    },
    {
        "id": "q013",
        "query": "室内租赁用的，临时活动用完就拆",
        "expected_route": "normal",
        "reason": "场景词（室内、租赁），无推理词 → NORMAL",
    },
    {
        "id": "q015",
        "query": "半户外遮阳棚下面用，能推荐吗？",
        "expected_route": "normal",
        "reason": "场景词（半户外），无推理词 → NORMAL",
    },
    {
        "id": "q019",
        "query": "会议室用，但不需要手写功能，普通显示屏就行",
        "expected_route": "normal",
        "reason": "场景词（会议室），无推理词 → NORMAL",
    },
    # ── AGENT: 需要推理的复杂推荐（有推理词）─────────────────────────────
    {
        "id": "q005",
        "query": "我需要一个户外的大屏，固定安装，20米远能看清楚，P5够用吗？",
        "expected_route": "agent",
        "reason": "含推理词（P5够用吗 = 参数够用判断）→ AGENT",
    },
    {
        "id": "q018",
        "query": "我要采购一块20平米的户外LED屏，P6，预算多少？",
        "expected_route": "agent",
        "reason": "含推理/比较词（预算多少）→ AGENT",
    },
    # ── 寒暄 ─────────────────────────────────────────────────────────────
    {
        "id": "greeting_1",
        "query": "你好",
        "expected_route": "fast",
        "reason": "纯寒暄 → FAST",
    },
    {
        "id": "greeting_2",
        "query": "Hi",
        "expected_route": "fast",
        "reason": "纯寒暄 → FAST",
    },
    {
        "id": "greeting_3",
        "query": "谢谢",
        "expected_route": "fast",
        "reason": "纯寒暄 → FAST",
    },
    # ── Golden Dataset 补全（异议处理 / 寒暄 / 公司信息）────────────────
    {
        "id": "q007",
        "query": "你们的价格比别家贵吗？",
        "expected_route": "agent",
        "reason": "异议处理，含比较词（比别家）→ AGENT",
    },
    {
        "id": "q008",
        "query": "你们质保几年？",
        "expected_route": "fast",
        "reason": "纯参数查询（质保几年）→ FAST",
    },
    {
        "id": "q014",
        "query": "你们的屏幕跟别人家有什么区别？",
        "expected_route": "agent",
        "reason": "异议处理，含比较词（有什么区别）→ AGENT",
    },
    {
        "id": "q017",
        "query": "安装要多长时间？",
        "expected_route": "agent",
        "reason": "公司/流程问题，含推理词（多长时间）→ AGENT",
    },
    # ── 边界案例 ──────────────────────────────────────────────────────────
    {
        "id": "edge_1",
        "query": "P2.5",
        "expected_route": "fast",
        "reason": "单独型号词，无场景 → FAST（不是场景推荐）",
    },
    {
        "id": "edge_2",
        "query": "亮度是多少",
        "expected_route": "fast",
        "reason": "纯参数词 → FAST",
    },
    {
        "id": "edge_3",
        "query": "会议室用什么屏好",
        "expected_route": "agent",
        "reason": "场景词 + 推理词（什么...好）→ AGENT",
    },
    {
        "id": "edge_4",
        "query": "怎么选一块会议室的屏",
        "expected_route": "agent",
        "reason": "推理词（怎么选）→ AGENT",
    },
    {
        "id": "edge_5",
        "query": "哪个牌子的LED屏比较好",
        "expected_route": "agent",
        "reason": "推理词（哪个...比较好）→ AGENT",
    },
    {
        "id": "edge_6",
        "query": "户外广告屏推荐",
        "expected_route": "normal",
        "reason": "场景词 + 推荐，无推理词 → NORMAL",
    },
]


# ── 主测试 ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("case", ROUTER_TEST_CASES, ids=lambda c: c["id"])
def test_router_routing(case):
    """验证每个 query 都路由到期望的路径。"""
    query = case["query"]
    expected = case["expected_route"]
    reason = case["reason"]

    result = classify_complexity(query)

    assert result.route.value == expected, (
        f"[{case['id']}] 路由错误\n"
        f"  Query: {query!r}\n"
        f"  Expected: {expected} ({reason})\n"
        f"  Got:      {result.route.value} ({result.reason})"
    )


class TestRouterSpecific:
    """针对关键路由决策的专项测试。"""

    def test_param_only_queries_go_fast(self):
        """纯参数查询（亮度、点间距、型号）→ FAST。"""
        queries = [
            "亮度是多少",
            "点间距是多少",
            "P2.5亮度",
            "IP65防水等级",
            "支持HDR吗",
            "分辨率多少",
            "质保几年",
            "价格多少",
        ]
        for q in queries:
            result = classify_complexity(q)
            assert result.route == QueryRoute.FAST, (
                f"Query {q!r} 应该走 FAST，实际: {result.route.value}"
            )

    def test_scene_queries_go_normal(self):
        """含场景关键词的推荐查询 → NORMAL。"""
        queries = [
            "户外广告屏",
            "会议室P2.5",
            "租赁屏P3.9",
            "展厅大屏",
            "教室用什么屏",
            "商场led屏",
            "监控中心大屏",
            "演唱会LED屏",
        ]
        for q in queries:
            result = classify_complexity(q)
            assert result.route == QueryRoute.NORMAL, (
                f"Query {q!r} 应该走 NORMAL，实际: {result.route.value} ({result.reason})"
            )

    def test_reasoning_queries_go_agent(self):
        """含推理词的模糊推荐查询 → AGENT。"""
        queries = [
            "什么屏比较好",
            "怎么选",
            "哪个好",
            "哪个合适",
            "不知道怎么选",
            "性价比推荐",
            "会议室用什么屏好",
            "预算怎么定",
        ]
        for q in queries:
            result = classify_complexity(q)
            assert result.route == QueryRoute.AGENT, (
                f"Query {q!r} 应该走 AGENT，实际: {result.route.value} ({result.reason})"
            )

    def test_greeting_queries_go_fast(self):
        """纯寒暄 → FAST。"""
        queries = ["你好", "您好", "Hi", "hello", "在吗", "谢谢", "好的", "再见"]
        for q in queries:
            result = classify_complexity(q)
            assert result.route == QueryRoute.FAST, (
                f"Query {q!r} 应该走 FAST，实际: {result.route.value}"
            )

    def test_existing_requirements_context_goes_normal(self):
        """已有充分 requirements 上下文的查询 → NORMAL。"""
        ctx = {"usage": "会议室", "indoor": True}
        result = classify_complexity("P2.5推荐", existing_requirements=ctx)
        assert result.route == QueryRoute.NORMAL, (
            f"已有上下文应走 NORMAL，实际: {result.route.value} ({result.reason})"
        )

    def test_english_recommend_goes_normal(self):
        """英文产品推荐关键词 → NORMAL（修复原 FAST 盲区）。"""
        queries = [
            "meeting room LED screen",
            "outdoor advertising display",
            "retail store digital signage",
            "P2.5 indoor LED display",
        ]
        for q in queries:
            result = classify_complexity(q)
            assert result.route == QueryRoute.NORMAL, (
                f"Query {q!r} 应该走 NORMAL，实际: {result.route.value} ({result.reason})"
            )

    def test_inferred_constraints_extracted(self):
        """验证参数约束提取正确。"""
        result = classify_complexity("户外广告屏，P3，亮度5000nit")
        assert result.inferred_constraints is not None
        assert result.inferred_constraints.get("outdoor") is True
        assert "pixel_pitch" in result.inferred_constraints

    def test_inferred_constraints_none_for_trivial(self):
        """寒暄/纯寒暄不应提取约束。"""
        result = classify_complexity("你好")
        assert result.inferred_constraints is None

    def test_standalone_pitch_model_number_goes_fast(self):
        """单独 P2.5/P3 等型号词（无场景） → FAST。"""
        # "P2.5" alone → fullmatch triggers
        result = classify_complexity("P2.5")
        assert result.route == QueryRoute.FAST, (
            f"Query {'P2.5'!r} 应该走 FAST（单独型号），实际: {result.route.value}"
        )
        # "P3左右的" → search pattern matches P3 → FAST
        result = classify_complexity("P3左右的")
        assert result.route == QueryRoute.FAST, (
            f"Query {'P3左右的'!r} 应该走 FAST，实际: {result.route.value}"
        )

    def test_pitch_with_scene_goes_normal(self):
        """型号词 + 场景词 → NORMAL。"""
        result = classify_complexity("会议室P2.5")
        assert result.route == QueryRoute.NORMAL, (
            f"会议室P2.5 应该走 NORMAL，实际: {result.route.value} ({result.reason})"
        )


# ── 回归测试：Golden Dataset 批量验证 ────────────────────────────────────────

def load_golden_dataset():
    """加载 Golden Dataset。"""
    dataset_path = os.path.join(project_root, "eval", "dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_golden_dataset_router_coverage():
    """验证 Golden Dataset 所有 query 都有路由测试覆盖。"""
    dataset = load_golden_dataset()
    queries = {item["id"]: item for item in dataset["queries"]}
    covered = {case["id"] for case in ROUTER_TEST_CASES if case["id"].startswith("q")}

    missing = [qid for qid in queries if qid not in covered]
    assert not missing, (
        f"Golden Dataset 中的以下 query 缺少路由测试覆盖: {missing}\n"
        f"请在 ROUTER_TEST_CASES 中补充。"
    )


# ── 回归测试：关键指标断言 ─────────────────────────────────────────────────────

def test_no_fast_for_scene_recommend():
    """
    回归断言：场景推荐查询（q001-q020 中所有 product_recommendation）
    不应进入 FAST。

    这是 Phase 1 修复的核心目标：原来"会议室P2.5"等场景推荐走 FAST，
    现在应该走 NORMAL 或 AGENT。
    """
    dataset = load_golden_dataset()
    failures = []

    for item in dataset["queries"]:
        if item.get("scenario") == "product_recommendation":
            query = item["query"]
            # 跳过含推理词的（应该走 AGENT）
            reasoning_kw = ["什么", "怎么", "哪个", "怎么选", "够用", "预算"]
            if any(kw in query for kw in reasoning_kw):
                continue
            result = classify_complexity(query)
            if result.route == QueryRoute.FAST:
                failures.append(
                    f"  [{item['id']}] {query!r} → FAST（应为 NORMAL/AGENT）"
                )

    assert not failures, (
        "以下场景推荐 query 被错误路由到 FAST：\n" + "\n".join(failures)
    )
