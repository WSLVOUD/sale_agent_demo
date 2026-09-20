# LED RAG v2.3 实施报告

> 计划文档：`LED_RAG_v2.3_综合优化方案.md`
> 实施日期：2026-09-20
> 提交：`69e6cfb feat(v2.3): 架构收敛 + 决策可靠性 + 对话自然度（按 v2.3 综合优化方案实施）`
> 验证：`python -m pytest tests/ -q` → **1310 passed, 4 skipped**（实施前 1261 passed / 4 skipped，本次新增 49 条测试）

本报告逐项说明计划里的每一条是如何完成的：计划要求 → 实际做法 → 关键文件与函数 → 怎么验证。

---

## 1. 改动地图（先看这张表）

| 层 | 模块 | 文件 | 职责 |
|---|---|---|---|
| 工程规则 | 常量/推导/窗口/几何/约束 | `src/engineering/*.py` | 距离→点间距、屏体几何、亮度、场地折算，**只有一份** |
| 工程规则 | 来源与守卫 | `src/engineering/provenance.py` | `FieldValue` + `check_provenance()` |
| 工程规则 | 冲突 | `src/engineering/conflicts.py` | 行内冲突检测（屏比房间大、室内+P10…） |
| 工程规则 | 需求分类 | `src/engineering/requirement_classes.py` | A/B/C（physical / constraint / semantic） |
| 决策 | 推荐唯一出口 | `src/rag/recommendation_coordinator.py` | 冲突 → Gate → 推导 → 来源守卫 → 引擎 → 校验 → 审计 |
| 决策 | 兼容层 | `src/rag/recommendation_service.py` | 保持历史返回字段，内部走协调器 |
| 决策 | Gate | `src/rag/readiness.py` | 新增 `CONFLICT` 状态（冲突未解决不推荐） |
| 决策 | 入档闸门 | `src/rag/fact_validation.py` | 字段/单位/范围/来源/覆盖权限校验 |
| 观测 | 决策审计 | `src/observability/decision_log.py` | 每条推荐一条 `[DecisionAudit]` JSON |
| 对话 | 提问唯一出口 | `src/dialogue/question_planner.py` | 换说法、不重复、降门槛 |
| 对话 | 表达结构 | `src/dialogue/response_planner.py` | ASK / RECOMMEND / CONFLICT / UNKNOWN 结构 |
| 对话 | 会话表达状态 | `src/dialogue/conversation_state.py` | 上一轮问了什么、答了什么、是否重复 |
| 对话 | 回复组装 | `src/dialogue/response_coordinator.py` | 售后口径 + 图片核对（从 Orchestrator 抽出） |

---

## 2. 逐项实施说明

### §3.1 RequirementProfile 成为唯一事实源 ✅

**计划要求**：需求只保留一个事实源，其他模块不得维护平行状态。

**实际做法**：`RequirementProfile` 就是唯一事实源，它同时承载：

- 字段值 + `sources`（`explicit` / `scenario_derived` / `vision_explicit` / `vision_inferred` / `inferred` / `default`）
- `field_decisions`（`MISSING / CONFIRMED / INFERRED / UNKNOWN / DELEGATED / DECLINED / DEFERRED`）
- `conflicts` + `conflict_slots`
- `ask_counts` / `unknown_reasons` / `last_asked_slot`

其它模块（引擎、Gate、对话层）都改成**读档案**，不再自己算一套：

- `src/rag/recommendation_coordinator.py`：只接收 `profile`，不接 legacy dict
- `src/agents/sales/nodes/requirement.py`：legacy `requirements` 只是档案的只读投影（v2.0 起已如此）
- `src/observability/decision_log.py`：审计里记录的 facts / decisions 都直接取自档案

**怎么验证**：`tests/test_v23_invariants.py::TestInferredStaysInferred`（档案状态与来源一致）、
`tests/test_state_unification.py`（旧字段只是投影）。

---

### §3.2 统一字段数据结构 ✅

**计划要求**：统一字段命名（不要 `distance` / `viewing_distance` / `viewing_distance_m` 三种写法混用），
并给字段一个统一结构 `FieldValue(value, unit, source, status, confidence, source_fields, formula_id)`。

**实际做法**

1. **命名统一**：`src/models/requirement.py::canonical_slot()` 把所有历史写法归一到规范名
   （`distance` / `viewing_distance` → `viewing_distance_m`，`target_width_mm` → `width`，`pixel_pitch_mm` → `pixel_pitch`），
   `ask_counts` / `unknown_reasons` 只写规范键。
2. **结构统一**：新增 `src/engineering/provenance.py`：

```python
FieldValue(
    value=8, unit="m", source="derived", status="INFERRED",
    confidence=0.65, source_fields=("audience",), formula_id="GEOMETRY_TO_VIEWING_DISTANCE_V1",
)
```

`build_provenance(profile, technical)` 会给关键参数（环境、固装租赁、点间距、观看距离、屏体尺寸）
各生成一条 `FieldValue`，`to_dict()` 直接进审计日志。

**怎么验证**：`tests/test_requirement_decision_state.py::test_slot_alias_writes_one_canonical_key`、
`tests/test_v23_invariants.py::TestProvenanceInvariant::test_derived_pitch_carries_formula_id`。

---

### §4 统一 Engineering Rule ✅

**计划要求**：距离→点间距的规则只能有一处，避免"同一输入不同结果"。

**实际做法**：新建 `src/engineering/`，把原来散在 `parameter_inference.py` 的规则整段搬进去：

| 文件 | 内容 |
|---|---|
| `constants.py` | `VIEWING_DISTANCE_PITCH_TABLE` / `INDOOR_PITCH_TABLE` / `OUTDOOR_PITCH_TABLE` / `FALLBACK_PITCH_BAND` / `BRIGHTNESS_BY_ENVIRONMENT` / 座位与屏高系数 / 1mm≈3m 系数 |
| `viewing_distance.py` | `estimate_viewing_distance()`（人数 / 面积 / 进深 / 屏尺寸 → 距离区间）、`parse_distance()`、`environment_from_facts()` |
| `pitch_window.py` | `preferred_pitch_for_environment()`（业务表）、`pitch_window_for_distances()`（物理窗口）、`pitch_range_for_distance()`、`fallback_pitch_band()` |
| `screen_geometry.py` | `screen_dims_m()`、`screen_size_for_distance()`、`suggest_screen_size()`（授权尺寸时的参考尺寸） |
| `constraints.py` | `brightness_range_for_environment()`、`rental_from_facts()` |

`parameter_inference.py` 现在只做协调（见 §12），规则链固定为：

```text
Customer Facts → Physical Derivation → Viewing Distance → Pitch Window
    → Product Constraints → Recommendation
```

**怎么验证**：`tests/test_viewing_distance_derivation.py`（45 条：解析、推导、窗口、端到端 6 组输入）、
全量回归（1261 条旧用例未受影响）。

---

### §5 Recommendation Provenance Guard ✅

**计划要求**：任何最终推荐参数都要有来源；没有合法 provenance → 不推荐。

**实际做法**（`src/engineering/provenance.py`）

```python
LEGAL_SOURCES = ("customer", "confirmed", "derived", "inferred", "vision")

def check_provenance(profile, technical, required=("pixel_pitch_mm", "environment")) -> ProvenanceReport
```

- `slot_provenance()` 把档案的 `sources` 映射成合法来源：`explicit/confirmed/vision_accepted → customer/confirmed`、
  `scenario_derived → derived`、`vision_* → vision`、`inferred/default → inferred`；
  **没有来源标记就是没有 provenance**（不会"猜"成 derived）。
- `build_provenance()` 给推导值补上 `formula_id`：
  `GEOMETRY_TO_VIEWING_DISTANCE_V1`、`DISTANCE_TO_PITCH_WINDOW_V1`、
  `ENVIRONMENT_DISTANCE_TO_PITCH_TARGET_V1`、`ENVIRONMENT_FALLBACK_PITCH_BAND_V1`。
- `RecommendationCoordinator` 在调引擎前调用它；不通过就返回 `REJECTED`，**一个型号都不产出**。
- 例外（写成代码注释）：客户**点名型号/系列**时，来源就是客户那句话，不再要求工程参数来源。

**怎么验证**：`tests/test_v23_invariants.py::TestProvenanceInvariant`（无来源 → REJECTED；推导值带公式号；
客户给的 P 值来源是 customer）。

---

### §6 Recommendation 统一出口 ✅

**计划要求**：Fast / Normal / Agent / Sales / Solution / API 都走同一个推荐出口，不允许各自实现推荐算法。

**实际做法**：新增 `src/rag/recommendation_coordinator.py`：

```text
RequirementProfile
   → Conflict 检查（§9）
   → Recommendation Ready Gate
   → Engineering Derivation（src/engineering）
   → Provenance Guard（§5）
   → RecommendationEngine
   → Validation（violations）
   → Final Recommendation + DecisionAudit（§16）
```

返回 `RecommendationOutcome(status, result, gate, provenance, audit, next_question,
missing_fields, reject_reasons, conflicts)`；`status ∈ {RECOMMENDED, DEGRADED,
NEED_CLARIFICATION, CONFLICT, REJECTED}`。

`RecommendationService` 变成兼容层：历史字段（`recommendation_status` / `recommendation_basis` /
`gate` / `unknown_requirements`）保持不变，同时透出 `coordinator_status` / `provenance` /
`decision_audit` / `reject_reasons` / `conflicts`。

**怎么验证**：`tests/test_recommendation_guard.py`（绕过 Gate 也推荐不了）、
`tests/test_recommendation_gate_v21.py`、`tests/test_v23_invariants.py`。

---

### §7 Recommendation Gate 与 Calculation Gate 分离 ✅

**计划要求**：两条 Gate 明确独立——可以"能推荐产品，但不能做完整箱体计算"。

**实际做法**：两条 Gate 一直是独立函数，本次把边界写清楚并补测试：

- 推荐 Gate 只看"选型必需的字段"（环境 / 固装租赁 / 点间距 / 尺寸 / 反推点间距用的观看距离）
- 计算 Gate 只看"尺寸是否够算箱体"：宽高齐 → READY；客户授权 AI 决定尺寸 → 用观看距离推导参考尺寸 → READY；
  尺寸 DEFERRED → `status="DEFERRED"`、`next_question=None`（只挂计算，推荐照常）

**与计划举例的差异（已记录）**：计划举例"室内会议室 + 观看距离 8m → 推荐 READY / 计算 NOT_READY"。
但客户口径把**尺寸**列为硬性条件，所以"从没问过尺寸"时两条 Gate 都会继续问；
只有客户明确给不出来（`DEFERRED`）时，才出现"推荐 READY / 计算 DEFERRED"。

**怎么验证**：`tests/test_v23_invariants.py::TestCalculationInvariant`、
`tests/test_deferred_fields.py::TestDeferredOnlyBlocksDependentActions`。

---

### §8 A/B/C 需求模型 ✅

**计划要求**：把客户说法分成 physical / constraint / semantic 三类；**C 类不得直接产生具体 P 值**。

**实际做法**：新增 `src/engineering/requirement_classes.py`

```python
@dataclass
class RequirementClassification:
    physical: List[str]     # 50 people / 25㎡ / 10x5m / 2000 nit …
    constraint: List[str]   # 4K / no flicker / waterproof / small text …
    semantic: List[str]     # better effect / premium / you decide …

def classify_requirement(text) -> RequirementClassification
def semantic_only(text) -> bool
```

分类结果用于：审计日志（一眼看出客户这句话给了哪类信息）与不变量测试。
C 类说法的处理仍在既有链路里：走"澄清"或"保守默认 + INFERRED"（点间距永远带公式号，不会凭空来一个 P 值）。

**怎么验证**：`tests/test_v23_invariants.py::TestSemanticClassInvariant`（semantic-only 句子没有 physical 值；
纯语义输入只能得到带来源的环境兜底档）。

---

### §9 Conflict 独立状态 ✅

**计划要求**：矛盾信息不要当 UNKNOWN，要单独立 `CONFLICT`，冲突未解决前禁止推荐。

**实际做法**

1. 新增 `src/engineering/conflicts.py`：

```python
@dataclass(frozen=True)
class EngineeringConflict:
    slot: str; message: str; evidence: str

def detect_engineering_conflicts(profile) -> List[EngineeringConflict]
def conflict_slots(profile) -> List[str]
def conflict_message(profile) -> Optional[str]   # 客户可读的英文澄清句
```

目前覆盖：屏体面积 > 场地面积、面积反推进深 < 1m、进深 < 屏高、
室内却要 P8+ 而观看距离并不远（< 30m）。

2. `src/rag/readiness.py`：Gate 第 0 步先看冲突 ——
   `status="CONFLICT"`、`blocked_slots=[冲突字段]`、`next_question` 是澄清问句（不是推荐）。
3. `RecommendationCoordinator`：冲突 → `CONFLICT`，不调引擎、不产出型号。

**怎么验证**：`tests/test_v23_invariants.py::TestConflictInvariant`（25㎡ 房间 + 50㎡ 屏 → CONFLICT 且无推荐）、
`tests/test_v23_negative.py::TestConflictingInputs`（室内+P10、进深<屏高）。

---

### §10 Question Planner 成为唯一提问出口 ✅

**计划要求**：Gate → Question Planner → Response Planner → LLM；一次一个问题、不重复问已确认信息、
客户说不知道时降门槛、重复 UNKNOWN 后允许 DEFERRED。

**实际做法**：新增 `src/dialogue/question_planner.py`

```python
@dataclass
class QuestionPlan:
    slot: str; question: str; action: str; easier: bool; reason: str; reused: bool

def plan_question(decision, profile, *, language, seed, session_id, conversation,
                  slot=None, question=None) -> QuestionPlan
```

规则：

- 调用方已调整过槽位（例如"这一项刚在图片核对里说过，改问下一项"）→ 以调用方为准（`slot` / `question` 覆盖参数）
- 同一句话术本会话里问过 → 自动换一种说法（`question_for(slot, seed+offset)`），并标记 `reused=True`
- 客户说过"不知道"的字段 → 用降门槛问法（`easier=True`）
- Gate 已经在做"问满两次就不问"，Question Planner 不重复这套判断，只负责措辞

`src/agents/sales/nodes/requirement.py` 现在把追问统一交给它，并把 `response_plan` 写进 state、
把本轮问答写进会话状态：

```python
question_plan = plan_question(decision, profile, seed=_turn_seed, session_id=_session_id,
                              conversation=conversation, slot=slot, question=question)
response_plan = plan_response(decision, profile=profile, question_plan=question_plan)
state["response_plan"] = response_plan.to_dict()
conversation.note_turn(answer=current_msg_text, question=question, slot=slot,
                       action=response_plan.action, intent=current_intent,
                       stage=stage_from_status(decision.status))
```

**怎么验证**：`tests/test_v23_dialogue.py::TestQuestionPlanner`（换说法、覆盖生效、空决策不出问句）、
`tests/test_soft_questions.py`（提问顺序仍为 环境→场景→安装→价位→P值→尺寸）、
`tests/test_unknown_tolerance.py`（第一次"不知道"换下一问、最后一轮再问、第二次 DEFERRED）。

---

### §11 Vision 统一进入需求链 ✅

**计划要求**：Vision 只能作为输入源；图片结果标记 `source=vision`、`status=INFERRED`；
尤其点间距 / 尺寸 / 环境 / 观看距离不能默认等同客户确认。

**实际做法**

- `src/vision/integration.py`：每个 Vision 字段在合并前先过 `validate_incoming_facts(..., source="vision")`，
  荒谬值（例如 `pixel_pitch_mm=9999`）直接丢弃；客户已确认的值永远优先，冲突照旧记录到 `conflicts` / `conflict_slots`。
- `src/engineering/provenance.py`：`vision_explicit` / `vision_inferred` → `source="vision"`、`status="INFERRED"`，
  不会显示成 customer。
- 图片结论仍然走"先跟客户核对"（`vision_confirmation_pending` + ResponseCoordinator 的核对句）。

**怎么验证**：`tests/test_v23_negative.py::TestVisionBadParams`（荒谬值被丢弃、Vision 点间距来源是 vision）、
`tests/test_vision.py` + `tests/test_vision_pipeline.py`（114 条既有用例）。

---

### §12 拆分 parameter_inference.py ✅

**计划要求**：`parameter_inference.py` 只负责协调调用，规则搬到 `src/engineering/`。

**实际做法**：机械搬迁（用一次性脚本删掉 377 行规则定义，改成从 `src.engineering` 导入并保持名字兼容）。
现在 `infer_technical_parameters()` 的流程是：

```text
facts → estimate_viewing_distance()（没有客户明确距离时）
      → preferred_pitch_for_environment()（业务表）
      → pitch_window_for_distances()（物理窗口，取交集）
      → fallback_pitch_band()（连距离都推不出来时的保守档）
      → brightness / installation 约束与 source 标记
```

对外名字（`parse_distance` / `estimate_viewing_distance` / `pitch_window_for_distances` …）保留，
已有 import 不会断。

**怎么验证**：全量 1310 条用例通过；`tests/test_viewing_distance_derivation.py` 覆盖搬迁后的行为。

---

### §13 拆分 orchestrator.py ✅

**计划要求**：Orchestrator 只负责流程编排，具体业务逻辑拆到协调器。

**实际做法**：新增 `src/dialogue/response_coordinator.py`

```python
class ResponseCoordinator:
    def attach_service_faq(self, response, message) -> str      # 售后口径（先清矛盾句，再接标准回答）
    def attach_vision_confirmation(self, response, session_id, message) -> str
    def finalize(self, response, *, session_id, message) -> str  # 固定顺序：FAQ → 图片核对
```

`src/orchestrator.py` 的 `_finalize_turn_response()` 现在只有一行委托：

```python
text = self._response_coordinator().finalize(str(result.get("response") or ""),
                                            session_id=session_id, message=message)
```

（旧的 `_attach_service_faq` / `_attach_vision_confirmation` 方法保留，内部行为一致，避免破坏既有测试。）

**说明**：本次只抽出"回复组装"和"推荐链路"两块（收益最大、风险最低），其余零散逻辑保持原样——
一次性大改会让 1261 条既有用例的语义漂移。

**怎么验证**：`tests/test_service_faq.py`、`tests/test_service_faq_consistency.py`、
`tests/test_v23_dialogue.py::TestResponseCoordinator`（含"Orchestrator 已委托"的源码断言）。

---

### §14 统一 Validation ✅

**计划要求**：LLM / Vision / Rewrite 的输出都先过 Validation，禁止 LLM JSON 直接改核心状态。

**实际做法**：新增 `src/rag/fact_validation.py`

```python
KNOWN_SLOTS / MARKER_KEYS     # 字段白名单
RANGES   # viewing_distance_m 0.3~200m；pixel_pitch_mm 0.3~20mm；target_width_mm 100~200000mm …
ENUMS    # environment / installation / display_type / content_type / budget_level / size_axis

@dataclass
class FactValidation:
    ok: bool; accepted: dict; rejected: dict[str, str]; notes: list[str]

def validate_incoming_facts(facts, *, source, profile=None) -> FactValidation
def validate_and_filter(facts, *, source, profile=None) -> dict
```

校验六件事：字段是否合法、单位/数值范围、枚举取值、来源是否存在、
是否与已有事实冲突（低优先级来源不能覆盖客户确认值）、是否允许更新。

接入点：

- `src/core/requirement_extractor.py`：合并规则 + LLM 结果后、建档案前过滤（非法字段直接丢，日志告警）
- `src/vision/integration.py`：Vision 字段逐个过滤

**怎么验证**：`tests/test_v23_negative.py::TestInvalidUnitsAndRanges`（7 组越界/错枚举 + 未知字段 + 覆盖客户值）、
`tests/requirement_extraction/*`（177 条既有抽取用例全部通过）。

---

### §15 统一产品数据来源 ✅

**计划要求**：产品参数不要同时维护在 TXT / JSON / Python 常量 / 推荐代码里。

**实际做法**：产品数据的读取入口只有 `src/rag/json_loader.py`
（`load_structured_products` / `load_canonical_models` / `canonical_model_index`），
引擎与检索都从它取；并用**不变量测试**锁死这条规则：

```python
# tests/test_v23_invariants.py::TestProductDataSourceInvariant
def test_only_json_loader_defines_product_data_reads(self):
    # 扫描 src/：只有"既提到产品数据文件名、又真的在读它"的行才算违规
    # （注释/文档字符串里提到文件名不算）
```

另有一条测试确认引擎确实用 loader 的模型（`RecommendationEngine(models=load_canonical_models(...))`）。

**怎么验证**：`tests/test_v23_invariants.py::TestProductDataSourceInvariant`（2 条）。

---

### §16 增加决策审计日志 ✅

**计划要求**：每次推荐记录 Turn ID / 客户输入 / 抽取事实 / 字段决策 / 推导参数 / 约束 /
Gate 决策 / 候选 / 被拒原因 / 选中型号 / 校验结果，能回答"为什么是这个型号"。

**实际做法**：新增 `src/observability/decision_log.py`

```python
@dataclass
class DecisionRecord:
    turn_id, session_id, customer_input, extracted_facts, field_decisions,
    derived_parameters, constraints, gate_decision, candidate_models,
    rejected_models, selected_model, provenance, validation_result,
    final_response, created_at

def record_from_recommendation(profile, *, session_id, customer_input, technical,
                              result, provenance, gate, validation, final_response)
    → .emit()  # logger.info("[DecisionAudit] {json}")
```

`RecommendationCoordinator` 对**每一种**结果都发一条审计（包括 `NEED_CLARIFICATION` /
`CONFLICT` / `REJECTED`），所以"为什么没推荐""为什么被拒"同样可追溯。

**怎么验证**：`tests/test_v23_invariants.py`（REJECTED / CONFLICT 的 outcome 带 audit，
`audit["selected_model"] == ""`；正常推荐时 audit 里有选中型号与公式号）。

---

### §17 测试升级 ✅

**计划要求**：不只增加数量，而是增加 Invariant / Negative / Regression 三类。

**实际做法**：新增 3 个文件共 **49 条**：

| 文件 | 条数 | 内容 |
|---|---|---|
| `tests/test_v23_invariants.py` | 15 | 来源守卫、冲突阻断、INFERRED 不升 CONFIRMED、两条 Gate 独立、产品数据单源、C 类不产 P 值 |
| `tests/test_v23_negative.py` | 19 | 室内+P10、25㎡+50㎡屏、进深<屏高、错单位/越界、缺视距、Vision 错参数、客户反复改需求 |
| `tests/test_v23_dialogue.py` | 15 | 四种表达策略、提问不重复/可覆盖、会话状态、ResponseCoordinator、Orchestrator 已拆分 |

Regression 部分沿用既有金标与历史 bug 用例（例如"100 人却推 P1.2"、"i dont konw 被反复追问"、
"包安装"自相矛盾），它们继续在同一个全量里跑。

**怎么验证**：`python -m pytest tests/ -q` → **1310 passed, 4 skipped**。

---

### §18 RAG 暂时不要大改 ✅（遵守）

**计划要求**：本次不要优先改 embedding / chunk / top-k / reranker。

**实际做法**：`src/rag/retriever.py` / `fusion.py` / `bm25.py` / `sparse.py` / `rerank.py` /
`vector_store.py` **一行未改**；本次改动集中在决策层、校验层与对话层。

---

### §19 AI 话术僵硬专项改造 ✅

**计划要求**：不要让 Gate 直接吐模板；建立 ResponsePlanner，Python 决定"做什么"，LLM 决定"怎么说"。

**实际做法**：新增 `src/dialogue/response_planner.py`

```python
@dataclass
class ResponsePlan:
    action: str            # ASK / RECOMMEND / CONFLICT / UNKNOWN
    slot: str
    strategy: str
    blocks: List[str]      # 这一轮的表达结构
    why: str               # 为什么问这一项（英文半句）
    question: str
    one_question: bool = True
    expand_only_when_asked: bool = True
    instructions: List[str]

def plan_response(decision, *, profile, question_plan, unknown, session_id) -> ResponsePlan
```

`why` 内置一句一字段的口径（例如视距 → "the viewing distance decides which pixel pitch is enough"），
只在有用时解释；`prompt_lines()` 会把结构 + 行为规范交给 Sales LLM。

**怎么验证**：`tests/test_v23_dialogue.py::TestResponseStrategies`（4 种策略 + 提示词里包含
"ONE question" / "field names" 等硬性规范）。

---

### §20 增加 Conversation State ✅

**计划要求**：保存当前对话表达状态（阶段 / 上一轮意图 / 上一轮动作 / 上一句问题 / 客户回答 /
重复次数），避免"客户刚答完 8m，下一轮又问视距"。

**实际做法**：新增 `src/dialogue/conversation_state.py`

```python
@dataclass
class ConversationState:
    session_id, conversation_stage, last_customer_intent, last_action,
    last_question, last_slot, last_answer, repeated_question_count,
    asked_questions, asked_slots

    def note_turn(...)              # 每轮更新；同一句话重复出现时 repeated_question_count += 1
    def asked_before(question)      # 最近 6 句里问过吗
    question_is_repeating           # 是否连着重复

get_conversation_state(session_id) / reset_conversation_state(session_id)
# 进程内按 session_id 存储，最多 256 个会话（LRU 淘汰）
```

注意：**这不是 Memory**（不落库、不参与需求事实），只影响"这一轮怎么说"。

**怎么验证**：`tests/test_v23_dialogue.py::TestConversationState`（记录、重复检测、会话隔离）。

---

### §21 Response Strategy ✅

**计划要求**：ASK / RECOMMEND / CONFLICT / UNKNOWN 各有表达结构。

**实际做法**（`response_planner.py` 的 `blocks`）

| Action | 结构 |
|---|---|
| ASK | `ACKNOWLEDGE → EXPLAIN_WHY → ASK_ONE_QUESTION` |
| RECOMMEND | `SHORT_SUMMARY → RECOMMEND → KEY_REASON` |
| CONFLICT | `ACKNOWLEDGE → EXPLAIN_CONFLICT → ASK_ONE_CLARIFICATION` |
| UNKNOWN | `REASSURE → OFFER_SIMPLE_ALTERNATIVE → ASK_ONE_QUESTION` |

**怎么验证**：`tests/test_v23_dialogue.py::TestResponseStrategies`（逐条断言 blocks 顺序）。

---

### §22 Sales Prompt 重构 ✅

**计划要求**：把"IF indoor 就说 XXX"的模板改成行为规范。

**实际做法**：在 `src/agents/sales/nodes/script_generator.py` 里

1. 新增 `_plan_rules_for_prompt(state)`：把 `ResponsePlan.blocks` + `why` 翻译成一条结构规则；
2. `_QUESTION_POLISH_PROMPT` 增加第 9 条 `{plan_rules}`，与既有硬性规则并列：
   一次一个问句、不重复、不复述需求清单、不新增参数/型号/价格、不暴露内部字段名、不像问卷；
3. `ResponsePlan.prompt_lines()` 里也带同一套行为规范，两条路径一致。

**怎么验证**：`tests/test_v23_dialogue.py::TestResponseStrategies::test_prompt_lines_never_expose_internal_names`、
`tests/test_question_phrasing.py`（既有话术轮换/不重复用例）。

---

### §23 推荐话术优化 ✅

**计划要求**：第一轮短，客户追问再展开技术依据。

**实际做法**：`ResponsePlan(expand_only_when_asked=True)` 会给 LLM 明确指令
"Keep the first recommendation short; expand the technical detail only if the customer asks why"；
推荐话术本身仍是既有口径（一个回复只给一个型号、两种箱体拼法都给、不提价格、不主动提质保）。

**怎么验证**：`tests/test_v23_dialogue.py::TestResponseStrategies::test_recommend_strategy_is_short_first`、
`tests/test_sizing_and_expression.py`（两种拼法 / 不提价格）。

---

### §24 开发顺序 ✅

| 阶段 | 内容 | 状态 |
|---|---|---|
| 第一阶段 决策层收敛 | 唯一事实源、字段结构、Pitch 规则、Provenance、统一出口、Gate 关系、Conflict | ✅ §3~§9 |
| 第二阶段 代码重构 | 拆 parameter_inference、拆 orchestrator、清理 Legacy、产品数据源、Validation、异常处理 | ✅ §12~§15（Legacy Adapter 保留但只读；异常处理沿用既有 try/except + 审计） |
| 第三阶段 自然对话 | ResponsePlanner、ConversationState、ResponseStrategy、Prompt 重构、承接、重复控制、推荐话术 | ✅ §19~§23 |
| 第四阶段 Vision | VisionExtractionResult、Evidence Validation、Vision→Profile、Vision Provenance、Vision Confirmation | ✅ §11（既有 VisionExtractionResult/确认链路 + 本次补 Validation 与 provenance） |
| 第五阶段 测试与稳定化 | Invariant / Negative / Regression / Decision Log / 性能 / 线上异常 | ✅ §16~§17（性能与线上异常观察不在本地范围） |

---

### §25 暂时不要做 ✅（遵守）

本次**没有**做：增加更多 Agent、继续堆关键词、大规模改 RAG、继续加 Pitch 特殊规则、
让 LLM 决定产品参数、让不同 Agent 各自推荐、用大量固定模板解决话术、推倒 LangGraph 重写。

---

### §26 最终验收标准 ✅

| 验收项 | 要求 | 证据 |
|---|---|---|
| 需求理解 | 输入分成 Physical / Constraint / Semantic | `engineering/requirement_classes.py` + `TestSemanticClassInvariant` |
| 参数推导 | Customer Fact → Engineering Rule → Derived Value → 有 provenance | `engineering/*` + `check_provenance` + `test_derived_pitch_carries_formula_id` |
| 推荐 | 无依据不推荐 / 有冲突不推荐 / 满足 Gate 才走统一出口 / 能解释为什么选 | `RecommendationCoordinator` + `test_pitch_without_source_is_rejected` + `test_conflict_blocks_recommendation` + `decision_log` |
| 对话 | 自然承接、只问一个关键问题、不重复、不暴露内部字段、自然推进 | `dialogue/*` + `test_v23_dialogue.py`（15 条） |
| Vision | 图片 → Vision → Evidence → Validation → Profile → 同一套决策链 | `vision/integration.py`（过 Validation）+ `TestVisionBadParams` |

---

### §27 核心原则 ✅

1. **RequirementProfile 是唯一事实源** —— 所有模块读它，不维护平行状态。
2. **决策层 deterministic、可验证、有来源** —— 规则集中在 `src/engineering/`，每个推导值带 `formula_id`。
3. **Python 决定做什么，LLM 决定怎么说** —— Gate/Planner 决定 action 与结构，LLM 只润色话术。
4. **解析层可以越来越强，决策层不能越来越乱** —— 解析层新增能力（人数/面积/进深、拼写容错）都折成物理量，决策层规则未增加分支。
5. **不堆功能，让边界清晰、结果稳定、行为自然** —— 本次没有新增 Agent、没有大改 RAG。

---

## 3. 文件清单

### 新增（16 个源码文件 + 3 个测试文件）

```text
src/engineering/__init__.py            src/engineering/constants.py
src/engineering/viewing_distance.py    src/engineering/pitch_window.py
src/engineering/screen_geometry.py     src/engineering/constraints.py
src/engineering/provenance.py          src/engineering/conflicts.py
src/engineering/requirement_classes.py
src/dialogue/__init__.py               src/dialogue/conversation_state.py
src/dialogue/question_planner.py       src/dialogue/response_planner.py
src/dialogue/response_coordinator.py
src/rag/fact_validation.py             src/rag/recommendation_coordinator.py
src/observability/decision_log.py
tests/test_v23_invariants.py           tests/test_v23_negative.py
tests/test_v23_dialogue.py
```

### 修改（9 个文件）

```text
src/rag/parameter_inference.py        规则搬走，只做协调（-377 行规则定义）
src/rag/readiness.py                  Gate 新增 CONFLICT（工程冲突 + 澄清问句）
src/rag/recommendation_service.py     变成统一出口的兼容层
src/core/requirement_extractor.py     入档前过统一 Validation
src/vision/integration.py             Vision 值过 Validation，客户值优先
src/agents/sales/nodes/requirement.py 追问走 Question Planner + 写会话状态
src/agents/sales/nodes/script_generator.py 注入 ResponsePlan 结构规则
src/orchestrator.py                   回复组装委托给 ResponseCoordinator
README.md / LED_RAG_v2.3_综合优化方案.md  文档同步（含 ✅ 标注与 §28 实施记录）
```

---

## 4. 如何复现验证

```bash
# 全量（约 6 分钟）
python -m pytest tests/ -q

# 只跑 v2.3 新增的三类测试
python -m pytest tests/test_v23_invariants.py tests/test_v23_negative.py tests/test_v23_dialogue.py -q

# 决策链路的重点回归
python -m pytest tests/test_recommendation_guard.py tests/test_recommendation_gate_v21.py \
                 tests/test_viewing_distance_derivation.py tests/test_unknown_tolerance.py -q
```

---

## 5. 已知限制与后续建议

1. **§7 的口径差异**：尺寸在客户口径里是硬性条件，所以"推荐 READY / 计算 DEFERRED"只在尺寸
   被客户明确放弃（DEFERRED）时出现。若希望"尺寸没问到也能先推荐范围"，需要改的是
   `field_policy` 里 `size` 的 `hard_condition` 标记（一行配置 + 回归）。
2. **§13 的拆分只做了两块**：`ResponseCoordinator` 与 `RecommendationCoordinator`。
   若要继续拆，建议优先拆 `orchestrator` 里"多屏拆分 / 项目条目"与"首次接待"两段。
3. **Conversation State 是进程内的**：重启后丢失（只影响话术去重，不影响需求事实）。
   若要持久化，建议放到 Memory 模块（本次计划明确排除 Memory）。
4. **审计日志目前走 logger**：如需落盘 / 检索，可在 `DecisionRecord.emit()` 里加一个 sink
   （JSONL 文件或数据库），不影响现有调用方。
