# LED RAG v2.4：提问顺序随机化 + 硬性条件复问（实施记录）

> 客户口径（2026-09-20）：
>
> 1. 提问顺序**随机**（不再是固定优先级）；
> 2. 客户说"不知道"或**没回答** → 下一次**不要再问这个问题**，换下一个；
> 3. 全部问题随机问完**一遍**之后，才回头问**硬性条件**里还缺的，并**说明为什么需要知道**；
> 4. 客户的补充确认：**立即推荐只在"客户已经给了硬性条件、并且明确想要推荐"时发生**，
>    不是"硬性条件集齐了就推荐"；
> 5. 随机顺序按"会话种子"洗牌（同一会话内稳定、可复现，跨会话各不相同）。

---

## 一、实施项与完成情况

| # | 实施项 | 状态 | 落地位置 |
|---|---|---|---|
| 1 | 提问顺序随机化（会话种子洗牌，稳定可复现） | ✅ 已完成 | `src/dialogue/question_order.py`（`shuffled_slots` / `next_in_order`）、`src/dialogue/question_flow.py::random_order` |
| 2 | 随机轮内"问过就不再问"（不知道 / 没回答 / 答非所问） | ✅ 已完成 | `question_flow.pass1_pending()`（只取 `ask_count == 0` 且 Action 仍为"该问"的槽位） |
| 3 | 一轮走完 → 只复问**硬性条件**，且说明"为什么需要知道" | ✅ 已完成 | `question_flow.hard_recap_pending()` + `HARD_WHY` + `ResponsePlan.why`（注入销售话术） |
| 4 | 硬性条件"最多两次接触"上限保持不变 | ✅ 已完成 | `question_flow`（`ask_count < MAX_ASKS_PER_SLOT`）+ 既有 `field_policy` 守卫（第二次仍不知道 → DEFERRED） |
| 5 | 立即推荐：只在客户**明确要推荐 / 授权 AI 决定**时 | ✅ 已完成 | 销售节点 `customer_wants_recommendation`（显式请求正则 + `detect_response_intents` 的 DELEGATED） |
| 6 | 软问题（使用场景 / 价位取向）不进入复问 | ✅ 已完成 | `hard_recap_pending()` 只包含 `HARD_SLOTS` |
| 7 | 内容类型仍然不主动问（只记录） | ✅ 已完成 | 提问池 `ASK_POOL` 不含 `content_type` |
| 8 | 裸尺寸（"129,2cm"）方向确认优先于随机轮 | ✅ 已完成 | `question_flow.next_question_plan()` 在该场景直接返回 None，交给 Gate 的 `size_axis` 问句 |
| 9 | 避免"紧接着重复同一问"（档案被重建时也不重复） | ✅ 已完成 | 复问候选把 `last_asked_slot` 排到最后 + 复用 `ConversationState.asked_slots` |
 | 10 | 单元测试与全量回归 | ✅ 已完成 | 新增 `tests/test_v24_question_flow.py`（19 条）+ 更新 5 个旧用例；全量 **1348 passed, 4 skipped** |

---

## 二、规则细节（现在的实际行为）

```text
提问池（随机顺序，按会话洗牌）
    室内外 · 使用场景 · 固装租赁 · 价位取向 · P值 · 观看距离 · 尺寸

Phase 1（随机轮）
    · 每轮从"还没问过"的池子里随机挑一个问
    · 客户说"不知道" / 没回答 / 答非所问 → 这一项本轮不再问，换下一个
    · 能推导的项不问（例如客户给了观看距离 → 不再问 P 值；给了 P 值 → 不再问观看距离）

Phase 2（复问：一轮走完之后）
    · 只问还缺的**硬性条件**：室内外 / 固装租赁 / P值 / 观看距离 / 尺寸
    · 每一问都带一句"为什么需要知道"（英文）
    · 第二次仍拿不到 → DEFERRED：不再问，推荐照常降级；室内外拿不到 → BLOCKED
    · 软问题（使用场景 / 价位取向）客户没答过就不再补问

立即推荐（跳过 Phase 1）
    · 客户明确要推荐："recommend / suggest / 帮我推荐 / 选一款 / 报价…"
    · 客户授权 AI 决定："you decide / 你决定 / 都行 / 你推荐就行"
    · 硬性条件齐 → 直接推荐；还缺 → 进入 Phase 2（问缺的硬性条件，带"为什么"）
    · 只是"把硬性条件凑齐" **不会**触发推荐
```

"为什么需要知道"的口径（英文一句，客户口径：复问硬性条件时才解释）：

| 槽位 | 说明 |
|---|---|
| 室内外 | indoor and outdoor screens use different cabinets and brightness |
| 固装租赁 | fixed and rental cabinets are built differently |
| P值 | the pitch decides how sharp the image looks from where people sit |
| 观看距离 | the viewing distance tells me which pixel pitch is enough |
| 尺寸 | the screen size decides the cabinet and module layout |

---

## 三、实施记录

### 3.1 新增 / 修改

| 类型 | 文件 | 说明 |
|---|---|---|
| 新增 | `src/dialogue/question_flow.py` | 随机轮 / 复问的状态机（`next_question_plan`、`pass1_pending`、`pass1_complete`、`hard_recap_pending`、`HARD_WHY`） |
| 复用 | `src/dialogue/question_order.py` | 顺序洗牌的唯一实现（`shuffled_slots` / `next_in_order`），`random_order()` 直接调它 |
| 修改 | `src/dialogue/question_planner.py` | `QuestionPlan` 增加 `why`；`plan_question()` 支持透传 `easier` / `reason` / `why` |
| 修改 | `src/dialogue/response_planner.py` | 复问时用 Question Flow 给的 `why`（话术里解释"为什么问"） |
| 修改 | `src/dialogue/__init__.py` | 导出随机轮 / 复问 API |
| 修改 | `src/agents/sales/nodes/requirement.py` | 追问改由 Question Flow 决定；"客户明确要推荐 / 授权 AI 决定"时跳过随机轮；裸尺寸方向确认仍交给 Gate |
| 新增 | `tests/test_v24_question_flow.py` | 19 条：随机性 / 会话内稳定 / 一轮不重复 / 可推导项不问 / 复问带 why / 两次上限 / 何时立即推荐 |
| 修改 | `tests/test_requirement_dialogue.py`、`tests/test_unknown_tolerance.py`、`tests/test_soft_questions.py`、`tests/test_new_questions.py`、`tests/test_reply_composer.py` | 断言从"固定顺序 / 集齐即推荐"改为"随机顺序 + 客户要推荐才推荐" |

### 3.2 关键实现点

1. **顺序只有一个出口**：`question_order.shuffled_slots(slots, seed=session_id)`；
   同一会话内顺序稳定（客户不会觉得系统跳来跳去），不同会话顺序不同。
2. **"该不该问"复用 Gate 的 Action**：`question_flow._actions()` 调
   `field_action` + `apply_cross_slot_rules` —— 与 Gate 完全同一套判断，
   所以"P 值可由观看距离推导""给了 P 值就不再问视距"这些规则自动生效，不会出现两套标准。
3. **"问过没答就不再问"双重记账**：
   档案里的 `ask_counts`（权威）+ `ConversationState.asked_slots`（档案被重建时的兜底）。
4. **复问避免连着重复**：复问候选里把上一轮刚问过的那一项排到最后（还有别的可问时）。
5. **为什么需要知道**：`HARD_WHY` → `QuestionPlan.why` → `ResponsePlan.why` →
   销售话术的结构规则（`_plan_rules_for_prompt`），并要求"说明为什么问它有用"。

### 3.3 验证

```text
python -m pytest tests/test_v24_question_flow.py -q          → 19 passed
python -m pytest tests/ -q                                   → 1348 passed, 4 skipped
```

重点回归（全部通过）：

| 验证点 | 覆盖 |
|---|---|
| 顺序随机且会话内稳定 | `test_v24_question_flow.py::TestRandomOrder` |
| 问过 / 不知道 / 没答 → 本轮不再问 | `TestPass1NoRepeat` |
| 一轮走完才复问硬性条件 + 带 why | `TestPass2HardRecap` |
| 集齐不自动推荐、客户要推荐才推荐 | `TestImmediateRecommendation` |
| 旧接口与旧业务语义不变 | 全量 1348 条（含 v2.3.1 的边界不变量测试） |

---

## 四、本次带来的行为变化（需要知情）

1. **室内外不再固定第一个问**：顺序随机，可能先问尺寸 / P 值 / 场景。
   室内外仍然是"不可绕过的硬阻塞"——一轮走完复问一次仍拿不到 → `BLOCKED`。
2. **硬性条件齐了不再自动推荐**：还要把随机轮走完（含使用场景 / 价位取向）。
   客户明确说"给我推荐一个"或"你决定"时才会立刻推荐。
3. **"没回答"也算一次**：以前会再问一遍，现在只留到"一轮走完后的硬性条件复问"；
   复问仍拿不到 → DEFERRED（推荐照常降级）。
4. **每次复问都会说明理由**：例如"the viewing distance tells me which pixel pitch is enough"。

---

## 五、配置与回退

- 想改回"硬性条件齐了就推荐"：在销售节点里把 `customer_wants_recommendation` 的判定改成
  `True`（或在 Gate 加一个开关），随机轮逻辑不用动。
- 想固定调试顺序：调用 `question_order.shuffled_slots(pool, seed_override=<编号>)`，
  或在测试里传固定 `session_id`（顺序可复现）。
- 想调整复问理由：改 `question_flow.HARD_WHY` 一处即可（话术会自动跟随）。
