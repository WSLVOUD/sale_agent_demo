# Orchestrator.py 二次瘦身计划

目标
--------------------------------------------------
当前：src/orchestrator.py ≈ 1544 行

目标：
- 恢复 Orchestrator 的“纯编排”职责
- 清理重新回流的 Dialogue / Multi-Screen / Vision / Response 逻辑
- 保留真正需要的流程控制
- 目标控制在约 600～800 行
- 不追求机械压缩行数，以职责边界为最终标准

核心原则
--------------------------------------------------
1. 不改变现有业务行为
2. 不重写核心功能
3. 不修改 RequirementProfile
4. 不修改 LED / LCD / IFP 分类逻辑
5. 不修改 Vision 图片识别模型和 Pipeline
6. 不修改 Multi-Screen 核心逻辑
7. 不修改 RAG / Recommendation / Calculator
8. 不修改 Sales Agent / Solution Agent 核心流程
9. 每迁移一组逻辑立即运行相关测试
10. 所有旧逻辑确认有唯一新出口后，才能删除旧代码


一、先建立 Orchestrator 职责清单
--------------------------------------------------
对当前 1544 行代码逐函数检查：

> ✅ **Phase 1 已完成（2026-09-28）**
> - 起点：`src/orchestrator.py` = **1536 行**（Plan 写的 1544 行是上一版快照）
> - 盘点结论（`ast` 逐函数 + 全仓调用点扫描）：
>
> | 类别 | 函数 | 结论 |
> |---|---|---|
> | A 真编排 | `process_message`(443 行原始) / `_finalize_turn_response`(370) / `_finish_llm_turn` / `_note_ai_turn` / `_emit_decision_audit` / `_note_customer_turn` / `_requirement_summary` / `_replace_placeholder_history` / `_stored_profile` / `_load_history` | 保留（Phase 4 把其中"决策"部分外移） |
> | B Dialogue 回流 | `_question_candidates` / `_dialogue_action_label` / `_select_turn_action` / `_action_candidates` / `_question_text_for_slot` / `_facts_for_turn` / `_conversation_snapshot` / `_continuation_only_text` / `_neutral_continuations` / `_conversation_state` / `_answer_match_object` / `_profile_slot_map` / `_newly_filled_slots` / `_ranked_question_candidates` / `_apply_duplicate_firewall` | → 迁 `src/dialogue/`（Phase 2 已做） |
> | C Multi-Screen 回流 | `_split_and_apply_screen_specs` … `_multi_item_follow_up`（8 个都是 1~2 行委托） | → 直接调 `MultiScreenManager`（Phase 3） |
> | D Vision 回流 | `_vision_confirmation_sentence` / `_attach_vision_confirmation`（1~3 行委托） | 删除委托，直接调 `ResponseCoordinator`（Phase 4） |
> | E Response 回流 | `_attach_service_faq` / `_compose_with_requirement_question` / `_finalize_turn_response` 里的收口决策序列 | → `ResponseCoordinator`（Phase 4） |
> | F Audit / Logging | `_note_ai_turn` / `_emit_decision_audit` / `_finish_llm_turn` / `_requirement_summary` | 保留调用（不含业务判断） |
>
> 调用点扫描：除 `process_message` / `_response_coordinator` / `_final_response_coordinator`
> 之外，其它 private 方法**没有任何测试或生产代码**从外部调用 → 可以迁移/删除。

A. 真正的流程编排
   → 保留

B. Dialogue 决策 / 提问 / 对话状态
   → 迁移到现有 dialogue 模块

C. Multi-Screen 业务逻辑
   → 使用现有 src/rag/multi_screen.py

D. Vision 业务逻辑
   → 使用现有 src/vision/

E. Response 组装
   → 使用现有 response_coordinator / response 模块

F. Audit / Logging
   → 保留调用，不在 Orchestrator 实现业务判断


二、重点清理 Dialogue 回流
--------------------------------------------------
重点检查并迁移：

> ✅ **Phase 2 已完成（2026-09-28）**：下列方法已从 Orchestrator 迁出并删除，
> 唯一实现落在对话层（`python -m pytest -q` → 850 passed；Orchestrator 1536 → 1255 行）：
>
> | 旧方法 | 新出口 |
> |---|---|
> | `_question_candidates` | `dialogue.final_response.question_candidates_from_result` |
> | `_dialogue_action_label` | `dialogue.action_bridge.dialogue_action_label` |
> | `_select_turn_action` / `_action_candidates` | `dialogue.turn_action.select_turn_action` / `dialogue_action_candidates` |
> | `_question_text_for_slot` | `dialogue.turn_action.question_text_for_slot` |
> | `_facts_for_turn` | `dialogue.grounded_facts.facts_to_dicts` |
> | `_conversation_snapshot` / `_conversation_state` / `_answer_match_object` | `dialogue.conversation_state.conversation_snapshot` / `get_conversation_state` / `last_answer_match_object` |
> | `_continuation_only_text` / `_neutral_continuations` | `dialogue.natural_continuation.continuation_only_text` / `neutral_continuation` |
> | `_profile_slot_map` / `_newly_filled_slots` | `dialogue.profile_slots.profile_slot_map` / `newly_filled_slots` |
> | `_ranked_question_candidates` | `dialogue.policy.ranked_question_slots` |
> | `_apply_duplicate_firewall` | `dialogue.duplicate_firewall.apply_question_firewall` |
>
> 依赖旧方法的 4 条测试按计划"先迁移测试、再删旧函数"迁到新出口
> （`test_action_consistency` / `test_no_stall_after_quality_answer` /
> `test_profile_slots` / `test_v295_led_chain_fixes`）。

_question_candidates()
_dialogue_action_label()
_select_turn_action()
_action_candidates()
_question_text_for_slot()
_facts_for_turn()

_conversation_snapshot()
_continuation_only_text()
_neutral_continuations()

_answer_match_object()
_profile_slot_map()
_newly_filled_slots()
_ranked_question_candidates()
_apply_duplicate_firewall()

目标：

Orchestrator 不再自己实现：
- 下一问选择
- Dialogue Action 判断
- 重复提问判断
- Conversation State 判断
- Question Candidate 排序
- Duplicate Firewall
- Continuation 生成

这些职责只能有一个真正实现。


三、清理 Multi-Screen 回流
--------------------------------------------------
检查：

> ✅ **Phase 3 已完成（2026-09-28）**
> - `MultiScreenManager` 增加对外唯一入口：`split_and_apply_screen_specs` /
>   `maybe_target_screen` / `maybe_start_new_item` / `share_common_facts` /
>   `recommend_all_screens` / `multi_item_follow_up`（实现仍在 `multi_screen.py`）。
> - Orchestrator 的 8 个 1~2 行委托（含 `_screen_pending_block` / `_model_matches_screen`
>   两个零调用死代码）**全部删除**，`process_message` 直接调 `self._multi_screen().xxx(...)`。
> - 验收：`python -m pytest -q` → 850 passed；Orchestrator 1255 → 1232 行。

_split_and_apply_screen_specs()
_maybe_target_screen()
_maybe_start_new_item()
_screen_pending_block()
_share_common_facts()
_model_matches_screen()
_recommend_all_screens()
_multi_item_follow_up()

如果这些逻辑已经由：

src/rag/multi_screen.py

提供，则 Orchestrator 只负责：

收到消息
→ 调用 MultiScreenManager
→ 获取处理结果
→ 继续主流程

禁止在 Orchestrator 再实现第二套多屏逻辑。

注意：
- 不删除 multi_screen.py
- 不改变多屏业务行为
- 不改变“修改第二块屏幕”等现有功能


四、清理 Vision 回流
--------------------------------------------------
重点检查：

> ✅ **Phase 4（Vision 部分）已完成（2026-09-28）**
> - `_attach_vision_confirmation` / `_vision_confirmation_sentence` 两个委托删除；
>   图片核对唯一实现在 `src/dialogue/response_coordinator.py`（`attach_vision_confirmation`），
>   Vision 识别仍由 `src/vision/` 负责（未改动模型与 Pipeline）。
> - 依赖旧方法的两条测试按计划迁到新出口（`tests/test_vision_pipeline.py`）。

_vision_confirmation_sentence()
_attach_vision_confirmation()

以及所有 Vision 判断。

Orchestrator 只负责：

图片
→ 调用 Vision Pipeline
→ 获得结构化结果
→ 合并 RequirementProfile
→ 继续 Sales / Solution 流程

Vision 识别、字段处理、Vision 规则继续由：

src/vision/

负责。

禁止重新实现图片识别逻辑。


五、清理 Response 回流
--------------------------------------------------
重点检查：

> ✅ **Phase 4（Response 部分）已完成（2026-09-28）**
> - Orchestrator 的 `_attach_service_faq` / `_compose_with_requirement_question`
>   委托删除（调用点直接走 `self._response_coordinator()`）。
> - 收口决策段整段迁入对话层：`ResponseCoordinator.prepare_final()`（覆盖率 →
>   重复提问闸门 → Policy 对齐 → 产品类型闸门 → Momentum → 唯一 Action →
>   承接上下文 → 承接额度），返回 `TurnPlan`；
>   `ResponseCoordinator.postprocess_final()` 接管"承接轮去问句 / 去机械话术 /
>   型号闸门"；正文中间件与流水线实现都在 `dialogue/`。
> - 架构契约保持：`orchestrator.py` 仍含 `_response_coordinator().finalize(` 与
>   `_final_response_coordinator().build(`（客户可见文本唯一出口未变）。
> - 验收：850 passed；`_finalize_turn_response` 370 → 72 行。

_attach_service_faq()
_attach_vision_confirmation()
_vision_confirmation_sentence()
_compose_with_requirement_question()

如果现有：

src/dialogue/response_coordinator.py

已经负责这些逻辑：

Orchestrator
→ 调用 ResponseCoordinator
→ 获取最终回复

不要在 Orchestrator 保留第二套回复组装逻辑。


六、保留真正的 Orchestrator 核心
--------------------------------------------------
以下职责原则上继续保留：

1. 接收 TurnExecutor 输入
2. First Contact 判断与调用
3. 调用 Vision Pipeline
4. 调用 Sales Agent
5. 调用 Solution Agent
6. 根据 Agent 返回结果进行流程转发
7. 调用 Multi-Screen Manager
8. 调用 Response Coordinator
9. Turn / Perf / Audit 生命周期管理
10. 最终返回统一结果


七、删除“兼容层膨胀”
--------------------------------------------------
检查所有：

> ✅ **Phase 5 已完成（2026-09-28）**
> - 全仓调用点扫描：Orchestrator 内部已无"零调用"方法（剩余方法都有实际调用方）。
> - 清理迁移后遗留的无用导入（`question_text_for_slot` / `render_minimal` /
>   `continuation_only_text` / `strip_mechanical_phrases` / `DETAILED` /
>   `configured_ack_streak_limit` / `decide_continuation` 等）。
> - 迁移前依赖旧方法的测试共 7 条，先迁移到新出口再删旧函数
>   （`test_action_consistency` / `test_no_stall_after_quality_answer` /
>   `test_profile_slots` / `test_v295_led_chain_fixes` / `test_service_faq` /
>   `test_vision_pipeline`）。

旧方法包装
旧接口兼容函数
已经没有生产调用的 helper
重复的 private method
历史版本遗留代码

原则：

如果：
旧函数
→ 没有生产代码调用
→ 没有测试依赖
→ 已经存在新的唯一出口

则删除。

如果仍有测试依赖：
先迁移测试
→ 再删除旧函数。

不要为了兼容而永久保留两套业务实现。


八、建立“唯一职责”检查
--------------------------------------------------
清理后必须满足：

Question Planner
→ 唯一负责提问规划

Dialogue State
→ 唯一负责会话状态

Duplicate Firewall
→ 唯一负责重复问题控制

MultiScreenManager
→ 唯一负责多屏业务

Vision Pipeline
→ 唯一负责图片需求识别

ResponseCoordinator
→ 唯一负责回复组装

RequirementProfile
→ 唯一需求事实源

RecommendationCoordinator
→ 唯一推荐决策出口

Orchestrator
→ 只负责协调这些模块


九、测试策略
--------------------------------------------------
不要一次性大改。

> ✅ **Phase 6 已完成（2026-09-28）**
>
> | 验证项 | 结果 |
> |---|---|
> | `python -m pytest -q` | **850 passed**（每阶段各跑一次全量：Phase 2 / 3 / 4 / 5 均 850 passed） |
> | `python -m eval.recommendation_eval` | Slot 0.9914 / Hard 0.9896 / ProductType 1.0（与瘦身前一致） |
> | `python -m eval.calculator_eval` | 1.0（14/14） |
> | `python -m eval.retrieval_eval` | 硬约束违规 0 |
> | `tests/architecture/`（70 条边界护栏） | 全绿（唯一出口 / 唯一决策入口 / 层间边界未被破坏） |

按照下面顺序：

Phase 1
→ 函数/调用关系盘点
→ 不修改代码

Phase 2
→ 清 Dialogue 回流
→ pytest

Phase 3
→ 清 Multi-Screen 回流
→ pytest

Phase 4
→ 清 Vision / Response 回流
→ pytest

Phase 5
→ 删除无调用兼容代码
→ pytest

Phase 6
→ 全量测试

最终至少验证：

- First Contact
- LED / LCD / IFP
- Vision 图片识别
- Multi-Screen
- RequirementProfile
- Sales Agent
- Solution Agent
- Recommendation
- Calculator
- RAG
- Duplicate Question Firewall
- Response
- Turn / Audit


十、最终验收标准
--------------------------------------------------
代码层：

[x] orchestrator.py 不再包含独立 Dialogue 决策系统
      （问哪一项 / 唯一动作 / 重复提问 / 承接额度 / 会话状态 全部在 `src/dialogue/`）
[x] 不再包含第二套 Multi-Screen 逻辑
      （8 个委托删除，只调 `MultiScreenManager` 的 6 个公开入口）
[x] 不再包含第二套 Vision 业务逻辑
      （图片核对句只在 `ResponseCoordinator`；识别仍在 `src/vision/`）
[x] 不再包含第二套 Response 组装逻辑
      （决策段 = `ResponseCoordinator.prepare_final`，正文加工 = `postprocess_final`）
[x] 不再维护平行 Requirement 状态
      （`RequirementProfile` 仍是唯一事实源；槽位映射在 `dialogue/profile_slots.py`）
[x] 不再存在无调用的历史兼容代码
      （调用点扫描：零外部调用的旧包装方法已全部删除；无用的导入已清）
[x] process_message() 保持清晰的主流程
      （首接 → Vision → 多屏 → Sales → 分支转发 → 收口；无业务判断）

架构层：

[x] 一个职责只有一个实现（7 个新出口各自唯一，见 §二/§四/§五 的迁移表）
[x] 一个业务只有一个出口（客户可见文本仍只从 FinalResponse 出）
[x] RequirementProfile 仍然是唯一事实源
[x] Recommendation 仍然只有一个决策出口（未改动 recommendation_*）
[x] Vision / Multi-Screen / Dialogue 均通过现有模块处理

功能层：

[x] 不改变现有业务行为（850 条测试逐阶段全绿 + 三个评测数字不变）
[x] 不改变现有 API（API 层未改；ChatResponse 字段与语义不变）
[x] 不改变图片识别模型（未触碰 `src/vision/` 的模型与 Pipeline）
[x] 不改变多屏功能（`multi_screen.py` 只是加了公开入口，逻辑一行未改）
[x] 不改变 LED/LCD/IFP 分类（未触碰 `ProductTypeRouter`）
[x] 不改变推荐和计算逻辑（未触碰 `rag/` / `tools/` 的推荐与计算）

实际结果：

```text
src/orchestrator.py   1536 行  →  874 行（-43%）
_finalize_turn_response  370 行 →  72 行
process_message          443 行 → 450 行（首接 / Vision / 多屏 / 分支转发，纯编排）

注：874 行比"约 600~800"略高，剩余都是 §六 明确保留的职责
（主流程转发 + Turn/Perf/Audit 生命周期 + 访问器）。
若要继续压，只剩 process_message 里 4 个分支的 result 字面量可以抽成构造器 ——
属于"为凑行数而重构"，按计划第十节的说明没有做。
```

> 本轮**未提交**（按用户要求"不要自己提交"）：改动都在工作区，可直接 review。

最终目标：

1544 行
    ↓
清理职责回流
    ↓
约 600～800 行

但最终以“职责单一、没有重复实现”为标准，
不要为了达到 800 行而强行压缩代码。
