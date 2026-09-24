# 行为基线（Phase 0）

> 目的：把"现在系统怎么表现"钉死，瘦身前后逐条对比。
> 采集方式：**真 LangGraph + 真 runner**，LLM 用测试里的 canned 响应（本机无外网）。
> 所以记录的是**结构与流程**（类型判断、下一问、路由），措辞由生产 LLM 决定。

## 1. 关键对话样例（当前行为）

格式：`客户 → 产品类型判断(状态) / 下一问 / AI 回复节选`

### S1 单轮：模糊需求

```text
客户: i need a display
  类型=UNKNOWN/UNKNOWN  下一问=environment
  AI: Alright. Are you looking for an LED display, or an LCD such as a video wall…?
```

### S2 四轮需求序列

```text
客户: i need a display        → UNKNOWN/UNKNOWN  下一问=environment   （问 LED/LCD）
客户: indoor                  → UNKNOWN/UNKNOWN  下一问=pixel_pitch   （信息已入档，仍先定类型）
客户: 3m x 5m                 → UNKNOWN/UNKNOWN  下一问=viewing_distance
客户: viewing distance 8m     → LED/INFERRED     下一问=purpose       （≥6m 远距离 → 倾向 LED，要客户确认）
```

> 口径说明（计划 §五/§六）：类型没确认前只问类型；客户给的需求信息照常入档，不会因为没答类型而丢。

### S3 明确 LED

```text
客户: i need an led display, indoor, 3m x 5m
  类型=LED/CONFIRMED  下一问=pixel_pitch
  AI: What pixel pitch do you have in mind: a finer one like P1.5, P2.5, or …?
```

### S4 明确 LCD

```text
客户: i need an lcd video wall
  类型=LCD/CONFIRMED  下一问=environment
  AI: Understood, let's go with LCD video wall for this project.
```

### S5 IFP 特征（归 LCD 子类型）

```text
客户: i need a touch screen for a meeting room
  类型=LCD/INFERRED（subtype=IFP）  下一问=pixel_pitch
  AI: Alright, a conference room. From what you have described, LCD (interactive …)
```

### S6 图片轮（Vision 判 LED）

```text
客户: （图片）like this
  类型=LED/INFERRED（source=VISION）  下一问=environment
  AI: From what you have described, LED looks like the better fit here, shall I …?
```

### S7 多消息合并

```text
客户一次发多条（"i need a display" + "3*5"）→ 只回一条、只决策一次
锚点测试：tests/test_message_aggregation_chain.py、tests/input/test_collect_messages.py
```

## 2. 量化基线（同 `baseline.md` 第 6 节）

```text
Calculator Accuracy             1.0（14/14）
Requirement Slot Accuracy       0.7032
Route Accuracy                  0.7439
检索 Series Recall@5 / Model Recall@10   1.0 / 0.9333（抽样 12）
硬约束违规率                     0.0
```

## 3. 行为锚点测试（瘦身时不能动这些）

| 场景 | 锚点测试 |
|---|---|
| 一轮一个问题 / 不重复问 | `tests/dialogue/test_one_question_per_turn.py`、`test_no_duplicate_question.py` |
| 多消息合并 | `tests/test_message_aggregation_chain.py`、`tests/input/test_collect_messages.py` |
| 答非所问也吸收 | `tests/dialogue/test_answer_wrong_slot.py`、`test_environment_first.py` |
| 产品类型判断（LED/LCD/IFP/锁定/改口/语境） | `tests/dialogue/test_v293_product_type_router.py`、`test_v294_type_context_branches.py`、`test_v294_type_understanding.py`、`test_v294_vision_product_type.py` |
| 图片链路 | `tests/test_vision_pipeline.py`、`tests/vision/*` |
| 推荐 / Gate | `tests/test_new_questions.py`、`tests/dialogue/test_quotation_request.py`、`eval/recommendation_eval.py` |
| 计算 | `tests/test_screen_calculator.py`、`python -m eval.calculator_eval` |
| 检索 | `python -m eval.retrieval_eval` |
| API / 前端契约 | `tests/test_api_http.py`、`tests/test_frontend_turn_merge.py` |
| 会话重置 / 隔离 | `tests/test_session_reset.py`、`tests/test_session_isolation.py` |

## 4. 明确"不改变"的行为清单（对应计划 §1.1）

LED/LCD/IFP 推荐、室内外判断、Ready Gate（推荐 + 计算）、硬过滤、评分排序、
RAG 混合检索（Dense + Sparse + BM25 + RRF）、产品 JSON 数据、Screen Calculator、
Sales/Solution 两 Agent、First Contact、Session/Turn 上下文、多消息合并、
一轮一个问题、Response Validator / Final Guard、图片识别、FastAPI 接口、
前端聊天、WhatsApp 入口、现有 API 调用方式。
