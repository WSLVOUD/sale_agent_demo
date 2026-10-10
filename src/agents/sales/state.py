"""Sales Agent state definition."""
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import add_messages


class SalesState(TypedDict):
    """State for the Sales Agent.

    The Sales Agent runs alongside the existing Solution Agent and shares
    the same `session_id` / `memory_store` so that both agents see the
    same conversation history.
    """

    # Conversation
    messages: Annotated[List[Dict], add_messages]
    session_id: str
    current_message: str  # The user's current message being processed

    # Intent classification: "greeting" | "need_query" | "objection" | "closing" | "industry"
    intent: str

    # Requirements extracted from the user so far
    requirements: Dict[str, Any]
    # {
    #   "location_type": "indoor" | "outdoor" | None,
    #   "usage": str | None,
    #   "size": str | None,
    #   "brightness": str | None,
    #   "resolution": str | None,
    # }

    # Mining progress（M3：这两个字段已降级为"Ready Gate 结果的投影"，
    # 只用于旧调用方兼容，不再参与任何判断）
    required_met: List[str]        # = Gate ready
    required_missing: List[str]   # = Gate missing
    turn_count: int

    # Additional/custom requirements not in the predefined list
    # These will be passed to AI for analysis (pixel pitch, brightness, rental, etc.)
    additional_requirements: List[str]

    # Trigger flag（M5：唯一写入方是 Recommendation Ready Gate；
    # 其它模块只能读取，不得再自己判断是否推荐）
    should_generate_solution: bool

    # Sales script RAG
    retrieved_scripts: List[Dict[str, Any]]

    # Outputs
    response: str
    solutions: Optional[List[Dict[str, Any]]]   # 由 Solution Agent 填充

    # Routing: "ask" | "answer" | "trigger_solution" | "end"
    next_action: str

    # Shared references (set by the runner)
    sales_search: Optional[Any]
    solution_runner: Optional[Any]

    # 首次接待刚完成后抑制销售问候语（因为已经自我介绍过了）
    suppress_greeting: bool

    # ── Phase 6/7：结构化需求档案 + 下一个待问的高价值问题 ────────────────
    requirement_profile: Optional[Any]
    pending_question: str
    pending_slot: str   # 待问的是哪一个槽位（用于判断答案里是否已经在问同一件事）

    # ── v2.6 §16/§17：Dialogue Policy 的判定结果（SpeechAct + 唯一 Action）──
    # 必须写进 state schema，否则 LangGraph 不会把这两个键带出图，
    # 上层就只能看到空值（实测：日志里 speech_act=-、action=ask 而不是 ask_only）。
    speech_act: Dict[str, Any]
    dialogue_action: Dict[str, Any]
    response_plan: Dict[str, Any]

    # ── 会话内需求重置（客户拿到推荐后又要换产品 / 换项目 / 改需求）──────
    # requirements_reset=True 时本轮必须回到需求采集，不得沿用旧需求直接推荐。
    requirements_reset: bool
    reset_reason: str

    # ── 本会话是否已经给过推荐（决定"客户后续提问时要不要再推荐一遍"）────
    already_recommended: bool
    # 本会话已经推荐过的型号（"另外推荐一款"时换一个没给过的）
    previous_recommended_models: List[str]

    # ── 客户开始说"新的一块屏"（客户口径 2026-10）──────────────────────────
    # {"reason": "explicit_multi_item", "cleared": [...]} —— 这一轮清掉了上一块屏的
    # 哪些需求事实（新的一块屏**不继承**上一块的需求）。留痕供日志 / 测试查。
    lcd_new_item: Dict[str, Any]

    # ── 本轮是否带了图片（图片识别结果要跟客户确认一次）──────────────────
    vision_applied: bool
    # ── 本轮图片识别出的关键信息（计划 v2.9.4 §六）───────────────────────
    # {"display_type": "LED"|"LCD"|"IFP", "source": "vision_explicit", "reason": str}
    # 第一层产品判断（Product Type Router）的"图片判断"分支读它。
    vision: Dict[str, Any]

    # ── 第一层产品判断（LED / LCD）的结果（计划 v2.9.2/v2.9.3/v2.9.4）──────
    # 【必须写在 schema 里】LangGraph 只把**声明过的键**带出图。这几个键以前没声明，
    # 于是 orchestrator 收到的 display_type_decision 恒为 {} —— 收口层的
    # "类型没确认就不许问需求细节"闸门在真实链路上从来没生效过
    # （2026-09-23 线上日志：客户说 "i need a display" 却被问了室内外）。
    display_type_decision: Dict[str, Any]
    product_entry: str
    product_domain: str
    product_subtype: str
    product_switch: Dict[str, Any]
    turn_kind: str
    turn_understanding: Dict[str, Any]
    understanding: Dict[str, Any]
    # LCD 入口 / 类型闸门的留痕（供 orchestrator、日志与审计读取）
    lcd_entry: Dict[str, Any]
    product_type_gate: Dict[str, Any]
    # ── LCD / IFP 需求链的决策结果（《LCD_IFP 整改计划》Phase 5）─────────────
    # 【必须写在 schema 里】和上面 display_type_decision 同一个坑：LangGraph 只把
    # **声明过的键**带出图，没声明的键会在节点之间被丢掉。
    # 实测 2026-09-30：requirement_mining 明明算出了
    # "Do you need a video wall (spliced screens) or single displays?"，
    # 但 script_generator 拿不到 lcd_action → 走旧兜底 → 回复空 → 被 API 的
    # "放宽条件"话术顶上，客户看到的既不是问题也不是承接。
    lcd_action: Dict[str, Any]

    # ── 本轮客户说的是与需求无关的话（只"接住"这句话，再继续问需求）──────
    offtopic_turn: bool

    # ── 客户在**同意推进**（语义判定，不是关键词）──────────────────────────
    # 已推荐过之后，客户回来说"yes / 可以 / 就这个 / go ahead"这类话 ——
    # 到底是在同意出报价，还是在回答某个需求问题，必须结合上下文理解整句话。
    # 由 requirement 节点用 LLM（带最近对话）判定，orchestrator 据此回"报价在准备"。
    quote_confirmation: bool

    # ── 客户**直接指名**的型号（客户口径 2026-10）────────────────────────────
    # 客户说 "i need a TW-OD-11"（= 目录里的 TW11-OD）时记下系列名，后续
    # **围绕这个型号**问需求：型号本身已确定的属性（LED/LCD、室内外）不再重复问。
    # 解析不出来时为空串 —— 绝不猜。
    specified_model: str

    # 本轮"回应客户这句话"的口语回应（LLM 生成，只影响措辞，不参与 Gate 判定）
    acknowledgement: str

    # ── 2026-09-30：把"实际在用但从没声明"的键补进 schema ───────────────────
    # LangGraph 只把**声明过的键**带出图（同一个坑见上面 display_type_decision 的
    # 注释）。下面这些键在节点之间本来是**静默丢失**的：
    #   · pitch_resolution / recommendation / screen_calculation / recommendation_gate
    #     → 表达层拿不到，LLM 只能自己编或只说半句；
    #   · response_action / response_source / service_faq_answered → 留痕与收口判断失效；
    #   · greeting_entry / greeting_sent_this_turn / legacy_reply_path → 招呼与旧链路标记失效。
    # 护栏：tests/architecture/test_architecture_sales_state_channels.py 会检查这一点。
    greeting_entry: bool
    greeting_sent_this_turn: bool
    legacy_reply_path: bool
    pitch_resolution: Dict[str, Any]
    recommendation: Dict[str, Any]
    recommendation_gate: Dict[str, Any]
    response_action: str
    response_source: str
    screen_calculation: Dict[str, Any]
    service_faq_answered: str
