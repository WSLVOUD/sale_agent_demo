"""classify node - intent classification for the Sales Agent."""
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import SalesState
from ....config import config
from ....rag.parameter_inference import detect_intent

logger = logging.getLogger(__name__)


def classify(state: SalesState) -> SalesState:
    """Classify the user's intent.

    Possible intents:
      - "greeting": user is just saying hello
      - "product_question": user asks about product specs, LED vs LCD comparison,
        capability (4K/HDR support), differences, specific model features
      - "need_query": user is describing requirements or asking for recommendations
      - "objection": user is raising an objection
      - "industry": user mentions a specific industry scenario
      - "closing": user wants to close / wrap up
      - "others": user is asking about anything OUTSIDE the standard sales flow
    """
    message = state["current_message"]

    # 【关键修复】先调用 LLM 分类，再检查 detect_intent。
    # 如果 detect_intent 检测到推荐意图，即使 LLM 分类为 others，也强制走推荐流程。
    detected = detect_intent(message)
    logger.info("detect_intent result: %r", detected)
    
    # 注意：不要在 LLM 调用前 return，否则 detect_intent 的返回值无法覆盖 LLM 的结果
    # 正确做法：先 LLM，再 override

    llm = ChatOpenAI(
        model=config.MODEL_NAME,
        temperature=0,
        api_key=config.DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )

    prompt = SystemMessage(content="""你是销售意图分类器。分析用户消息，返回以下意图之一：
- greeting: 问候、打招呼（你好、hi、hello、谢谢、再见）—— 注意：「好的」「知道了」「明白了」等确认词不属于greeting，属于need_query
- product_question: 用户在问产品参数、功能、支持情况、LED和LCD的区别、某款产品能否做某事、显示效果好不好等——任何不是「帮我推荐/采购/选哪款」的提问
- need_query: 用户描述需求或请求推荐（我要会议室用的屏、想采购广告屏、帮我选一个）或确认需求（好的、知道了、明白了、对的、是的）
- objection: 提出针对销售方案的异议或询问销售政策（价格太贵、能不能便宜、报价有效期、付款方式、采购流程、售后政策、安装调试）
- industry: 提到特定行业场景（零售、教育、会议室、体育场馆等）
- closing: 准备下单或询问下一步（那就这样定了、下单、签合同）
- others: 项目正常流程无法处理的问题，比如公司信息（你们公司在哪里、联系方式）、工作时间（几点上班、节假日）、产品保修政策（保修多久、保修范围）、商务流程疑问（怎么签约、什么时候能交付）、闲聊话题等。

判断规则：
1. 只有当用户在同一句话里同时包含业务问题和其他问题时，才选择其他更精确的类别（如 product_question）。否则默认走 others。
2. 「价格能便宜点吗」「能再优惠点吗」「报价怎么样」属于 objection，不是 others。
3. 「保修多久」「保修几年」「质保范围」属于 objection，不是 others。
4. 「你们公司在哪里」「怎么联系你们」「几点上班」「工厂在哪」属于 others。
5. 「好的」「知道了」「明白了」「对的」「是的」等确认词属于 need_query，不属于 greeting。

只返回意图类型，不要其他内容。""")

    response = llm.invoke([prompt, HumanMessage(content=message)])
    intent = response.content.strip().lower()

    logger.info(f"Classified intent: {intent}")

    # 【关键修复】如果 detect_intent 检测到推荐意图，即使 LLM 分类为 others/industry，也强制走推荐流程
    if detected == "recommendation" and intent in ("others", "industry"):
        logger.info(f"Overriding LLM intent '{intent}' → 'need_query' (detect_intent matched)")
        intent = "need_query"
    # 【兜底】如果 detect_intent 返回非推荐非空，且 LLM 分类为 others，但消息包含推荐关键词，也强制覆盖
    elif detected == "" and intent == "others":
        # 再次检查消息是否包含推荐关键词（独立于 detect_intent 的正则）
        recommendation_keywords = ["会议室", "教室", "培训", "广告", "屏", "LED", "LCD", "IFP", "手写", "触控",
                                   "舞台", "演出", "租赁", "户外", "室内", "需要", "采购", "推荐"]
        if any(kw in message for kw in recommendation_keywords):
            logger.info(f"Overriding LLM intent 'others' → 'need_query' (fallback keyword check)")
            intent = "need_query"
    
    # 【首次接待后修复】如果首次接待刚完成(suppress_greeting=True)且消息包含场景关键词，
    # 强制分类为 need_query 而不是 greeting，避免进入问候流程
    if state.get("suppress_greeting") and intent == "greeting":
        scenario_keywords = [
            "会议室", "教室", "培训", "会议", "学校", "培训室",
            "零售", "店铺", "商场", "商店", "超市",
            "广告", "传媒", "宣传",
            "舞台", "演出", "演唱会", "表演", "剧场",
            "体育", "赛场", "场馆", "球场",
            "展厅", "展览", "博物馆",
            "幕墙", "外墙", "建筑",
            "租赁", "活动", "临展",
            "LED", "LCD", "IFP", "显示屏", "屏幕",
            "屏", "大屏", "显示器",
            "室内", "室外", "户外",
            "meeting", "classroom", "retail", "store", "advertising", "stage", "LED", "LCD", "display",
            "screen", "outdoor", "indoor"
        ]
        if any(kw in message.lower() for kw in scenario_keywords):
            logger.info(f"Overriding LLM intent 'greeting' → 'need_query' (suppress_greeting + scenario keywords detected)")
            intent = "need_query"

    state["intent"] = intent

    # Set next_action so orchestrator can route appropriately
    if intent == "product_question":
        state["next_action"] = "product_question"
    elif intent == "others":
        state["next_action"] = "others"
    elif intent == "greeting":
        state["next_action"] = "ask"
    elif intent in ("objection", "industry"):
        state["next_action"] = "ask"
    elif intent == "closing":
        state["next_action"] = "end"
    else:
        state["next_action"] = "ask"  # need_query falls through to requirement_mining

    return state
