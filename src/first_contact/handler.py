"""
First Contact Handler：首次客户固定接待完整流程控制器。

职责：
1. 判断是否需要执行首次接待（由 Memory.first_contact_sent 决定）
2. 读取 Profile，生成自我介绍
3. 依次发送自我介绍 + 案例视频 + PDF
4. 单个素材发送失败不阻塞后续流程
5. 记录首次接待完成状态到 Memory

注意：
- handler 只负责首次接待，不负责产品推荐/RAG/销售策略
- 固定流程由代码控制，不交给 LLM 自主决定
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import logging
import time

from src.first_contact.profile import load_profile, CompanyProfile
from src.first_contact.assets import (
    ASSETS,
    FirstContactAsset,
    get_asset_path,
    get_asset_url,
    get_available_assets,
)
from src.core.llm import get_llm
from src.memory.store import memory

logger = logging.getLogger(__name__)


def _language_rule(language: str) -> str:
    """v2.0 Phase 14：回复语言指令（默认英语，RESPONSE_LANGUAGE_POLICY=auto 时跟随客户语言）。"""
    try:
        from src.rag.query_understanding import response_language_rule

        return response_language_rule(language)
    except Exception:  # pragma: no cover - 防御式
        return "ALWAYS use English, regardless of the customer's language."

# ── Prompt ─────────────────────────────────────────────────────────────────
# 注意：所有回复强制使用英语，无论客户使用何种语言

SELF_INTRO_PROMPT = """\
You are {sales_name}, a sales representative. Generate a warm, natural self-introduction \
for first-time customer contact based on the company information provided below.

CRITICAL RULES:
1. Use ONLY information from the company profile below. DO NOT invent or assume any details.
2. If a field is empty or missing, skip it entirely.
3. Write in a conversational, friendly tone - like a real person introducing themselves, not a corporate script.
4. {language_rule}
5. Structure: Start with greeting + your name/role → briefly mention company highlights (founded year, expertise, experience) → mention 1-2 key strengths or what makes the company stand out → end with a friendly question asking for their name and needs.
6. Keep it 3-5 sentences. Be concise but personable.
7. Vary your wording naturally , don't use rigid templates. Make it feel genuine.
8. DO not use "—" in the introduction.
Company Profile:
{profile_text}

Customer's first message (for context):
{customer_message}

Generate a natural, personable self-introduction:"""


ASSET_TRANSITION_PROMPT = """\
You just sent case videos and product catalogs to a potential customer. \
Generate ONE brief, friendly message to transition after sending these materials.

CRITICAL RULES:
1. Generate ONLY ONE message - do not provide multiple options or variations.
2. Keep it natural and conversational - like you're chatting with a colleague.
3. {language_rule}
4. Acknowledge that you've shared materials (videos/catalogs) to show what the company is about.
5. Keep it to 1-2 sentences maximum.
6. Sound warm and helpful, not robotic or overly formal.
7. Vary your wording naturally each time - don't use rigid templates.
8. DO NOT use emojis or special characters.

Company name: {company_name}

Generate ONE friendly transition message (no options, just the message):"""


BUSINESS_CARD_PROMPT = """\
You just sent your business card to a potential customer in a first contact scenario. \
Generate TWO sentences:

1. First sentence: Acknowledge you're sending your business card (vary the wording naturally, like "Here's my business card" or "This is my card" or similar)
2. Second sentence: Politely ask the customer to share their company website, personal email, and business card to move forward.

CRITICAL RULES:
1. Generate EXACTLY TWO sentences, no more, no less.
2. {language_rule}
3. Keep it professional but friendly and conversational.
4. Vary your wording naturally - don't use rigid templates.
5. DO NOT use emojis or special characters.
6. The request should feel helpful and natural, not demanding.

Sales person name: {sales_name}

Generate the two-sentence message:"""


# ── 返回结构 ───────────────────────────────────────────────────────────────

@dataclass
class SendResult:
    """单个素材发送结果。"""
    asset: FirstContactAsset
    success: bool
    url: str | None = None
    error: str | None = None
    skipped: bool = False   # 文件不存在时跳过


@dataclass
class FirstContactResult:
    """
    首次接待完整执行结果。

    Attributes:
        session_id: 会话 ID
        intro_text: 生成的自我介绍文本
        asset_results: 每个素材的发送结果
        transition_text: 素材发送后的过渡语
        business_card_text: 名片发送后的话术
        business_card_url: 名片URL
        all_success: 是否所有素材全部发送成功
        intro_success: 自我介绍是否发送成功
        transition_success: 过渡语是否生成成功
        business_card_success: 名片是否发送成功
        should_continue: 是否应该继续进入 Sales Agent
    """
    session_id: str
    intro_text: str = ""
    asset_results: list[SendResult] = field(default_factory=list)
    transition_text: str = ""
    business_card_text: str = ""
    business_card_url: str = ""
    all_success: bool = False
    intro_success: bool = False
    transition_success: bool = False
    business_card_success: bool = False
    should_continue: bool = True

    def to_messages(self) -> list[dict[str, Any]]:
        """
        将执行结果转换为消息列表，供前端展示或存入 Memory。

        返回格式示例：
        [
            {"role": "assistant", "content": "Hi, I'm John..."},
            {"role": "assistant", "content": "[video] case_video_01.mp4", "url": "/static/first_contact/..."},
            {"role": "assistant", "content": "Thank you for reaching out..."},
            ...
        ]
        """
        messages = []
        if self.intro_text:
            messages.append({"role": "assistant", "content": self.intro_text})
        for result in self.asset_results:
            if result.skipped:
                continue
            if result.success and result.url:
                content = f"[{result.asset.asset_type}] {result.asset.name}"
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "asset_type": result.asset.asset_type,
                    "asset_name": result.asset.name,
                    "asset_url": result.url,
                    "send_success": True,
                })
            elif not result.success and not result.skipped:
                logger.warning(
                    f"[FirstContact] Send failed | session={self.session_id} "
                    f"asset={result.asset.filename} error={result.error}"
                )
        # 添加素材发送后的过渡语
        if self.transition_text:
            messages.append({"role": "assistant", "content": self.transition_text})
        
        # 添加名片和话术
        if self.business_card_success and self.business_card_url:
            messages.append({
                "role": "assistant",
                "content": "[image] Business Card",
                "asset_type": "image",
                "asset_name": "Business Card",
                "asset_url": self.business_card_url,
                "send_success": True,
            })
        if self.business_card_text:
            messages.append({"role": "assistant", "content": self.business_card_text})
        
        return messages


# ── Handler ────────────────────────────────────────────────────────────────

class FirstContactHandler:
    """
    首次客户固定接待流程控制器。

    用法：
        result = first_contact_handler.run(session_id, customer_message)
        for msg in result.to_messages():
            await send_to_client(msg)
        if result.should_continue:
            await sales_agent.handle(customer_message)
    """

    def __init__(self) -> None:
        self._llm = get_llm()

    # ── 主入口 ──────────────────────────────────────────────────────────────

    def run(
        self,
        session_id: str,
        customer_message: str,
        *,
        language: str = "en",
    ) -> FirstContactResult:
        """
        执行首次接待完整流程。

        Args:
            session_id: 会话 ID
            customer_message: 客户发送的第一条消息
            language: 自我介绍语言（en/zh），默认英文

        Returns:
            FirstContactResult，包含自我介绍文本、素材发送结果
        """
        start = time.time()
        logger.info(f"[FirstContact] Starting | session={session_id}")

        # 1. 加载 Profile
        profile = load_profile()
        profile_text = self._profile_to_text(profile)

        # 2. 生成自我介绍
        intro_text, intro_ok = self._generate_intro(profile_text, customer_message, language)

        # 3. 发送素材
        asset_results = self._send_assets(session_id)
        all_success = all(r.success or r.skipped for r in asset_results)

        # 4. 生成素材发送后的过渡语
        transition_text, transition_ok = self._generate_transition(profile.company, language)

        # 5. 发送名片
        business_card_url, business_card_ok = self._send_business_card(session_id)
        
        # 6. 生成名片话术
        business_card_text = ""
        business_card_text_ok = False
        if business_card_ok:
            business_card_text, business_card_text_ok = self._generate_business_card_message(
                profile.sales_name, language
            )

        elapsed_ms = (time.time() - start) * 1000
        logger.info(
            f"[FirstContact] Done | session={session_id} "
            f"intro_ok={intro_ok} all_success={all_success} transition_ok={transition_ok} "
            f"business_card_ok={business_card_ok} business_card_text_ok={business_card_text_ok} "
            f"elapsed_ms={elapsed_ms:.0f}"
        )

        return FirstContactResult(
            session_id=session_id,
            intro_text=intro_text,
            asset_results=asset_results,
            transition_text=transition_text,
            business_card_text=business_card_text,
            business_card_url=business_card_url,
            all_success=all_success,
            intro_success=intro_ok,
            transition_success=transition_ok,
            business_card_success=business_card_ok and business_card_text_ok,
            should_continue=True,  # 素材失败也不阻塞，继续进入 Sales Agent
        )

    # ── 内部方法 ────────────────────────────────────────────────────────────

    def _generate_intro(
        self,
        profile_text: str,
        customer_message: str,
        language: str,
    ) -> tuple[str, bool]:
        """
        使用 LLM 生成自我介绍（强制英语）。

        Returns:
            (intro_text, success)
        """
        try:
            # 强制使用英语，无论 language 参数值
            # 注入销售人员名字和客户消息到 prompt
            p = load_profile()
            sales_name = p.sales_name or "a sales representative"
            
            prompt = SELF_INTRO_PROMPT.format(
                sales_name=sales_name,
                profile_text=profile_text,
                customer_message=customer_message,
                language_rule=_language_rule(language),
            )

            response = self._llm.invoke(prompt)
            text = response.content.strip() if hasattr(response, "content") else str(response)
            return text, True
        except Exception as e:
            logger.error(f"[FirstContact] Intro generation failed: {e}")
            # 降级方案：从 profile 读取信息，不添加任何额外内容
            p = load_profile()
            parts = [f"Hi, I'm {p.sales_name}." if p.sales_name else "Hi."]
            if p.position and p.company:
                parts.append(f"I work at {p.company} as {p.position}.")
            elif p.company:
                parts.append(f"I work at {p.company}.")
            parts.append("May I know your name and what you're looking for?")
            fallback = " ".join(parts)
            return fallback, False

    def _generate_transition(self, company_name: str, language: str = "en") -> tuple[str, bool]:
        """
        生成素材发送后的过渡语（强制英语）。

        Returns:
            (transition_text, success)
        """
        try:
            prompt = ASSET_TRANSITION_PROMPT.format(
                company_name=company_name, language_rule=_language_rule(language)
            )
            response = self._llm.invoke(prompt)
            text = response.content.strip() if hasattr(response, "content") else str(response)
            return text, True
        except Exception as e:
            logger.error(f"[FirstContact] Transition generation failed: {e}")
            # 降级方案：使用简单的固定话术
            fallback = f"Thank you for reaching out to {company_name}. I've shared some materials to show you what we're all about."
            return fallback, False

    def _generate_business_card_message(self, sales_name: str, language: str = "en") -> tuple[str, bool]:
        """
        生成名片发送后的话术（强制英语）。

        Returns:
            (business_card_text, success)
        """
        try:
            prompt = BUSINESS_CARD_PROMPT.format(
                sales_name=sales_name, language_rule=_language_rule(language)
            )
            response = self._llm.invoke(prompt)
            text = response.content.strip() if hasattr(response, "content") else str(response)
            return text, True
        except Exception as e:
            logger.error(f"[FirstContact] Business card message generation failed: {e}")
            # 降级方案
            fallback = "Here's my business card. To get that moving, could you share your company website, a personal email and personal business card?"
            return fallback, False

    def _send_business_card(self, session_id: str) -> tuple[str, bool]:
        """
        发送名片图片。

        Returns:
            (url, success)
        """
        import os
        # 查找 data/buisness_card 目录下的名片文件
        card_dir = os.path.join("data", "buisness_card")
        if not os.path.exists(card_dir):
            logger.warning(f"[FirstContact] Business card directory not found: {card_dir}")
            return "", False
        
        # 查找第一个图片文件
        card_files = [f for f in os.listdir(card_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not card_files:
            logger.warning(f"[FirstContact] No business card image found in {card_dir}")
            return "", False
        
        card_filename = card_files[0]
        # 生成静态文件 URL
        url = f"/static/buisness_card/{card_filename}"
        logger.info(f"[FirstContact] Business card ready | session={session_id} url={url}")
        return url, True

    def _send_assets(self, session_id: str) -> list[SendResult]:
        """
        依次发送所有素材，单个失败不影响后续。

        Returns:
            每个素材的 SendResult 列表
        """
        results = []
        for asset in ASSETS:
            result = self._send_single_asset(session_id, asset)
            results.append(result)
            # 失败只记录日志，不中断
            if not result.success and not result.skipped:
                logger.warning(
                    f"[FirstContact] Asset send failed | "
                    f"session={session_id} asset={asset.filename} error={result.error}"
                )
        return results

    def _send_single_asset(self, session_id: str, asset: FirstContactAsset) -> SendResult:
        """
        发送单个素材。

        失败返回 SendResult(success=False)，不抛异常。
        """
        path = get_asset_path(asset)
        if not path.exists():
            logger.info(
                f"[FirstContact] Asset not found (skipping) | "
                f"session={session_id} asset={asset.filename}"
            )
            return SendResult(asset=asset, success=False, skipped=True, error="File not found")

        try:
            url = get_asset_url(asset)
            # TODO: 这里可以接入实际的发送渠道（如邮件/API/WebSocket等）
            # 当前阶段只记录 URL，供调用方决定如何发送
            return SendResult(asset=asset, success=True, url=url)
        except Exception as e:
            return SendResult(
                asset=asset,
                success=False,
                skipped=False,
                error=str(e),
            )

    def _profile_to_text(self, profile: CompanyProfile) -> str:
        """将 CompanyProfile 转换为 LLM 可读的文本。"""
        lines = [
            f"Sales Name: {profile.sales_name}",
            f"Position: {profile.position}",
            f"Company: {profile.company}",
            f"Company Introduction: {profile.company_introduction}",
            f"Main Products: {', '.join(profile.main_products)}",
            f"Main Markets: {', '.join(profile.main_markets)}",
            f"Company Advantages: {', '.join(profile.company_advantages)}",
        ]
        return "\n".join(lines)


# 全局单例
first_contact_handler = FirstContactHandler()
