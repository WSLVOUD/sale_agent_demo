"""回归：客户文案必须按**客户自己的需求**说，不能拿内部品类当客户的说法。

实测 bug（客户 2026-10）：

    客户全程说：展会（exhibition）
    系统内部品类：advertising（语境理解器判的，用于决定走哪条需求链）
    最终推荐：  "For your advertising video wall, we recommend the H6530LN-B …"   ← ❌

客户从没说过广告。根因是**内部品类 token 被当成"客户的说法"喂进了写话的提示词**：

  1. ``recommend._express_lcd_recommendation`` 的 LCD 推荐提示词里写了
     ``Customer's category: advertising`` → 模型照抄成 "advertising video wall"；
  2. ``script_generator._known_facts`` 把 ``lcd_category`` 标成 ``use case=advertising``
     交给表达层 —— 直接声称"客户的使用场景是广告"；
  3. LCD 追问轮的 business_goal 里带 ``(LCD branch: advertising)``。

修法不是拿关键词去堵（那样只是治症状），而是**把客户自己的场景给到模型**：
``purpose``（客户说的使用场景）→ ``reply_composer.purpose_phrase()`` 的客户口径说法，
并且明确要求模型按客户说的场景组织措辞、不许用内部品类名。

本文件锁的就是这个行为：**客户要的东西进提示词，内部品类不进。**
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _exhibition_profile():
    """复刻客户那条会话：客户说展会，内部品类判成 advertising。"""
    from src.models.requirement import RequirementProfile

    profile = RequirementProfile()
    profile.display_type = "LCD"
    profile.purpose = "exhibition"
    profile.lcd_category = "advertising"
    profile.environment = "indoor"
    profile.installation = "fixed"
    profile.lcd_size_inch = 65.0
    profile.lcd_is_splicing = True
    profile.lcd_splicing_layout = "3x3"
    profile.sources.update(
        {
            "purpose": "explicit",
            "environment": "explicit",
            "installation": "explicit",
            "lcd_category": "understanding",
            "lcd_size_inch": "explicit",
            "lcd_is_splicing": "explicit",
            "lcd_splicing_layout": "explicit",
        }
    )
    return profile


class TestCustomerUseCaseReplacesInternalCategory:

    def test_locked_facts_give_the_customer_scene_not_the_category(self):
        """交给表达层的是客户自己的场景，内部品类 token 不许出现。"""
        from src.agents.sales.nodes.script_generator import _known_facts

        profile = _exhibition_profile()
        facts = _known_facts(
            {
                "requirement_profile": profile,
                "lcd_action": {
                    "locked_facts": {
                        "purpose": "exhibition",
                        "lcd_category": "advertising",
                        "lcd_size_inch": 65.0,
                        "lcd_is_splicing": True,
                        "lcd_splicing_layout": "3x3",
                    }
                },
            }
        )
        joined = " ".join(facts).lower()
        assert "advertising" not in joined, facts
        assert "purpose=exhibition" in joined, facts
        # 真正有用的客户事实一条都不能少
        assert any("screen size=65" in item for item in facts), facts
        assert any("wall layout=3x3" in item for item in facts), facts

    def test_recommend_prompt_uses_the_customers_own_use_case(self):
        """LCD 推荐提示词带客户场景（an exhibition），正文不带内部品类。"""
        from src.agents.solution.nodes.recommend import (
            LCD_RECOMMEND_PROMPT,
            _customer_use_case,
        )

        profile = _exhibition_profile()
        use_case = _customer_use_case(profile)
        assert use_case == "an exhibition", use_case

        prompt = LCD_RECOMMEND_PROMPT.format(
            product_family="commercial LCD display",
            model="H6530LN-B",
            facts="65 panels, 4K resolution, 3.5mm bezel, 500nit brightness",
            layout="3x3 layout (9 panels)",
            reasons="video-wall panels with 3.5mm bezel",
            use_case=use_case or "not stated by the customer",
            follow_up_line="",
            seam_facts="selected panel bezel = 3.5mm; customer asked for <= 0.88mm; "
                       "it does NOT meet the customer's seam requirement — do not claim it does",
            conversation="(no earlier turns available)",
        )
        body = prompt.split("Rules:")[0]
        assert "advertising" not in body.lower(), body
        assert "an exhibition" in prompt

    def test_no_use_case_means_do_not_invent_one(self):
        """客户没说场景时，不给场景、也不退回内部品类（避免凭空替客户说）。"""
        from src.agents.solution.nodes.recommend import _customer_use_case
        from src.models.requirement import RequirementProfile

        profile = RequirementProfile()
        profile.lcd_category = "advertising"  # 只有内部品类，没有客户场景
        assert _customer_use_case(profile) == ""

    def test_purpose_phrase_is_the_single_customer_facing_vocabulary(self):
        """客户口径说法只有一个来源（reply_composer 的词表）。"""
        from src.rag.reply_composer import purpose_phrase

        assert purpose_phrase("exhibition", "en") == "an exhibition"
        assert purpose_phrase("conference", "en") == "a conference room"
        assert purpose_phrase("exhibition", "zh") == "展会"
        # 认不出的 token 原样返回，不编造
        assert purpose_phrase("some_new_scene", "en") == "some_new_scene"
        assert purpose_phrase(None, "en") == ""
