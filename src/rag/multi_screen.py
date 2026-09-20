"""v2.3.1 Phase 2：一个项目多条屏的业务逻辑（从 orchestrator 迁出）。

客户口径（2026-09-18）：一句话能看出两块屏就分别记录；客户指明"改哪一块"就切到
那一块；没说清差异时两块记成一样的；同一个问题全项目最多问两次。

本模块只做"多屏需求与多屏推荐"，所有依赖通过构造参数注入：

    MultiScreenManager(store=…, profile_lookup=…, history_lookup=…, solution_agent=…)

方法体是从 `orchestrator.py` **原样搬过来**的（逻辑一行未改），因此
`self.memory_store` / `self._stored_profile` 这些属性名保持一致（在 __init__ 里绑定）。
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class MultiScreenManager:
    """多屏需求 / 多屏推荐的唯一实现（Orchestrator 只调用它）。"""

    _SHARED_FACT_FIELDS = (
        "display_type", "purpose", "content_type", "installation",
        "price_preference", "budget_level", "viewing_distance_m",
        # 没说清就两块一样：点间距也一样（"P3 for the indoor P5 for the outdoor"
        # 这种明确说了的才各自不同）
        "pixel_pitch_mm",
    )

    def __init__(
        self,
        store: Any = None,
        profile_lookup: Optional[Callable[[str], Any]] = None,
        history_lookup: Optional[Callable[[str], Any]] = None,
        solution_agent: Any = None,
    ) -> None:
        # 属性名与原来在 Orchestrator 里的写法保持一致，方法体可以原样搬
        self.memory_store = store
        self._stored_profile = profile_lookup or (lambda _sid: None)
        self._load_history = history_lookup or (lambda _sid: [])
        self.solution_agent = solution_agent

    # ── 多屏回复的"是哪一块屏"标注（v2.3.1 从 Orchestrator 迁入）─────────────
    def prefix_active_screen_label(self, session_id: str, message: str, response: str) -> str:
        """第二块（及以后）的推荐要标清楚是哪一块屏。

        实测：客户开了第二块屏之后，推荐话术里仍会出现 "For your church indoor
        screen …"（历史里的第一块屏），客户分不清在说哪一块。
        """
        text = str(response or "")
        if not text or not session_id:
            return text
        store = self.memory_store
        if store is None or not hasattr(store, "get_active_item_index"):
            return text
        try:
            from ..rag.project_items import screen_label
            from ..rag.reply_composer import reply_language

            active_index = store.get_active_item_index(session_id) or 0
            profile = self._stored_profile(session_id)
            if active_index < 1 or profile is None:
                return text
            label = screen_label(active_index, profile, reply_language(message))
            if label and not text.lstrip().startswith(label.strip()):
                return f"{label}{text}"
        except Exception as exc:  # pragma: no cover - 防御式
            logger.warning("Screen label prefix failed: %s", exc)
        return text

    def _split_and_apply_screen_specs(self, session_id: str, message: str) -> list:
        """一句话里给了多块屏的规格 → 每块屏各存一份需求。

        客户口径：一句里能看出两块屏（"4m wide x2.5 high for indoor and 3m x2m
        for the outdoor"）就**分别记录**；分不清哪组参数属于哪块屏时两块记成一样的。
        返回拆分结果（空列表表示这一轮不是多屏描述）。
        """
        from ..models.requirement import RequirementProfile
        from .project_items import split_multi_screen_specs

        memory_store = self.memory_store
        if memory_store is None or not hasattr(memory_store, "get_project_items"):
            return []

        specs = split_multi_screen_specs(message)
        if len(specs) < 2:
            return []

        base = self._stored_profile(session_id)
        base_slots = base.model_dump() if base is not None else {}
        base_sources = dict(getattr(base, "sources", None) or {}) if base is not None else {}
        shared_skip = (
            "sources", "ask_counts", "unknown_reasons", "conflicts", "conflict_slots",
            "last_asked_slot", "model", "series_id", "pixel_pitch_mm",
        )

        items = memory_store.get_project_items(session_id)
        while len(items) < len(specs):
            items.append({})

        for index, spec in enumerate(specs):
            slots = {key: value for key, value in base_slots.items() if key not in shared_skip}
            # 【关键】继承来源标记：从基准档案抄过来的值，来源必须跟着抄过来。
            # 之前这里把所有字段一律写成 explicit，于是"教堂默认固装"这种
            # 系统默认值被当成"客户明说" → Gate 再也不问固装/租赁（实测 bug）。
            sources = {
                key: base_sources[key]
                for key, value in slots.items()
                if value not in (None, "", [], {}) and base_sources.get(key)
            }
            # 这一块屏**自己已经有**的值优先（否则第二句话只说了点间距时，
            # 会把另一块屏的安装方式/尺寸覆盖成 base 的值 —— 实测踩过）
            existing = dict((items[index] or {}).get("profile") or {})
            existing_sources = dict(existing.get("sources") or {})
            for key, value in existing.items():
                if key == "sources" or value in (None, "", [], {}):
                    continue
                slots[key] = value
                if existing_sources.get(key):
                    sources[key] = existing_sources[key]
            # 客户这句话里**明说**的字段（只有这些才记 explicit）
            explicit_fields = []
            if spec.get("environment"):
                slots["environment"] = spec["environment"]
                explicit_fields.append("environment")
            if spec.get("width_m"):
                slots["target_width_m"] = spec["width_m"]
                explicit_fields.append("target_width_m")
            if spec.get("height_m"):
                slots["target_height_m"] = spec["height_m"]
                explicit_fields.append("target_height_m")
            # 安装方式与点间距也要按屏幕分开（"permanent for indoor, rental for
            # outdoor" / "p3 for indoor p5 for outdoor" 实测会被混到一块上）
            if spec.get("installation"):
                slots["installation"] = spec["installation"]
                explicit_fields.append("installation")
            if spec.get("pixel_pitch_mm"):
                slots["pixel_pitch_mm"] = spec["pixel_pitch_mm"]
                explicit_fields.append("pixel_pitch_mm")
            slots["sources"] = sources
            try:
                profile = RequirementProfile.model_validate(slots)
            except Exception as exc:  # pragma: no cover - 防御式
                logger.warning("[%s] Multi-spec profile build failed: %s", session_id, exc)
                continue
            # 客户这句话里明说的值 → explicit（可直接用于选型/计算）；
            # 从基准档案继承来的值保留原来源（场景默认/推断不会被"洗白"）。
            for field in explicit_fields:
                profile.sources[field] = "explicit"
            record = dict(items[index] or {})
            record["profile"] = profile.model_dump()
            # 规格变了 → 这一块之前的推荐作废，重新推荐
            record.pop("model", None)
            items[index] = record

        memory_store.set_project_items(session_id, items)
        memory_store.set_active_item_index(session_id, len(specs) - 1)
        last = memory_store.get_project_items(session_id)[len(specs) - 1].get("profile")
        if last:
            memory_store.set_requirement_profile(
                session_id, RequirementProfile.model_validate(last)
            )
        logger.info("[%s] Multi-item: 按一句话拆出 %d 块屏的规格", session_id, len(specs))
        return specs

    def _maybe_target_screen(self, session_id: str, message: str) -> Optional[int]:
        """客户指明"改哪一块屏" → 把当前档案切到那一块，本轮就改它。"""
        from ..models.requirement import RequirementProfile
        from .project_items import detect_screen_target

        memory_store = self.memory_store
        if memory_store is None or not hasattr(memory_store, "get_project_items"):
            return None

        items = memory_store.get_project_items(session_id)
        index = detect_screen_target(message, items)
        if index is None:
            return None

        current = memory_store.get_active_item_index(session_id)
        if index == current:
            return index

        profile = self._stored_profile(session_id)
        if profile is not None and current < len(items):
            record = dict(items[current] or {})
            record["profile"] = profile.model_dump()
            items[current] = record
            memory_store.set_project_items(session_id, items)

        target = (items[index] or {}).get("profile")
        if target:
            try:
                memory_store.set_requirement_profile(
                    session_id, RequirementProfile.model_validate(target)
                )
            except Exception as exc:  # pragma: no cover - 防御式
                logger.warning("[%s] Screen switch failed: %s", session_id, exc)
                return None
        memory_store.set_active_item_index(session_id, index)
        logger.info("[%s] Multi-item: 客户指定改第 %d 块屏", session_id, index + 1)
        return index

    def _maybe_start_new_item(self, session_id: str, message: str) -> str:
        """客户这句话如果在说"另一块屏" → 归档当前这块，开一条新需求档案。

        客户口径："教堂里一块室内屏，门口再来一块室外屏" —— 一个项目、两块屏，
        各自收集需求、各自推荐、各自算箱体，最后给一份汇总。
        """
        from .project_items import detect_new_item

        memory_store = self.memory_store
        if memory_store is None or not hasattr(memory_store, "get_project_items"):
            return ""

        profile = self._stored_profile(session_id)
        if profile is None:
            return ""

        items = memory_store.get_project_items(session_id)
        index = memory_store.get_active_item_index(session_id)
        already = bool(
            getattr(memory_store, "has_recommendation", lambda *_: False)(session_id)
        ) or bool((items[index] if index < len(items) else {}).get("model"))

        should_start, reason = detect_new_item(
            message, profile, already_recommended=already
        )
        if not should_start:
            return ""

        # 归档第 N 块屏（保留需求档案，最后汇总要用）
        while len(items) <= index:
            items.append({})
        archived = dict(items[index] or {})
        archived["profile"] = profile.model_dump()
        items[index] = archived
        memory_store.set_project_items(session_id, items)
        memory_store.set_active_item_index(session_id, index + 1)

        # 客户口径：客户没说"这块要什么、那块要什么"时，两块**记成一样的** ——
        # 所以新条目先复制当前这一份需求，本轮这句话里明说的差异（"室外的"）再覆盖上去。
        from ..models.requirement import RequirementProfile

        seeded = RequirementProfile.model_validate(profile.model_dump())
        items = memory_store.get_project_items(session_id)
        while len(items) <= index + 1:
            items.append({})
        items[index + 1] = {"profile": seeded.model_dump()}
        memory_store.set_project_items(session_id, items)
        memory_store.set_requirement_profile(session_id, seeded)
        # 旧 requirements 是 Profile 的只读投影，下一轮由 Sales 重建
        memory_store.clear_requirements(session_id)
        if hasattr(memory_store, "clear_recommendation"):
            memory_store.clear_recommendation(session_id)
        logger.info(
            "[%s] Multi-item: 开始收集第 %d 块屏的需求（%s）", session_id, index + 2, reason
        )
        return reason

    @staticmethod
    def _screen_pending_block(index: int, profile_data: dict, language: str) -> str:
        """某块屏这轮还没定下来时，也要在回复里占一段（客户口径：有多少屏就出多少屏）。"""
        from .project_items import screen_label

        label = screen_label(index, profile_data or {}, language)
        if str(language or "").lower().startswith("zh"):
            return label + "这一块还在确认需求，确认好我马上把型号给你。"
        return label + "this one is still being confirmed, I'll come back with the model shortly."

    def _share_common_facts(self, session_id: str) -> None:
        """把当前这块屏已确认的共有项同步给其他屏，避免同一个问题被问第二遍。

        实测：两块屏时，客户答了 "both"（视频/图片）之后，系统又为另一块屏
        把同一个问题问了一遍 —— 客户体验就是"回答过了还问"。
        """
        memory_store = self.memory_store
        if memory_store is None or not hasattr(memory_store, "get_project_items"):
            return
        items = memory_store.get_project_items(session_id)
        if len(items) < 2:
            return
        active = memory_store.get_active_item_index(session_id)
        if active >= len(items):
            return
        # 当前这块的**实时**档案（items 里的可能还没同步过来）
        live = self._stored_profile(session_id)
        source = live.model_dump() if live is not None else dict(
            (items[active] or {}).get("profile") or {}
        )
        if not source:
            return
        sources = dict(source.get("sources") or {})
        changed = False
        for index, item in enumerate(items):
            merged = dict(item or {})
            if index == active:
                # 当前这块：把实时档案写回条目，保证"每条记录都是完整的"
                merged["profile"] = source
                items[index] = merged
                continue
            profile = dict((item or {}).get("profile") or {})
            if not profile:
                continue
            for field in self._SHARED_FACT_FIELDS:
                value = source.get(field)
                if value in (None, "", [], {}) or profile.get(field) not in (None, "", [], {}):
                    continue
                profile[field] = value
                if sources.get(field):
                    profile.setdefault("sources", {})[field] = sources[field]
                changed = True
            if changed:
                merged = dict(item or {})
                merged["profile"] = profile
                items[index] = merged
        if changed:
            memory_store.set_project_items(session_id, items)

    def _share_common_facts(self, session_id: str) -> None:
        """把当前这块屏已确认的共有项同步给其他屏（只填空值），

        这样同一个问题全项目只会问一次；能区分两块屏的参数不会被冲掉。
        """
        from ..models.requirement import CONFIRMED_SOURCES

        memory_store = self.memory_store
        if memory_store is None or not hasattr(memory_store, "get_project_items"):
            return
        items = memory_store.get_project_items(session_id)
        if len(items) < 2:
            return
        active = memory_store.get_active_item_index(session_id)
        if active >= len(items):
            return
        live = self._stored_profile(session_id)
        source = live.model_dump() if live is not None else {}
        if not source:
            return
        sources = dict(source.get("sources") or {})
        changed = False
        for index, item in enumerate(items):
            merged = dict(item or {})
            profile = dict(merged.get("profile") or {})
            if index == active:
                # 当前这块：把实时档案写回它的条目（避免"条目是旧值"看起来像串台）
                merged["profile"] = source
                items[index] = merged
                continue
            if not profile:
                continue
            target_sources = dict(profile.get("sources") or {})
            for field in self._SHARED_FACT_FIELDS:
                value = source.get(field)
                if value in (None, "", [], {}):
                    continue
                source_tag = sources.get(field)
                if profile.get(field) not in (None, "", [], {}):
                    # 已经有值：只有"目标这块只是场景默认/系统推断、客户明确答的就是
                    # 同一个值"时才把来源升级为客户确认 —— 这样客户答过的共有项
                    # （固装/租赁…）不会在另一块屏再问一遍；一旦两台的值不同
                    # （固装 vs 租赁、P3 vs P5）就保留差异，绝不覆盖。
                    if not source_tag or source_tag not in CONFIRMED_SOURCES:
                        continue
                    if target_sources.get(field) in CONFIRMED_SOURCES:
                        continue
                    if profile.get(field) != value:
                        continue
                profile[field] = value
                if source_tag:
                    target_sources[field] = source_tag
                changed = True
            profile["sources"] = target_sources
            merged["profile"] = profile
            items[index] = merged
        memory_store.set_project_items(session_id, items)

    @staticmethod
    def _model_matches_screen(model_name: str, profile_data: Dict[str, Any]) -> bool:
        """型号的使用环境是否和这块屏一致（避免"室外屏推室内型号"）。"""
        environment = str((profile_data or {}).get("environment") or "").strip().lower()
        if environment not in ("indoor", "outdoor"):
            return True
        try:
            from ..config import config
            from .json_loader import canonical_model_index

            record = canonical_model_index(config.DATA_DIR).get(model_name)
            if record is None:
                return True
            if environment == "indoor":
                return bool(getattr(record, "indoor", False))
            return bool(getattr(record, "outdoor", False))
        except Exception:  # pragma: no cover - 防御式
            return True

    def _recommend_all_screens(self, session_id: str, message: str) -> Optional[str]:
        """多块屏：**每块屏各跑一次推荐**，合成一条回复（一块屏一个型号）。

        实测问题：客户一次给了两块屏的规格时，只有"当前那块"会跑推荐，
        另一块没有型号 → 回复里只出现一个型号、另一个屏被漏掉。
        """
        from ..models.requirement import RequirementProfile
        from .project_items import product_model, screen_label
        from .reply_composer import reply_language

        memory_store = self.memory_store
        if memory_store is None or not hasattr(memory_store, "get_project_items"):
            return None

        items = memory_store.get_project_items(session_id)
        if len(items) < 2:
            return None

        history = self._load_history(session_id)
        language = reply_language(message)
        blocks: list = []
        changed = False

        # 【修复串台】先把**当前这块屏的实时档案**写回它的条目：
        # 之前用 items[active] 里的旧拷贝（会少掉刚说的点间距/视距），
        # 导致那一块屏拿旧档案去推荐 → 0 候选 → 兜底话术。
        live = self._stored_profile(session_id)
        if live is not None and 0 <= memory_store.get_active_item_index(session_id) < len(items):
            active_index = memory_store.get_active_item_index(session_id)
            merged_active = dict(items[active_index] or {})
            merged_active["profile"] = live.model_dump()
            items[active_index] = merged_active
            memory_store.set_project_items(session_id, items)

        for index, item in enumerate(items):
            profile_data = item.get("profile") or {}
            model_name = str(item.get("model") or "").strip()
            text = str(item.get("reply") or "").strip()
            if not model_name:
                if not profile_data:
                    # 这一块还没有需求档案（例如刚开出来还没填）也要占一段，
                    # 否则多屏回复里会只剩另一块（实测：只出现 Screen 2）。
                    blocks.append(self._screen_pending_block(index, profile_data, language))
                    continue
                try:
                    outcome = self.solution_agent.run(
                        message=message,
                        history=history,
                        session_id=session_id,
                        profile=RequirementProfile.model_validate(profile_data),
                        intent="need_query",
                    )
                except Exception as exc:  # pragma: no cover - 防御式
                    logger.warning("[%s] Per-screen recommend failed: %s", session_id, exc)
                    blocks.append(self._screen_pending_block(index, profile_data, language))
                    continue
                products = outcome.get("products") or []
                if not products:
                    blocks.append(self._screen_pending_block(index, profile_data, language))
                    continue
                model_name = product_model(products[0])
                if not model_name:
                    blocks.append(self._screen_pending_block(index, profile_data, language))
                    continue
                # 【修复串台】型号必须和这块屏的环境一致：
                # 实测室外那块屏被推了室内租赁型号 TW11-IR-P4.8 —— 那种宁可不出，
                # 也不能把不同环境的型号写给客户。
                if not self._model_matches_screen(model_name, profile_data):
                    logger.warning(
                        "[%s] 屏 %d 环境=%s 但推荐出 %s（环境不符）→ 跳过",
                        session_id, index + 1, profile_data.get("environment"), model_name,
                    )
                    blocks.append(self._screen_pending_block(index, profile_data, language))
                    continue
                text = str(outcome.get("answer") or "").strip()
                item["model"] = model_name
                item["reply"] = text
                changed = True

            label = screen_label(index, profile_data, language)
            if not text:
                text = label + model_name
            elif not text.lstrip().startswith(label.strip()):
                text = label + text
            blocks.append(text)

        if changed:
            memory_store.set_project_items(session_id, items)
        if len(blocks) < 2:
            return None
        logger.info("[%s] Multi-item: 按 %d 块屏分别推荐", session_id, len(blocks))
        return "\n\n".join(blocks)

    def _multi_item_follow_up(
        self, session_id: str, result: Dict[str, Any], message: str = ""
    ) -> Optional[str]:
        """推荐完之后：记录这一块屏的结果；第二块（及以后）推荐完就给整份汇总。

        客户口径（2026-09-18 二次确认）：**不要**在推荐后主动追问
        "By the way — is this the only screen in the project…"。
        多屏仍然支持 —— 客户自己提到第二块屏（"门口再来一块室外的屏" /
        "another screen for the entrance"）时照常开新条目，最后出汇总。
        """
        from .project_items import product_model, screen_label
        from .reply_composer import reply_language

        memory_store = self.memory_store
        if memory_store is None or not hasattr(memory_store, "get_project_items"):
            return None

        products = result.get("products") or []
        profile = self._stored_profile(session_id)
        index = memory_store.get_active_item_index(session_id)

        # 只有真的走了推荐路径才算"这一块屏推荐完了"（自由问答里的检索片段不算）
        route = str((result.get("_perf") or {}).get("route") or "")
        if route != "trigger_solution":
            return None

        customer_message = str(message or "")

        # 这一轮没有推荐出产品 → 不改动项目状态
        if not products:
            return None

        model = product_model(products[0])
        if not model:
            return None

        memory_store.record_item_recommendation(
            session_id,
            {
                "model": model,
                "profile": profile.model_dump() if profile is not None else {},
                # 这一块屏自己的推荐话术（拼"多屏一起给"的时候要用）
                "reply": str(result.get("response") or "").strip(),
            },
        )
        items = memory_store.get_project_items(session_id)
        language = reply_language(customer_message)

        # 客户口径（2026-09-18）多块屏时要**按屏幕数量**推荐：
        # 每块屏给一个型号，不再只报一块屏。这里把每块屏自己的推荐话术拼成一条，
        # 每条前面带 "Screen N (…)" 标签，客户一眼能看出哪块屏对应哪个型号。
        blocks: list = []
        for position, item in enumerate(items, start=1):
            model_name = str(item.get("model") or "").strip()
            if not model_name:
                continue
            text = str(item.get("reply") or "").strip()
            label = screen_label(position - 1, item.get("profile") or {}, language)
            if not text:
                text = label + model_name
            elif not text.lstrip().startswith(label.strip()):
                # 第一块屏当初是单屏推荐、没带标签 → 这里补上，客户才分得清哪块是哪块
                text = label + text
            blocks.append(text)
        if len(blocks) >= 2:
            result["response"] = "\n\n".join(blocks)
            result["multi_screen"] = True
            logger.info("[%s] Multi-item: 按 %d 块屏分别给出推荐", session_id, len(blocks))
            return None

        return None
