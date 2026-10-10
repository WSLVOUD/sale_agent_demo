"""客户**指名型号**时的解析（LED / LCD / IFP 通用）。

客户口径（2026-10）：

    客户: i need a TW-OD-11
    AI  : … are you after an LED display, or something like an LCD video wall?  ← 还在问 LED/LCD
    客户: LED
    AI  : … Is this going to be set up indoors or outdoors?                    ← 还在问室内外
    客户: outdoor

客户已经指名型号，系统却把型号丢掉、从头问起 —— **没有接住**。
而且 `TW-OD-11` 是 `TW11-OD` 的错序写法：目录里 `TW11-OD` 是真实系列。

客户口径：**LED 和 LCD 两条链路都要支持** —— 客户直接说型号时，
要**围绕这个型号**问需求，型号本身已经确定的属性不许再问。

目录来源：``load_structured_products(config.DATA_DIR)`` —— 一次覆盖
LED + LCD + IFP 共 61 条，每条都带 ``product_id`` / ``display_type`` /
``environment`` / ``is_rental``。（注意：``load_canonical_models`` **只有 LED**
的 56 个展开型号，用它做解析会导致 LCD 永远解析不出来。）

匹配算法（可解释、不猜）：
  1. 从客户原话切出型号样式 token；
  2. 按**分隔符**切块（``TW11-OD-P4`` → ``['TW11','OD','P4']``）；
  3. token 的每个块都能在某个已知名字里找到对应块（相等或互为子串）；
  4. 命中里取"块数不少于客户所写、且最短"的那个 → 它就是客户指的系列。
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODEL_TOKEN_RE = re.compile(r"\b[A-Za-z]{1,6}[\w.]*(?:-[A-Za-z0-9.]+){0,4}\b")
_PITCH_BLOCK_RE = re.compile(r"^P\d+(?:\.\d+)?[A-Z]?$")


def _blocks(text: str) -> List[str]:
    """按分隔符切块：TW11-OD-P4 → ['TW11','OD','P4']。

    **不能**用"字母/数字"正则切（会把 TW11 拆成 TW + 11，系列名就拼错了）。
    """
    return [b for b in re.split(r"[^A-Z0-9./]+", str(text or "").upper()) if b]


def _is_pitch_block(block: str) -> bool:
    return bool(_PITCH_BLOCK_RE.match(block))


def _digit_fuzzy(a: str, b: str) -> bool:
    """两串纯数字是否"只差一个错"：数字顺序颠倒，或打错 / 多打 / 少打一位。

    实测（客户口径 2026-10）：客户写 ``TW11-3261-P4``，目录里是 ``TW11-3216-P4``
    —— **数字颠倒了**。这种一眼就是同一个型号，不该判成"不认识"，
    更不能退回去问"你要 LED 还是 LCD"。
    """
    if a == b:
        return True
    if len(a) == len(b):
        if sorted(a) == sorted(b):  # 顺序颠倒：3261 vs 3216
            return True
        return sum(1 for x, y in zip(a, b) if x != y) == 1  # 打错一位
    if abs(len(a) - len(b)) == 1:  # 多打 / 少打一位
        longer, shorter = (a, b) if len(a) > len(b) else (b, a)
        for i in range(len(longer)):
            if longer[:i] + longer[i + 1 :] == shorter:
                return True
    return False


def _block_score(needle: str, haystack: str) -> int:
    """对上的程度：0=对不上，3=完全相同，2=父子串，1=数字笔误。"""
    if not needle or not haystack:
        return 0
    if needle == haystack:
        return 3
    if needle in haystack or haystack in needle:
        return 2
    if needle.isdigit() and haystack.isdigit() and _digit_fuzzy(needle, haystack):
        return 1
    return 0


def _block_matches(needle: str, haystack: str) -> bool:
    """客户写的块能否对上目录块（相等 / 互为子串 / 数字笔误）。"""
    return _block_score(needle, haystack) > 0


def candidate_tokens(message: str) -> List[str]:
    """从客户原话里切出疑似型号的 token（按出现顺序，去重）。"""
    out: List[str] = []
    for match in _MODEL_TOKEN_RE.finditer(str(message or "")):
        token = match.group(0).strip(" -.")
        if not token or not re.search(r"\d", token):
            continue
        if token.upper() not in [t.upper() for t in out]:
            out.append(token)
    return out


@lru_cache(maxsize=1)
def _catalog() -> tuple:
    """(已知名字 → 所属系列记录) 的条目表，LED / LCD / IFP 都在内。

    展开型号（``TW11-OD-P4``）也会指向它的系列（``TW11-OD``）。
    """
    entries: List[Dict[str, Any]] = []
    try:
        from ..config import config
        from .json_loader import load_canonical_models, load_structured_products

        records: Dict[str, Dict[str, Any]] = {}
        for product in load_structured_products(config.DATA_DIR):
            pid = str(getattr(product, "product_id", "") or "").strip()
            if not pid:
                continue
            env = getattr(product, "environment", None) or []
            if isinstance(env, str):
                env = [env]
            records[pid.upper()] = {
                "series": pid,
                "display_type": str(getattr(product, "display_type", "") or ""),
                "environment": [str(e).lower() for e in env if str(e).strip()],
                "is_rental": getattr(product, "is_rental", None),
            }
        try:
            for model in load_canonical_models(config.DATA_DIR):
                name = str(getattr(model, "model", "") or "").strip()
                series = str(getattr(model, "series", "") or "").strip()
                series = series.replace(" series", "").strip()
                if name and series.upper() in records:
                    entries.append({"name": name, "record": records[series.upper()]})
        except Exception as exc:  # pragma: no cover - 防御式
            logger.warning("Canonical models unavailable: %s", exc)
        for rec in records.values():
            entries.append({"name": rec["series"], "record": rec})
    except Exception as exc:  # pragma: no cover - 防御式
        logger.warning("Model catalog unavailable: %s", exc)
    return tuple(entries)


def _related_to_message(name: str, message: str) -> bool:
    """模型给的型号名与客户原话**有没有真实关系**（防止"换个型号"式捏造）。

    为什么必须有这道闸门（实测）：

        客户: 我想要 T65Omni 那款       ← T65Omni 不在目录里
        模型: DS-W-65                  ← 自作主张"对齐"到一个毫不相干的型号 ❌

    规则（宁可不认，也不许认错）：
      · 名字里有 **≥2 字符的字母块** → 必须有字母块与原话重叠（父子串）才行；
        字母块与原话零重叠 → 拒绝（`T65Omni` vs `DS-W-65` 就是这样被拒的）。
      · 名字的字母块全是单字符（`P65` / `P110` 这类 LCD 短名）→ 允许名字整体
        出现在原话里。
    """
    import re as _re

    raw = str(message or "").upper()
    flat = _re.sub(r"[^A-Z0-9]", "", raw)
    name_flat = _re.sub(r"[^A-Z0-9]", "", str(name or "").upper())
    if name_flat and name_flat in flat:
        return True

    name_blocks = [b for b in _blocks(name) if b]
    msg_blocks = [b for b in _blocks(message) if b]
    # 注意：**不能**用 b.isalpha() 判"字母块" —— `TW11` 含数字会被误判成非字母块，
    # 于是落到下面那条不认错序数字的规则上，把 `TW11-3261-P4 → TW11-3216` 这种
    # 正确案例也一起拒掉（实测踩过）。只要块里**有字母**且长度≥2 就算有辨识度。
    strong_letters = [b for b in name_blocks if len(b) >= 2 and any(c.isalpha() for c in b)]
    if strong_letters:
        # 有"有辨识度的块" → 必须与原话重叠，否则不认
        return any(
            b in mb or mb in b
            for b in strong_letters
            for mb in msg_blocks
            if mb
        )
    # 只有单字符字母块（P65 / P110）→ 需要名字整体出现（上面已查）或数字块对上
    digits = [b for b in name_blocks if b.isdigit()]
    msg_digits = "".join(b for b in msg_blocks if b.isdigit())
    return bool(digits) and all(d in msg_digits for d in digits)


_EXTRACT_PROMPT = """你在帮销售判断：客户**这句话**是不是在指名某个具体型号。

最近的对话（越靠下越新）：
{recent}

客户最新一句：{message}

我们目录里现有的型号（供你对齐，**只准从这里选**）：
{catalog}

判断要求：
1. 客户**确实在说某个具体型号**时（含写法有误、字母/数字顺序颠倒、拼错、只写系列名），
   返回**目录里最接近的那一个**型号名（原样，不要解释）。
2. 客户只是泛泛地说"显示屏 / 大屏 / LED 屏 / 想要一块屏"——**没有指名型号**，返回 NONE。
3. 拿不准就返回 NONE。**绝对不许**返回目录里没有的型号名，也不许编造。
4. 只返回型号名或 NONE，不要任何其他文字。
"""


@lru_cache(maxsize=256)
def _llm_mentioned_model(message: str, session_id: str = "") -> str:
    """让模型读整句话 + 上下文，判断客户指的型号；拿不到就返回空串（调用方回退正则）。

    客户口径（2026-10）：**不要只靠关键词**识别"客户是否想要指定型号"——
    要理解整句的句意和上下文。实测客户写 ``TW11-3261-P4``（数字颠倒），
    正则能勉强救回来，但换一种说法就未必；交给模型理解更稳。
    """
    text = str(message or "").strip()
    if not text:
        return ""
    recent = ""
    try:
        if session_id:
            from ..memory.history_window import dialogue_window_text

            recent = dialogue_window_text(session_id, max_items=8) or ""
    except Exception:  # pragma: no cover - 防御式
        recent = ""
    names = []
    for entry in _catalog():
        name = str(entry.get("name") or "")
        if name and name not in names:
            names.append(name)
    if not names:
        return ""
    try:
        from ..core.llm import get_llm

        prompt = _EXTRACT_PROMPT.format(
            recent=str(recent or "")[:1200] or "(none)",
            message=text[:300],
            catalog=", ".join(names[:80]),
        )
        response = get_llm(temperature=0.0).invoke(prompt)
        raw = (response.content if hasattr(response, "content") else str(response)) or ""
        answer = " ".join(str(raw).split()).strip().strip('"').strip()
        if not answer or answer.upper().startswith("NONE"):
            return ""
        # 只接受目录里真实存在的名字（模型被明确要求只从这里选，这里再兜一层）
        upper = answer.upper()
        for name in names:
            if name.upper() == upper:
                # 【防捏造】还得与客户原话**真有关系**，否则宁可当作"没指名型号"。
                # 实测：客户说 T65Omni（目录里没有），模型自作主张给 DS-W-65。
                if not _related_to_message(name, text):
                    logger.warning(
                        "模型给的型号 %r 与客户原话无关（.%r）→ 按未指名处理",
                        name, text[:40],
                    )
                    return ""
                return name
        logger.info("模型给出的型号 %r 不在目录里，按未指名处理", answer[:40])
        return ""
    except Exception as exc:  # pragma: no cover - 模型不可用
        logger.warning("Model-mention understanding failed: %s", exc)
        return ""


def resolve_mentioned_model(
    message: str,
    session_id: str = "",
) -> Optional[Dict[str, Any]]:
    """客户指名的型号 → 目录里的系列及其**已确定属性**。

    顺序：**先让模型读整句 + 上下文理解**（客户口径 2026-10），
    模型不可用或判断不出时，再用正则切 token 兜底。

    返回 ``{token, series, display_type, environment, is_rental}``；
    解析不出来返回 ``None``（调用方据此**就型号向客户确认**，不要重头问一遍）。
    """
    tokens: List[str] = []
    understood = _llm_mentioned_model(str(message or ""), session_id)
    if understood:
        tokens.append(understood)
    for token in candidate_tokens(message):
        if token.upper() not in [t.upper() for t in tokens]:
            tokens.append(token)
    entries = _catalog()
    if not tokens or not entries:
        return None

    for token in tokens:
        needle = _blocks(token)
        if not needle:
            continue
        scored: List[tuple] = []
        for entry in entries:
            hay = _blocks(entry["name"])
            if not hay:
                continue
            per = [max((_block_score(n, h) for h in hay), default=0) for n in needle]
            if all(s > 0 for s in per):
                scored.append((sum(per), entry))
        if not scored:
            continue
        # 越像越靠前（完全相同 > 父子串 > 数字笔误）；同等像度取块数更少的（更接近系列）
        scored.sort(
            key=lambda item: (
                -item[0], len(_blocks(item[1]["name"])), len(item[1]["name"])
            )
        )
        hits = [entry for _, entry in scored]
        # 取"块数不少于客户所写、且最短"的命中 —— 它才是客户指的那个系列
        hits.sort(key=lambda e: (len(_blocks(e["name"])), len(e["name"])))
        chosen = next(
            (e for e in hits if len(_blocks(e["name"])) >= len(needle)), hits[0]
        )
        # 【防捏造】正则兜底这条路同样要过关系校验 —— 否则"数字块恰好在原话里出现过"
        # 就会乱认型号。实测：客户说 T65Omni，`DS-W-65` 的数字块 65 是 T65OMNI 的子串，
        # 于是被认成了 DS-W-65。
        if not _related_to_message(chosen["name"], str(message or "")):
            logger.info(
                "型号 %r 与客户原话无关 → 按未指名处理（input=%r）",
                chosen["name"], str(message or "")[:40],
            )
            continue
        rec = chosen["record"]
        result = {
            "token": token,
            "series": rec["series"],
            "display_type": rec["display_type"],
            "environment": (rec["environment"] or [""])[0],
            "is_rental": rec["is_rental"],
            "matched_name": chosen["name"],
        }
        logger.info(
            "客户指名型号 %r → %s（%s / %s）",
            token, result["series"], result["display_type"], result["environment"] or "-",
        )
        return result
    return None


__all__ = ["resolve_mentioned_model", "candidate_tokens"]
