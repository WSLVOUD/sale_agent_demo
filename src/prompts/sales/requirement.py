"""Sales requirement extraction prompt templates."""

REQUIREMENT_PROMPT = """Extract LED and LCD display product requirements from the conversation. Return as JSON:
{{
  "location_type": "indoor" | "outdoor" | null,
  "usage": "user's exact scene description" | null,
  "viewing_distance": "viewing distance description, e.g. '3m', '10-20m', 'far distance'" | null,
  "size": "size description" | null,
  "brightness": "brightness requirement" | null,
  "resolution": "resolution requirement" | null,
  "display_type": "LED" | "LCD" | null,
  "additional_requirements": ["other special requirements mentioned by user"]
}}

Core principle: extract only fields explicitly stated by the user — do not guess or infer.
If a field is not mentioned by the user, it must be null.

IMPORTANT: Always respond in English regardless of customer's language.

Return JSON only, nothing else."""


def get_requirement_prompt(existing: dict = None, conversation: str = "") -> str:
    """Format the requirement extraction prompt."""
    existing_str = "\n".join([f"{k}: {v}" for k, v in (existing or {}).items() if v]) or "None"
    return f"""Existing requirements:
{existing_str}

Current conversation:
{conversation}

{REQUIREMENT_PROMPT}"""
