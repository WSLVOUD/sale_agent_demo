"""计划 v2.9.4 §六（图片输入流程）：Vision 的 display_type 必须真正接进第一层产品类型判断。

本轮执行前实测到的缺口（v2.9.3 遗留）：

  - Vision 侧**已经**能抽出 display_type（``src/vision/extractor.py::_display_type``），
    Product Type Router 也**已经**写了"③ 图片（Vision）判断"分支；
  - 但生产路径上 ``state["vision"]`` 从来没被写过 —— Orchestrator 只把
    ``has_vision=True/False`` 这个布尔量传给 Sales Agent，图片识别出的类型
    在到达 classify 之前就丢了；
  - 结果：图片轮永远走不到路由器的图片分支（事实上的死代码），客户发一张 LED 的
    照片，得到的还是"你要 LED 还是 LCD？"，而不是 §六 要求的
    "图片里看到的是 LED，是按这个继续吗？"。

所以这一组的验收口径就是计划 §六 的四条：
  1. Vision 判断出 LED / LCD → 路由给出 INFERRED / VISION + 要客户确认；
  2. Vision 判断出 IFP → LCD + ``subtype=IFP``（§十四：IFP 属于 LCD）；
  3. 客户确认 → 锁定（CONFIRMED / CUSTOMER / locked），再进对应链路；
  4. Vision 无法判断 → 不硬猜（照旧询问）。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _route(message: str, **kwargs):
    from src.dialogue.product_type_router import route_display_type

    return route_display_type(message, **kwargs)


class _RecordingVisionExtractor:
    """预设图片识别结果（走真实的 integration 合并逻辑）。"""

    def __init__(self, payloads):
        from src.vision.extractor import VisionExtractor

        self.results = [VisionExtractor.from_payload(payload) for payload in payloads]

    def extract_many(self, images, session_id="", customer_text="", **_kwargs):
        return list(self.results)


class _RecordingSalesAgent:
    def __init__(self):
        self.calls = []

    def run(self, session_id, message, has_vision=False, **kwargs):
        self.calls.append({"session_id": session_id, "message": message, "has_vision": has_vision, **kwargs})
        return {
            "intent": "need_query",
            "next_action": "ask",
            "response": "Thanks for the photo.",
            "requirements": {},
            "products": [],
        }


class _StubSolutionAgent:
    def run(self, *args, **kwargs):
        return {"answer": "Sure.", "products": []}


class TestVisionTypeIsRouted:

    def test_vision_led_is_inferred_and_asks_for_confirmation(self):
        decision = _route("here is a photo of the screen", vision_display_type="LED")
        assert decision.display_type == "LED"
        assert decision.status == "INFERRED"
        assert decision.source == "VISION"
        assert decision.ask_customer is True, "图片判断必须让客户确认（计划 §六）"

    def test_vision_ifp_is_lcd_with_the_ifp_subtype(self):
        """计划 §十四：IFP 属于 LCD —— 图片认出 IFP 时，第一层也只能给 LCD。"""
        decision = _route("here is a photo of the screen", vision_display_type="IFP")
        assert decision.display_type == "LCD"
        assert decision.subtype == "IFP"
        assert decision.source == "VISION"
        assert decision.ask_customer is True

    def test_vision_that_cannot_tell_is_not_guessed(self):
        decision = _route("here is a photo of the screen", vision_display_type="")
        assert decision.display_type == "UNKNOWN"
        assert decision.ask_customer is True


class TestImageTurnCarriesTheVisionType:

    def test_orchestrator_hands_the_vision_type_to_the_sales_agent(self, monkeypatch):
        """图片轮：Vision 识别出的产品类型必须带到 Sales Agent（否则路由器拿不到）。"""
        import src.vision.integration as integration
        from src.memory.store import memory
        from src.orchestrator import DualAgentOrchestrator

        monkeypatch.setattr(
            integration,
            "get_vision_extractor",
            lambda: _RecordingVisionExtractor(
                [{"display_type": {"value": "LED", "source": "vision_explicit", "confidence": 0.95}}]
            ),
        )

        session_id = "v294-vision-wiring"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        try:
            sales = _RecordingSalesAgent()
            orchestrator = DualAgentOrchestrator(
                sales_agent=sales, solution_agent=_StubSolutionAgent()
            )
            monkeypatch.setattr(orchestrator, "memory_store", memory)
            orchestrator.process_message(
                "can you recommend something like this?", session_id, images=[b"img"]
            )

            assert sales.calls, "Sales Agent 必须被调用"
            vision = sales.calls[0].get("vision") or {}
            assert vision.get("display_type") == "LED", (
                "图片识别出的产品类型必须一路带到 Sales Agent（state['vision']），"
                "否则路由器的图片分支在真实链路上永远走不到"
            )
        finally:
            memory.clear(session_id)

    def test_sales_runner_puts_the_vision_payload_into_the_graph_state(self):
        from src.agents.sales.runner import SalesAgentRunner
        from src.memory.store import memory

        captured = {}

        class _FakeGraph:
            def invoke(self, state):
                captured.update(state)
                return dict(state, response="Understood.")

        session_id = "v294-runner-vision"
        memory.clear(session_id)
        try:
            runner = SalesAgentRunner(memory_store=memory)
            runner.graph = _FakeGraph()
            runner.run(
                session_id,
                "here is the screen I want",
                vision={"display_type": "LCD", "reason": "looks like a video wall"},
            )
            assert (captured.get("vision") or {}).get("display_type") == "LCD"
        finally:
            memory.clear(session_id)


class TestVisionTypeConfirmationLoop:

    def test_customer_confirms_the_vision_type_then_it_locks_and_enters_led(self):
        """计划 §六 闭环：图片 → 推断 → 客户确认 → 锁定 → 进 LED 成熟链路。"""
        from src.dialogue.product_router import LED_ENTRY, route_product_domain

        proposed = _route("here is a photo of my screen", vision_display_type="LED")
        confirmed = _route("yes, that's right", current=proposed)

        assert confirmed.display_type == "LED"
        assert confirmed.status == "CONFIRMED"
        assert confirmed.source == "CUSTOMER"
        assert confirmed.locked is True
        assert route_product_domain(confirmed.display_type) == LED_ENTRY

    def test_customer_denies_the_vision_type_and_it_is_rerouted(self):
        """计划 §六：客户否认 → 不强行沿用图片判断，按客户说的重新路由。"""
        proposed = _route("here is a photo of my screen", vision_display_type="LED")
        corrected = _route("no, it is an LCD video wall", current=proposed)

        assert corrected.display_type == "LCD"
        assert corrected.source == "CUSTOMER"
        assert corrected.locked is True

    def test_customer_lets_the_image_decide(self):
        """计划 §六：客户说"不知道" → 采用 AI（图片）判断，不再反复问。"""
        proposed = _route("here is a photo of my screen", vision_display_type="LCD")
        adopted = _route("i don't know", current=proposed)

        assert adopted.display_type == "LCD"
        assert adopted.status == "CONFIRMED"
        assert adopted.locked is True
