"""Coordinate background vector-store rebuilds and agent replacement."""
from __future__ import annotations

from typing import Any, Callable


async def rebuild_vectorstore_task(
    task_id: str,
    *,
    update_progress: Callable[[str, int, str], None],
    on_ready: Callable[[Any, Any], None],
) -> dict:
    """Rebuild retrieval resources, then hand the ready instances to the API."""
    from src.agents.sales.runner import SalesAgentRunner
    from src.agents.solution.runner import SolutionAgentRunner
    from src.config import config
    from src.core.embeddings import recreate_vectorstore
    from src.orchestrator import DualAgentOrchestrator
    from src.rag.corpus import build_retrieval_corpus, corpus_as_dicts

    update_progress(task_id, 10, "加载 Model 级产品数据...")
    corpus_documents = build_retrieval_corpus(config.DATA_DIR)
    documents = corpus_as_dicts(corpus_documents)

    update_progress(task_id, 30, "创建向量库...")
    new_vectorstore = recreate_vectorstore(corpus_documents)
    active_dir = getattr(new_vectorstore, "_active_persist_dir", config.VECTORSTORE_DIR)
    config.VECTORSTORE_DIR = active_dir
    update_progress(task_id, 40, f"向量库已重建: {active_dir}")

    update_progress(task_id, 50, "初始化 Sales Agent...")
    new_sales_agent = SalesAgentRunner(sales_search=new_vectorstore)

    update_progress(task_id, 70, "初始化 Solution Agent...")
    new_solution_agent = SolutionAgentRunner(new_vectorstore, documents)

    update_progress(task_id, 85, "初始化编排器...")
    new_orchestrator = DualAgentOrchestrator(new_sales_agent, new_solution_agent)

    update_progress(task_id, 95, "更新全局状态...")
    on_ready(new_vectorstore, new_orchestrator)

    update_progress(task_id, 100, "完成")
    return {
        "message": f"向量库重建成功，文档数: {len(documents)}",
        "document_count": len(documents),
    }
