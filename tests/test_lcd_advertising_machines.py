"""客户口径（2026-09-28）：data/LCD display.txt 新增的 7 个系列都是**广告机**。

来源（客户自己加进 txt 的原文）：

    DS-W   Wall Mounted Digital Display（室内壁挂广告机）
    DS-T55 Transparent side Floor Standing Digital Signal（室内落地广告机）
    DS-TB55 同上 + RS-232 遥控
    DS-P   Intelligent floor-standing digital signage（室内落地数字标牌）
    DS-D55 Double-Sided Digital Displays（双面落地广告机）
    DS-O   OD Wall Mounted Outdoor Digital Display（户外壁挂广告机）
    DS-M   Floor Standing Touch Screen Display（落地触控查询广告机）

要求：进 JSON、进检索语料，并且**标注清楚是广告机**
（category=advertising_machine / category_label=广告机）。
资料里没有的字段一律留空（不编造）。
"""
import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

DATA_DIR = os.path.join(project_root, "data")
NEW_SERIES = ("DS-W", "DS-T55", "DS-TB55", "DS-P", "DS-D55", "DS-O", "DS-M")


def _products():
    with open(os.path.join(DATA_DIR, "lcd_products.json"), encoding="utf-8") as handle:
        return json.load(handle)["products"]


def _advertising():
    return [item for item in _products() if item.get("category") == "advertising_machine"]


class TestAdvertisingMachinesAreInTheJson:
    def test_all_seven_new_series_are_present(self):
        ids = {item["product_id"] for item in _advertising()}

        for series in NEW_SERIES:
            assert any(model.startswith(series) for model in ids), (series, sorted(ids))

    def test_every_new_product_is_marked_as_advertising_machine(self):
        ads = _advertising()

        assert len(ads) == 26, len(ads)
        for item in ads:
            assert item["category_label"] == "广告机", item["product_id"]
            assert "advertising-machine" in item["features"], item["product_id"]
            assert "digital-signage" in item["features"], item["product_id"]
            assert "广告机" in item["features"], item["product_id"]

    def test_old_products_are_not_relabelled(self):
        others = [i for i in _products() if i.get("category") != "advertising_machine"]

        assert len(others) == 21, len(others)
        assert {i["product_id"] for i in others} == {
            "P43", "P49", "P55", "P65", "P75", "P85", "P98", "P110",
            "H4630LN-B", "H4930LN-B", "H5530LN-B", "H5518LN/HN-B",
            "H5510LN/HN-B", "H6530LN-B", "H7530LN-B",
            "P55TM", "P65TM", "P75TM", "P86TM", "P98TM", "P110TM",
        }

    def test_outdoor_model_keeps_its_environment(self):
        outdoor = next(i for i in _products() if i["product_id"] == "DS-O-75")

        assert outdoor["environment"] == ["outdoor"], outdoor
        assert "IP55-waterproof" in outdoor["features"], outdoor["features"]

    def test_floor_standing_models_are_marked_floor_standing(self):
        for model in ("DS-T55-55", "DS-TB55-55", "DS-P-55", "DS-D55", "DS-M-55"):
            item = next(i for i in _products() if i["product_id"] == model)
            assert item["installation"] == ["floor_standing"], (model, item["installation"])

    def test_missing_facts_are_left_empty_not_invented(self):
        """DS-T55 / DS-TB55 资料里没给对比度 / 24x7 / 寿命 → 不许编。"""
        for model in ("DS-T55-43", "DS-T55-55", "DS-TB55-43", "DS-TB55-55"):
            item = next(i for i in _products() if i["product_id"] == model)
            assert "contrast_ratio" not in item, (model, item.get("contrast_ratio"))
            assert "operation_hours" not in item, (model, item.get("operation_hours"))
            assert "service_life_hours" not in item, (model, item.get("service_life_hours"))

    def test_double_sided_series_has_a_data_note(self):
        item = next(i for i in _products() if i["product_id"] == "DS-D55")

        assert "Model" in item["notes"], item["notes"]  # 备注说明资料里没有型号行


class TestAdvertisingMachinesReachTheRetrievalCorpus:
    def test_corpus_carries_the_advertising_machine_label(self):
        from src.config import config
        from src.rag.corpus import build_retrieval_corpus

        documents = build_retrieval_corpus(config.DATA_DIR)
        ads = [d for d in documents if d.metadata.get("category") == "advertising_machine"]

        assert len(ads) == 26, len(ads)
        sample = next(d for d in ads if d.metadata["model"] == "DS-W-32")
        assert sample.metadata["category_label"] == "广告机"
        assert "广告机" in sample.page_content, sample.page_content[:200]
        assert "advertising_machine" in sample.page_content, sample.page_content[:200]

    def test_corpus_counts_add_up(self):
        from src.config import config
        from src.rag.corpus import build_retrieval_corpus, corpus_summary

        summary = corpus_summary(build_retrieval_corpus(config.DATA_DIR))

        assert summary["documents"] == 113, summary
        assert summary["indoor"] + summary["outdoor"] == 113, summary
