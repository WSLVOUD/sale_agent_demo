# LED RAG 关键词依赖优化 - 实施进度跟踪表

**版本**：v1.0  
**项目**：sale_agent_demo  
**最后更新**：2026-09-15

---

## 第一阶段：建立统一 Requirement Schema

### Phase 1.1：检查现有 RequirementProfile
- [x] 检查 `src/models/requirement.py` 的字段定义 ✅ 
  - 已有完整的 RequirementProfile Pydantic 模型
  - 包含所有必要字段：display_type, environment, purpose, installation 等
  - 支持 confirmed/inferred 来源标记
  - 包含 confidence 评分机制
  
- [x] 检查 `src/agents/sales/state.py` 的需求字段 ✅
  - 使用 requirement_profile（Optional[Any]）
  - 还保留了 requirements（Dict[str, Any]）作为兼容层
  - 已支持 confirmed/inferred 标记
  
- [x] 检查 `src/agents/solution/state.py` 的需求字段 ✅
  - 使用 requirement_profile（Optional[Any]）
  - 还保留了 requirement（Dict[str, Any]）
  - 已集成 RequirementProfile
  
- [x] 统一标准字段到唯一的 RequirementProfile ✅
  - 结论：使用现有 `src/models/requirement.py` 的 RequirementProfile
  - 无需重新创建，已是最完善的统一模型
  - 所有模块应优先使用 RequirementProfile 对象而非 Dict

---

## 第二阶段：建立 RequirementExtractor

### Phase 2.1：新建核心模块
- [x] 创建 `src/core/requirement_extractor.py` ✅
  - 统一入口：extract(message, previous_profile) → RequirementProfile
  - 集成规则解析（extract_slots）
  - 集成 LLM 语义理解
  - 实现结果合并（规则优先）
  - 包含冲突检测机制
  
- [x] 实现 LLM Semantic Extraction ✅
  - 仅在规则无法确定时调用 LLM
  - 明确禁止 LLM 推断客户未说的参数
  
- [x] 集成现有数字/尺寸/视距解析 ✅
  - 复用 extract_slots 中的 regex parser
  
- [x] 建立关键词 fast path ✅
  - 规则确定性高时跳过 LLM

### Phase 2.2：Purpose 标准化
- [x] 定义标准 PURPOSE enum ✅
  - 21 种标准 purpose token（retail, conference, stadium 等）
  
- [x] 实现 purpose 规范化函数 ✅
  - 直接匹配 + 模糊匹配
  - 置信度评分
  
- [x] LLM 映射到 canonical token ✅
  - 多语言支持（中英文为主）
  - 150+ 场景表达映射

### Phase 2.3：Environment 统一
- [x] 创建 `resolve_environment()` 函数 ✅
  - 4级优先级：explicit > scenario > inferred > default
  
- [x] 明确优先级规则 ✅
  - 客户明确说的环境绝不覆盖
  
- [x] 禁止自动推断模糊场景 ✅
  - concert/stage/wedding/rental 不自动推断环境

### Phase 2.4：Installation 统一
- [x] 创建 `resolve_installation()` 函数 ✅
  - 明确优先级：explicit > scenario default
  
- [x] 明确 rental vs fixed 判断规则 ✅
  - 场景默认值 + 冲突检测

---

## 第三阶段：保留确定性关键词

### Phase 3.1：关键词分类
- [ ] 保留高确定性关键词（LED/LCD/IFP/COB 等）
- [ ] 保留数字解析（P2.5/P3/P4 等）
- [ ] 保留单位转换（5m/10 meters/30ft）
- [ ] 降级 purpose 关键词为 fallback

### Phase 3.2：快速路径
- [ ] 实现 deterministic fast path
- [ ] 优先规则匹配
- [ ] 无法确定时调用 LLM

---

## 第四阶段：统一销售侧需求理解

### Phase 4.1：Sales requirement.py 改造
- [ ] 删除重复的 SCENE_CATEGORIES
- [ ] 删除独立的 indoor/outdoor 关键词
- [ ] 改调用统一 RequirementExtractor
- [ ] 保留数字解析逻辑

### Phase 4.2：Script Generator 改造
- [ ] 删除独立的 environment 推断
- [ ] 只消费 RequirementProfile
- [ ] 生成销售话术时不做场景判断

---

## 第五阶段：统一方案侧需求理解

### Phase 5.1：Solution requirement.py 改造
- [ ] 改调用统一 RequirementExtractor
- [ ] 保留 skip-understand 优化
- [ ] 避免重复 LLM extraction

### Phase 5.2：缓存语义结果
- [ ] 同一轮只生成一次 RequirementProfile
- [ ] Sales、Solution、Gate 共用结果
- [ ] 避免多次 LLM 调用

---

## 第六阶段：RAG 层改造

### Phase 6.1：query_understanding.py 兼容层
- [ ] 保留现有关键词表
- [ ] 创建 `canonicalize_requirement()` 函数
- [ ] `_detect_purpose()` 降级为 fallback
- [ ] 统一 canonical purpose 输出

### Phase 6.2：parameter_inference.py 改造
- [ ] 明确只负责工程参数推断
- [ ] 不重新识别客户场景
- [ ] 消费已确定的 RequirementProfile

---

## 第七阶段：Router 改造

### Phase 7.1：删除重复场景词表
- [ ] 删除 `_SCENE_KEYWORDS` 主判断职责
- [ ] 改用 RequirementProfile 判断
- [ ] 基于 `has_meaningful_context()` 路由

---

## 第八阶段：Recommendation Gate 重定义

### Phase 8.1：结构化字段检查
- [ ] 改为检查 RequirementProfile 完整性
- [ ] 定义 required_fields
- [ ] 定义 optional_fields

### Phase 8.2：来源验证
- [ ] confirmed 字段可打开 Gate
- [ ] inferred 字段需要业务规则允许
- [ ] 冲突场景 block Gate

### Phase 8.3：冲突检测
- [ ] 检测 environment 冲突
- [ ] 检测 purpose 冲突
- [ ] 检测 display_type 冲突

---

## 第九阶段：建立 confirmed/inferred 双来源

### Phase 9.1：数据结构扩展
- [ ] 扩展 RequirementProfile 字段
- [ ] 添加 source 标记
- [ ] 添加 confidence 评分
- [ ] 添加 conflict 列表

---

## 第十阶段：建立验收测试集

### Phase 10.1：Golden Dataset
- [ ] 创建 `tests/requirement_extraction/golden_cases.json`
- [ ] 建立 100+ 真实/模拟客户表达
- [ ] 覆盖所有 purpose 类型
- [ ] 覆盖所有 environment 类型

### Phase 10.2：Purpose 测试
- [ ] 测试 retail 场景（至少 10 个变体）
- [ ] 测试 stadium 场景（至少 10 个变体）
- [ ] 测试 conference 场景（至少 10 个变体）
- [ ] 测试其他场景

### Phase 10.3：Environment 测试
- [ ] 测试 outdoor 表达
- [ ] 测试 indoor 表达
- [ ] 测试模糊场景
- [ ] 测试冲突检测

### Phase 10.4：Installation 测试
- [ ] 测试 rental 表达
- [ ] 测试 fixed 表达
- [ ] 测试默认值

### Phase 10.5：参数解析测试
- [ ] 测试尺寸提取
- [ ] 测试距离转换
- [ ] 测试单位识别

### Phase 10.6：多语言测试
- [ ] 中文场景
- [ ] 英文场景
- [ ] 德文场景
- [ ] 法文场景
- [ ] 西班牙文场景
- [ ] 俄文场景
- [ ] 日文场景

### Phase 10.7：缺失关键词测试
- [ ] 30%+ 场景不在当前关键词表
- [ ] 验证 LLM 仍能正确理解

---

## 第十一阶段：性能优化

### Phase 11.1：缓存机制
- [ ] LLM 结果缓存
- [ ] 同轮共用 RequirementProfile

### Phase 11.2：快速路径
- [ ] 明确规则命中时不调用 LLM
- [ ] 测量性能改进

---

## 第十二阶段：冲突检测

### Phase 12.1：冲突定义
- [ ] 定义 environment 冲突
- [ ] 定义 purpose 冲突
- [ ] 定义 display_type 冲突

### Phase 12.2：优先级规则
- [ ] 客户明确事实 > 确定性规则 > LLM推断 > 业务默认

---

## 第十三阶段：验收指标

### Phase 13.1：准确率指标
- [ ] Purpose accuracy ≥ 95%
- [ ] Environment accuracy ≥ 98%
- [ ] Installation accuracy ≥ 98%
- [ ] Display type accuracy ≥ 98%
- [ ] Size extraction ≥ 98%
- [ ] Distance extraction ≥ 98%

### Phase 13.2：关键词独立性
- [ ] 30%+ 场景不含原始关键词
- [ ] 仍正确识别

### Phase 13.3：Gate 一致性
- [ ] 同一 Profile 无论来源如何
- [ ] 产品推荐结果一致

### Phase 13.4：回归测试
- [ ] 现有功能不破坏
- [ ] 推荐结果稳定

---

## 第十四阶段：代码清理

### Phase 14.1：删除重复代码
- [ ] 删除 Sales 中的场景分类
- [ ] 删除 Script Generator 中的环境推断
- [ ] 删除 Router 中的独立关键词表
- [ ] 删除 Solution 中的重复 extraction

### Phase 14.2：文档更新
- [ ] 更新架构文档
- [ ] 更新模块职责文档
- [ ] 更新 API 文档

---

## 最终验收清单

- [ ] 所有 Phase 完成
- [ ] 100+ Golden Cases 通过
- [ ] 7 种语言测试通过
- [ ] 准确率指标达成
- [ ] 回归测试通过
- [ ] 性能测试通过
- [ ] 文档完整
- [ ] 代码审查通过

---

## 优先级标记

- **P0（必须做）**：Phase 1-8
- **P1（强烈建议）**：Phase 9-10
- **P2（优化）**：Phase 11-14

---

**预计工作量**：
- P0：60-80 小时
- P1：40-60 小时
- P2：20-40 小时
- 总计：120-180 小时

---

**开始日期**：2026-09-15  
**状态**：未开始
