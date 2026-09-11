# First Contact 流程说明

## 概览

当客户首次发送消息时，系统会自动触发 First Contact 流程，包含三个步骤：

1. **动态自我介绍** - AI根据公司信息生成个性化的自我介绍
2. **发送素材** - 自动发送案例视频和产品目录
3. **过渡语** - AI生成友好的过渡消息，引导进入正式对话

## 流程细节

### 1. 动态自我介绍

AI会根据 `data/company_profile.txt` 中的信息动态生成自我介绍，包括：
- 销售人员姓名和职位（Mike）
- 公司名称和成立年份（iSEMC, 2013）
- 核心业务（室内LED显示屏）
- 关键优势（12年经验、5000+项目、24/7服务）
- 友好地询问客户姓名和需求

**特点：**
- 每次生成都略有不同，不是固定模板
- 会根据客户的首次消息调整侧重点
- 始终使用英语（符合国际销售场景）
- 语气友好、自然，像真人对话

**示例输出：**
> "Hi there, I'm Mike with iSEMC, and it's great to connect with you. We specialize in indoor LED displays, and with 12 years of international service experience behind us, we've helped plenty of businesses find the right screen for their spaces. What really sets us apart is our professional team's 24/7 service and support, so you're never left figuring things out alone. Could I get your name and a few details about your conference room needs?"

### 2. 发送素材

自动发送预设的素材（当前配置）：
- **Case Video 1** (`case_video_01.mp4`)
- **Case Video 2** (`case_video_02.mp4`)

**特点：**
- 单个素材发送失败不会阻塞整个流程
- 素材路径：`data/first_contact/`
- 前端URL：`/static/first_contact/`

### 3. 过渡语

AI生成一句友好的过渡消息，承接素材发送和正式对话：

**特点：**
- 1-2句话，简洁有力
- 确认已发送素材
- 鼓励客户提问或分享兴趣点
- 语气轻松、不生硬
- 每次生成都不同

**示例输出：**
> "Just sent over those case videos and our product catalog so you can get a feel for what iSEMC is all about. Let me know what catches your eye and I'm happy to dig into any details."

或：
> "Just shared some case videos and our product catalog so you can get a feel for what iSEMC is all about. Let me know if anything catches your eye or if you would like to dive deeper into any of it."

## 配置文件

### 公司信息
路径：`data/company_profile.txt`

修改此文件会自动影响AI生成的自我介绍内容。

### 素材配置
路径：`src/first_contact/handler.py`

```python
ASSETS = [
    FirstContactAsset(
        filename="case_video_01.mp4",
        name="Case Video 1",
        asset_type="video",
    ),
    FirstContactAsset(
        filename="case_video_02.mp4",
        name="Case Video 2",
        asset_type="video",
    ),
]
```

## 技术实现

### 返回结构
```python
FirstContactResult(
    session_id: str,
    intro_text: str,              # 自我介绍文本
    asset_results: list,           # 素材发送结果
    transition_text: str,          # 过渡语文本
    intro_success: bool,           # 自我介绍是否生成成功
    transition_success: bool,      # 过渡语是否生成成功
    all_success: bool,             # 所有素材是否发送成功
    should_continue: bool,         # 是否继续进入Sales Agent
)
```

### 消息格式
```python
[
    {"role": "assistant", "content": "Hi, I'm Mike..."},
    {"role": "assistant", "content": "[video] Case Video 1", "asset_type": "video", "asset_url": "..."},
    {"role": "assistant", "content": "[video] Case Video 2", "asset_type": "video", "asset_url": "..."},
    {"role": "assistant", "content": "Just sent over those case videos..."},
]
```

## 降级方案

如果AI生成失败（网络问题、API错误等），系统会自动使用降级方案：

**自我介绍降级：**
> "Hi, I'm Mike. I work at iSEMC as Sales Manager. May I know your name and what you're looking for?"

**过渡语降级：**
> "Thank you for reaching out to iSEMC. I've shared some materials to show you what we're all about."

## 测试

运行测试脚本：
```bash
python test_intro.py
```

这会模拟三个不同的客户消息，展示完整的First Contact流程。
