/**
 * LED产品助手 - 前端应用
 */

class LEDChatApp {
    constructor() {
        this.apiBase = window.location.origin;
        this.sessionId = this.generateSessionId();
        this.isLoading = false;
        
        // DOM 元素
        this.messagesArea = document.getElementById('messages-area');
        this.messageInput = document.getElementById('message-input');
        this.sendBtn = document.getElementById('send-btn');
        this.statusIndicator = document.getElementById('status-indicator');
        this.newChatBtn = document.getElementById('new-chat-btn');
        this.welcomeScreen = document.getElementById('welcome-screen');
        this.attachBtn = document.getElementById('attach-btn');
        this.imageInput = document.getElementById('image-input');
        this.imagePreview = document.getElementById('image-preview');
        // 待发送的图片：[{ dataUrl, mimeType, base64 }]
        this.pendingImages = [];
        this.maxImages = 3;
        
        this.init();
    }
    
    init() {
        // 会话 ID 放在 sessionStorage：**每个标签页一个独立会话**。
        //  - 新开标签页 → 全新对话（不会跟别的对话框共用上下文）
        //  - 刷新当前标签页 → 还是本标签页的对话，并会把历史渲染回来
        // 旧实现用 localStorage，导致所有标签页共用一个 session_id：
        // 在一个对话框里聊的内容会出现在另一个对话框里。
        this.sessionId = sessionStorage.getItem('led_session_id') || this.generateSessionId();
        sessionStorage.setItem('led_session_id', this.sessionId);
        // 清掉历史遗留的 localStorage 键，避免旧版本继续复用同一个会话
        localStorage.removeItem('led_session_id');
        
        // 绑定事件
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        this.newChatBtn.addEventListener('click', () => this.startNewChat());

        // 图片：按钮上传 + 粘贴 + 拖拽
        this.bindImageInput();
        
        // 快捷问题按钮
        document.querySelectorAll('.quick-question-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const question = btn.dataset.question;
                this.messageInput.value = question;
                this.sendMessage();
            });
        });
        
        // 自动调整输入框高度
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
        });
        
        // 检查API状态
        this.checkApiStatus();

        // 刷新后恢复本会话的历史（否则界面空白、但服务端还在这个会话里，
        // 看起来就像"AI 还记得别的对话"）
        this.restoreHistory();
        
        // 聚焦输入框
        this.messageInput.focus();
    }
    
    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    // ── 图片（客户发照片描述需求）─────────────────────────────────────────
    bindImageInput() {
        // 1) 按钮上传
        if (this.attachBtn && this.imageInput) {
            this.attachBtn.addEventListener('click', (event) => {
                event.preventDefault();
                this.imageInput.click();
            });
            this.imageInput.addEventListener('change', () => {
                this.addImages(Array.from(this.imageInput.files || []));
                this.imageInput.value = '';
            });
        }

        // 2) 粘贴：挂在 document 上并走捕获阶段 ——
        //    之前只监听输入框，鼠标点过别处（输入框失焦）时粘贴事件就接不到，
        //    这也是"粘贴图片用不了"的原因之一。
        document.addEventListener('paste', (event) => this.handlePaste(event), true);

        // 3) 拖拽：把图片拖进页面任意位置都能加进来
        const dropZone = document.querySelector('.chat-container') || document.body;
        ['dragenter', 'dragover'].forEach(name => {
            dropZone.addEventListener(name, (event) => {
                if (event.dataTransfer && Array.from(event.dataTransfer.types || []).includes('Files')) {
                    event.preventDefault();
                    dropZone.classList.add('drag-over');
                }
            });
        });
        ['dragleave', 'drop'].forEach(name => {
            dropZone.addEventListener(name, (event) => {
                dropZone.classList.remove('drag-over');
                if (name !== 'drop') return;
                const files = Array.from((event.dataTransfer && event.dataTransfer.files) || []);
                if (files.length) {
                    event.preventDefault();
                    this.addImages(files);
                }
            });
        });
    }

    async handlePaste(event) {
        const data = event.clipboardData;
        if (!data) return;

        // a) 剪贴板里直接有图片文件（截图 / 复制本地图片）
        const files = [];
        if (data.files && data.files.length) {
            files.push(...Array.from(data.files).filter(file => file.type.startsWith('image/')));
        }
        if (!files.length && data.items) {
            for (const item of Array.from(data.items)) {
                if (item.kind === 'file' && item.type && item.type.startsWith('image/')) {
                    const file = item.getAsFile();
                    if (file) files.push(file);
                }
            }
        }
        if (files.length) {
            event.preventDefault();
            this.addImages(files);
            return;
        }

        // b) 从网页复制的图片常常只有 URL（没有二进制），这里兜底接住
        const html = (data.getData && data.getData('text/html')) || '';
        const text = (data.getData && (data.getData('text/uri-list') || data.getData('text/plain'))) || '';
        const url = this.extractImageUrl(html) || this.extractImageUrl(text);
        if (url) {
            event.preventDefault();
            this.addImageUrl(url);
        }
    }

    extractImageUrl(text) {
        const source = String(text || '');
        if (!source) return '';
        const imgMatch = source.match(/<img[^>]+src=["']([^"']+)["']/i);
        if (imgMatch) return imgMatch[1];
        const urlMatch = source.match(/https?:\/\/[^\s"'<>]+\.(?:png|jpe?g|gif|webp|bmp)(?:\?[^\s"'<>]*)?/i);
        return urlMatch ? urlMatch[0] : '';
    }

    addImages(files) {
        for (const file of files) {
            if (!file.type.startsWith('image/')) continue;
            if (this.pendingImages.length >= this.maxImages) {
                this.addMessage('ai', `一次最多发送 ${this.maxImages} 张图片。`);
                break;
            }
            const reader = new FileReader();
            reader.onload = () => {
                const dataUrl = String(reader.result || '');
                const base64 = dataUrl.includes(',') ? dataUrl.split(',')[1] : '';
                if (!base64) return;
                this.pendingImages.push({
                    dataUrl: dataUrl,
                    mimeType: file.type || 'image/jpeg',
                    base64: base64,
                });
                this.renderImagePreview();
                this.messageInput.focus();
            };
            reader.readAsDataURL(file);
        }
    }

    addImageUrl(url) {
        if (this.pendingImages.length >= this.maxImages) {
            this.addMessage('ai', `一次最多发送 ${this.maxImages} 张图片。`);
            return;
        }
        this.pendingImages.push({ url: url, dataUrl: url, mimeType: '', base64: '' });
        this.renderImagePreview();
        this.messageInput.focus();
    }

    renderImagePreview() {
        if (!this.imagePreview) return;
        this.imagePreview.innerHTML = '';
        this.pendingImages.forEach((image, index) => {
            const wrapper = document.createElement('div');
            wrapper.className = 'image-preview-item';
            wrapper.innerHTML = `
                <img src="${image.dataUrl}" alt="待发送图片">
                <button class="image-remove" data-index="${index}" title="移除">×</button>
            `;
            this.imagePreview.appendChild(wrapper);
        });
        this.imagePreview.querySelectorAll('.image-remove').forEach(btn => {
            btn.addEventListener('click', () => {
                this.pendingImages.splice(Number(btn.dataset.index), 1);
                this.renderImagePreview();
            });
        });
        this.imagePreview.classList.toggle('has-images', this.pendingImages.length > 0);
    }

    clearImages() {
        this.pendingImages = [];
        this.renderImagePreview();
    }
    
    async checkApiStatus() {
        try {
            const response = await fetch(`${this.apiBase}/health`);
            if (response.ok) {
                this.statusIndicator.classList.add('online');
                this.statusIndicator.classList.remove('offline');
            } else {
                throw new Error('API unhealthy');
            }
        } catch (error) {
            this.statusIndicator.classList.remove('online');
            this.statusIndicator.classList.add('offline');
        }
    }
    
    startNewChat() {
        // 清空消息区
        this.messagesArea.innerHTML = `
            <div class="welcome-screen" id="welcome-screen">
                <div class="welcome-icon">💡</div>
                <h2 class="welcome-title">LED产品选型助手</h2>
                <p class="welcome-subtitle">请用英文描述您的需求</p>
                <div class="quick-questions">
                    <button class="quick-question-btn" data-question="Indoor display for conference room, 4 meters viewing distance">
                        <span class="quick-icon">🏢</span>
                        <span class="quick-text">会议室显示屏</span>
                    </button>
                    <button class="quick-question-btn" data-question="Virtual production LED screen, need HDR support">
                        <span class="quick-icon">🎬</span>
                        <span class="quick-text">虚拟制作屏幕</span>
                    </button>
                    <button class="quick-question-btn" data-question="Outdoor rental screen, 5m viewing distance, IP65">
                        <span class="quick-icon">🌤️</span>
                        <span class="quick-text">户外租赁屏</span>
                    </button>
                </div>
            </div>
        `;
        
        // 重新绑定快捷问题按钮
        document.querySelectorAll('.quick-question-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const question = btn.dataset.question;
                this.messageInput.value = question;
                this.sendMessage();
            });
        });
        // 上面的 innerHTML 重建了欢迎界面，引用要重新取（旧节点已经脱离文档）
        this.welcomeScreen = document.getElementById('welcome-screen');
        
        // 创建新会话
        const previousSessionId = this.sessionId;
        this.sessionId = this.generateSessionId();
        sessionStorage.setItem('led_session_id', this.sessionId);
        this.clearMemory();
        // 旧会话也从服务端清掉，避免长时间运行后内存里堆积一堆没人用的会话
        if (previousSessionId && previousSessionId !== this.sessionId) {
            this.clearMemoryFor(previousSessionId);
        }
    }
    
    async clearMemory() {
        await this.clearMemoryFor(this.sessionId);
    }

    async clearMemoryFor(sessionId) {
        if (!sessionId) return;
        try {
            await fetch(`${this.apiBase}/memory/clear`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId })
            });
        } catch (error) {
            console.error('清除记忆失败:', error);
        }
    }

    async restoreHistory() {
        if (!this.sessionId) return;
        try {
            const response = await fetch(
                `${this.apiBase}/memory/${encodeURIComponent(this.sessionId)}`
            );
            if (!response.ok) return;
            const data = await response.json();
            const messages = data.messages || [];
            if (!messages.length) return;

            const welcome = document.getElementById('welcome-screen');
            if (welcome) welcome.remove();
            this.welcomeScreen = null;
            for (const msg of messages) {
                if (msg.role === 'user') {
                    this.addMessage('user', msg.content);
                } else if (msg.role === 'assistant') {
                    this.addMessage('ai', msg.content);
                }
            }
        } catch (error) {
            console.error('恢复会话历史失败:', error);
        }
    }
    
    async sendMessage() {
        const question = this.messageInput.value.trim();
        const outgoingImages = this.pendingImages.slice();
        // 纯图片消息也允许发送（客户只发照片描述需求）
        if ((!question && outgoingImages.length === 0) || this.isLoading) return;
        
        // 移除欢迎界面
        const welcome = document.getElementById('welcome-screen');
        if (welcome) welcome.remove();
        this.welcomeScreen = null;
        
        // 添加用户消息
        this.addMessage('user', question, outgoingImages.map(image => image.dataUrl));
        this.clearImages();
        
        // 清空输入框
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
        
        // 显示打字指示器
        this.showTyping();
        
        try {
            const response = await fetch(`${this.apiBase}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    question: question,
                    images: outgoingImages.map(image => (
                        image.base64
                            ? { data: image.base64, mime_type: image.mimeType }
                            : { url: image.url }
                    )),
                })
            });
            
            if (!response.ok) throw new Error('API错误');
            
            const data = await response.json();
            this.hideTyping();
            
            // 处理首次接触消息（介绍 + 视频）
            if (data.first_contact_messages && data.first_contact_messages.length > 0) {
                for (const msg of data.first_contact_messages) {
                    if (msg.role === 'assistant') {
                        if (msg.asset_type && msg.asset_url) {
                            this.addAssetMessage(msg.asset_type, msg.asset_name, msg.asset_url);
                        } else {
                            this.addMessage('ai', msg.content);
                        }
                    }
                }
            } else {
                // 普通回复
                this.addMessage('ai', data.answer);
            }
            
        } catch (error) {
            this.hideTyping();
            this.addMessage('ai', '抱歉，发生了错误。请检查API服务是否正常运行。');
            console.error('聊天错误:', error);
        }
    }
    
    addMessage(role, content, images = null) {
        const messageEl = document.createElement('div');
        messageEl.className = `message ${role}`;
        
        const avatar = role === 'user' ? '👤' : '🤖';

        let imagesHtml = '';
        if (Array.isArray(images) && images.length) {
            imagesHtml = `<div class="message-images">${
                images.map(src => `<img src="${src}" alt="客户图片" class="message-image">`).join('')
            }</div>`;
        }
        
        messageEl.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-bubble">${imagesHtml}${content ? this.formatText(content) : ''}</div>
            </div>
        `;
        
        this.messagesArea.appendChild(messageEl);
        this.scrollToBottom();
    }
    
    addAssetMessage(assetType, assetName, assetUrl) {
        const messageEl = document.createElement('div');
        messageEl.className = 'message ai';
        
        let mediaHtml = '';
        
        if (assetType === 'video') {
            mediaHtml = `
                <span class="asset-label">🎬 ${assetName}</span>
                <video controls class="asset-video">
                    <source src="${assetUrl}" type="video/mp4">
                    您的浏览器不支持视频播放。
                </video>
            `;
        } else if (assetType === 'image') {
            mediaHtml = `
                <span class="asset-label">📇 ${assetName}</span>
                <img src="${assetUrl}" alt="${assetName}" class="asset-image">
            `;
        } else if (assetType === 'pdf') {
            mediaHtml = `
                <a href="${assetUrl}" target="_blank" class="asset-link">
                    <span class="asset-icon">📄</span>
                    <span class="asset-name">${assetName}</span>
                </a>
            `;
        } else {
            mediaHtml = `
                <a href="${assetUrl}" target="_blank" class="asset-link">
                    <span class="asset-icon">📎</span>
                    <span class="asset-name">${assetName}</span>
                </a>
            `;
        }
        
        messageEl.innerHTML = `
            <div class="message-avatar">🤖</div>
            <div class="message-content">
                <div class="message-bubble asset-message">
                    ${mediaHtml}
                </div>
            </div>
        `;
        
        this.messagesArea.appendChild(messageEl);
        this.scrollToBottom();
    }
    
    showTyping() {
        this.isLoading = true;
        this.sendBtn.disabled = true;
        
        const typingEl = document.createElement('div');
        typingEl.className = 'message ai';
        typingEl.id = 'typing-indicator';
        typingEl.innerHTML = `
            <div class="message-avatar">🤖</div>
            <div class="message-content">
                <div class="message-bubble">
                    <div class="typing-indicator">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
            </div>
        `;
        
        this.messagesArea.appendChild(typingEl);
        this.scrollToBottom();
    }
    
    hideTyping() {
        this.isLoading = false;
        this.sendBtn.disabled = false;
        
        const typing = document.getElementById('typing-indicator');
        if (typing) typing.remove();
    }
    
    formatText(text) {
        // HTML转义
        let formatted = text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
        
        // 换行
        formatted = formatted.replace(/\n/g, '<br>');
        
        // 粗体
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // 代码块
        formatted = formatted.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
        formatted = formatted.replace(/`([^`]+)`/g, '<code style="background:#f1f5f9;padding:2px 6px;border-radius:4px;">$1</code>');
        
        return formatted;
    }
    
    scrollToBottom() {
        requestAnimationFrame(() => {
            this.messagesArea.scrollTop = this.messagesArea.scrollHeight;
        });
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    window.ledChatApp = new LEDChatApp();
});
