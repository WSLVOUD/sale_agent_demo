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
        if (!question || this.isLoading) return;
        
        // 移除欢迎界面
        const welcome = document.getElementById('welcome-screen');
        if (welcome) welcome.remove();
        this.welcomeScreen = null;
        
        // 添加用户消息
        this.addMessage('user', question);
        
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
                    question: question
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
    
    addMessage(role, content) {
        const messageEl = document.createElement('div');
        messageEl.className = `message ${role}`;
        
        const avatar = role === 'user' ? '👤' : '🤖';
        
        messageEl.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-bubble">${this.formatText(content)}</div>
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
