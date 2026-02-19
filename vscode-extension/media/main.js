// DEBATE EXT — Chat panel webview script
(function () {
    // @ts-ignore
    const vscode = acquireVsCodeApi();

    const messagesEl = document.getElementById('messages');
    const inputEl = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const statusText = document.getElementById('status-text');
    const statusIcon = document.getElementById('status-icon');
    const statusLoader = document.getElementById('status-loader');
    const modeBtns = document.querySelectorAll('.mode-btn');

    let selectedMode = 'auto';
    let inputEnabled = true;

    // --- Welcome message ---
    function showWelcome() {
        messagesEl.innerHTML = `
            <div class="welcome">
                <h2>◆ DEBATE EXT</h2>
                <p>AI debate orchestrator</p>
                <p>Claude Code generates — Codex/GPT reviews — iterate until consensus.</p>
                <div class="hint">Select a mode and describe your task below.</div>
            </div>
        `;
    }
    showWelcome();

    // --- Mode selector ---
    modeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            modeBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedMode = btn.dataset.mode;
        });
    });

    // --- Send message ---
    function send() {
        const text = inputEl.value.trim();
        if (!text || !inputEnabled) return;

        // Prefix with mode
        const payload = `[mode:${selectedMode}] ${text}`;
        vscode.postMessage({ type: 'userInput', text: payload });

        inputEl.value = '';
        inputEl.style.height = 'auto';
    }

    sendBtn.addEventListener('click', send);

    inputEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            send();
        }
    });

    // Auto-resize textarea
    inputEl.addEventListener('input', () => {
        inputEl.style.height = 'auto';
        inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
    });

    // --- Receive messages from extension ---
    window.addEventListener('message', (event) => {
        const data = event.data;

        switch (data.type) {
            case 'addMessage':
                addMessageToUI(data.message);
                break;
            case 'setStatus':
                setStatus(data.text, data.loading);
                break;
            case 'setInputEnabled':
                setInputState(data.enabled);
                break;
            case 'clearMessages':
                showWelcome();
                break;
            case 'scrollToBottom':
                scrollToBottom();
                break;
        }
    });

    function addMessageToUI(msg) {
        // Remove welcome if present
        const welcome = messagesEl.querySelector('.welcome');
        if (welcome) welcome.remove();

        const div = document.createElement('div');
        div.className = `message ${msg.role}`;

        if (msg.role === 'status') {
            div.textContent = msg.text;
        } else {
            const roleLabels = {
                user: 'You',
                claude: 'Claude',
                openai: 'OpenAI',
                system: 'System',
                error: 'Error',
                diff: 'Diff',
            };

            let bodyHtml = escapeHtml(msg.text);
            // Basic code block formatting
            bodyHtml = bodyHtml.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
            bodyHtml = bodyHtml.replace(/`([^`]+)`/g, '<code>$1</code>');

            div.innerHTML = `
                <div class="message-header">
                    <span class="badge">${roleLabels[msg.role] || msg.role}</span>
                    <span class="time">${msg.timestamp}</span>
                </div>
                <div class="message-body">${bodyHtml}</div>
                ${msg.meta ? `<div class="message-meta">${escapeHtml(msg.meta)}</div>` : ''}
            `;
        }

        messagesEl.appendChild(div);
        scrollToBottom();
    }

    function setStatus(text, loading) {
        statusText.textContent = text;
        if (loading) {
            statusLoader.classList.remove('hidden');
            statusIcon.style.display = 'none';
        } else {
            statusLoader.classList.add('hidden');
            statusIcon.style.display = '';
        }
    }

    function setInputState(enabled) {
        inputEnabled = enabled;
        inputEl.disabled = !enabled;
        sendBtn.disabled = !enabled;
    }

    function scrollToBottom() {
        requestAnimationFrame(() => {
            messagesEl.scrollTop = messagesEl.scrollHeight;
        });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Notify extension we're ready
    vscode.postMessage({ type: 'ready' });
})();
