import * as vscode from 'vscode';
import * as path from 'path';
import { log } from './logger';

export type MessageRole = 'user' | 'claude' | 'openai' | 'system' | 'error' | 'diff' | 'status';

export interface ChatMessage {
    role: MessageRole;
    text: string;
    timestamp: string;
    meta?: string;
}

/**
 * Webview sidebar provider — renders the chat UI for CRISTAL CODE.
 */
export class CristalChatProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'cristalCode.chatView';

    private _view?: vscode.WebviewView;
    private _messages: ChatMessage[] = [];
    private _onDidReceiveInput = new vscode.EventEmitter<string>();
    public readonly onDidReceiveInput = this._onDidReceiveInput.event;

    constructor(private readonly _extensionUri: vscode.Uri) {}

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        _context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken
    ): void {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [vscode.Uri.joinPath(this._extensionUri, 'media')],
        };

        webviewView.webview.html = this._getHtml(webviewView.webview);

        webviewView.webview.onDidReceiveMessage((data) => {
            switch (data.type) {
                case 'userInput':
                    log(`User input received: ${data.text.slice(0, 100)}...`);
                    this._onDidReceiveInput.fire(data.text);
                    break;
                case 'ready':
                    // Re-send existing messages on webview reload
                    this._syncMessages();
                    break;
            }
        });
    }

    /** Add a message to the chat and render it */
    public addMessage(role: MessageRole, text: string, meta?: string): void {
        const msg: ChatMessage = {
            role,
            text,
            timestamp: new Date().toLocaleTimeString(),
            meta,
        };
        this._messages.push(msg);
        this._postMessage({ type: 'addMessage', message: msg });
    }

    /** Update the status bar at the top of the chat */
    public setStatus(text: string, loading: boolean = false): void {
        this._postMessage({ type: 'setStatus', text, loading });
    }

    /** Enable or disable the input area */
    public setInputEnabled(enabled: boolean): void {
        this._postMessage({ type: 'setInputEnabled', enabled });
    }

    /** Clear all messages */
    public clearMessages(): void {
        this._messages = [];
        this._postMessage({ type: 'clearMessages' });
    }

    /** Scroll to the bottom of the chat */
    public scrollToBottom(): void {
        this._postMessage({ type: 'scrollToBottom' });
    }

    private _syncMessages(): void {
        for (const msg of this._messages) {
            this._postMessage({ type: 'addMessage', message: msg });
        }
    }

    private _postMessage(data: unknown): void {
        this._view?.webview.postMessage(data);
    }

    private _getHtml(webview: vscode.Webview): string {
        const styleUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, 'media', 'style.css')
        );
        const scriptUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, 'media', 'main.js')
        );
        const nonce = getNonce();

        return /*html*/ `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
    <link href="${styleUri}" rel="stylesheet">
    <title>CRISTAL CODE</title>
</head>
<body>
    <div id="app">
        <div id="status-bar">
            <span id="status-icon">◆</span>
            <span id="status-text">Ready</span>
            <div id="status-loader" class="loader hidden"></div>
        </div>

        <div id="messages"></div>

        <div id="input-area">
            <div id="mode-selector">
                <button class="mode-btn active" data-mode="auto">Auto</button>
                <button class="mode-btn" data-mode="simple">Simple</button>
                <button class="mode-btn" data-mode="moyen">Moyen</button>
                <button class="mode-btn" data-mode="complexe">Complexe</button>
            </div>
            <div id="input-row">
                <textarea id="user-input" placeholder="Describe your task..." rows="3"></textarea>
                <button id="send-btn" title="Send">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                        <path d="M1 1.5l14 6.5-14 6.5V9l10-1-10-1V1.5z"/>
                    </svg>
                </button>
            </div>
        </div>
    </div>
    <script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
    }
}

function getNonce(): string {
    let text = '';
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    for (let i = 0; i < 32; i++) {
        text += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return text;
}
