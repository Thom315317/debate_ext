import * as vscode from 'vscode';
import { CristalChatProvider } from './chatPanel';
import { runDebateFromChat } from './orchestrator';
import { getOutputChannel, log, showInfo, dispose as disposeLogger } from './logger';
import { setSecretStorage, getOpenAIKey, setOpenAIKey, deleteOpenAIKey, getAnthropicKey, setAnthropicKey, deleteAnthropicKey } from './cliRunner';

let chatProvider: CristalChatProvider;

export function activate(context: vscode.ExtensionContext): void {
    log('CRISTAL CODE activating...');

    setSecretStorage(context.secrets);

    chatProvider = new CristalChatProvider(context.extensionUri);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            CristalChatProvider.viewType,
            chatProvider
        )
    );

    context.subscriptions.push(
        chatProvider.onDidReceiveInput((input) => {
            handleChatInput(input);
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('cristalCode.runDebate', () => {
            promptAndRun();
        }),
        vscode.commands.registerCommand('cristalCode.runSimple', () => {
            promptAndRun('SIMPLE');
        }),
        vscode.commands.registerCommand('cristalCode.runComplex', () => {
            promptAndRun('COMPLEXE');
        }),
        vscode.commands.registerCommand('cristalCode.showLogs', () => {
            getOutputChannel().show(true);
        }),
        vscode.commands.registerCommand('cristalCode.configureCLIs', () => {
            configure();
        }),
        vscode.commands.registerCommand('cristalCode.configureOpenAIKey', () => {
            configureOpenAIKey();
        }),
        vscode.commands.registerCommand('cristalCode.configureAnthropicKey', () => {
            configureAnthropicKey();
        }),
        vscode.commands.registerCommand('cristalCode.clearChat', () => {
            chatProvider.clearMessages();
        }),
        vscode.commands.registerCommand('cristalCode.stopDebate', () => {
            showInfo('Stop requested — the current operation will be cancelled.');
        })
    );

    checkSetup();
    log('CRISTAL CODE activated.');
}

export function deactivate(): void {
    disposeLogger();
}

function handleChatInput(input: string): void {
    const modeMatch = input.match(/^\[mode:(\w+)\]\s*([\s\S]*)$/);
    let mode: 'SIMPLE' | 'MOYEN' | 'COMPLEXE' | undefined;
    let prompt: string;

    if (modeMatch) {
        const modeStr = modeMatch[1].toUpperCase();
        prompt = modeMatch[2].trim();
        if (modeStr === 'SIMPLE' || modeStr === 'MOYEN' || modeStr === 'COMPLEXE') {
            mode = modeStr;
        }
    } else {
        prompt = input.trim();
    }

    if (!prompt) { return; }

    runDebateFromChat(chatProvider, prompt, mode);
}

async function promptAndRun(forceMode?: 'SIMPLE' | 'MOYEN' | 'COMPLEXE'): Promise<void> {
    const prompt = await vscode.window.showInputBox({
        prompt: 'Describe the task for CRISTAL CODE',
        placeHolder: 'e.g., Refactor the authentication module to use JWT tokens',
        ignoreFocusOut: true,
    });
    if (!prompt?.trim()) { return; }

    runDebateFromChat(chatProvider, prompt.trim(), forceMode);
}

async function checkSetup(): Promise<void> {
    const config = vscode.workspace.getConfiguration('cristalCode');

    const hasAnthropic = !!(await getAnthropicKey());
    const hasOpenAI = !!(await getOpenAIKey());

    log(`Setup check: anthropic=${hasAnthropic ? 'OK' : 'NOT SET'}, openai=${hasOpenAI ? 'OK' : 'NOT SET'}`);

    if (!hasAnthropic) {
        chatProvider.addMessage('error', 'Anthropic API key not set. Use the gear icon to configure it.');
    }

    if (!hasOpenAI) {
        chatProvider.addMessage('error', 'OpenAI API key not set. Use the gear icon to configure it.');
    }

    if (hasAnthropic && hasOpenAI) {
        const claudeModel = config.get<string>('claudeModel', 'claude-opus-4-6');
        const openaiModel = config.get<string>('openaiModel', 'gpt-5.1');
        chatProvider.addMessage('system', `Ready: Claude ${claudeModel} + OpenAI ${openaiModel}`);
    }
}

async function configureOpenAIKey(): Promise<void> {
    const existing = await getOpenAIKey();

    const items = [
        {
            label: '$(globe) Get API key from browser',
            description: 'Opens platform.openai.com to generate/copy an API key',
            id: 'browser',
        },
        {
            label: '$(edit) Paste API key directly',
            description: 'Enter an API key you already have',
            id: 'paste',
        },
        ...(existing ? [{
            label: '$(trash) Remove API key',
            description: 'Delete the stored API key',
            id: 'remove',
        }] : []),
        { label: '$(close) Cancel', description: '', id: 'cancel' },
    ];

    const action = await vscode.window.showQuickPick(items, {
        title: `CRISTAL CODE — OpenAI API Key ${existing ? '(configured)' : '(not set)'}`,
        placeHolder: existing ? 'Your key is already configured. Update or remove it?' : 'Choose how to connect your OpenAI account',
    });

    if (!action || action.id === 'cancel') { return; }

    if (action.id === 'remove') {
        await deleteOpenAIKey();
        showInfo('OpenAI API key removed.');
        chatProvider.addMessage('system', 'OpenAI API key removed.');
        return;
    }

    if (action.id === 'browser') {
        await vscode.env.openExternal(vscode.Uri.parse('https://platform.openai.com/api-keys'));
        showInfo('Browser opened — create or copy an API key, then paste it here.');
    }

    const key = await vscode.window.showInputBox({
        prompt: action.id === 'browser'
            ? 'Paste your API key from the browser (it starts with sk-...)'
            : 'Enter your OpenAI API key',
        placeHolder: 'sk-...',
        password: true,
        ignoreFocusOut: true,
    });

    if (key?.trim()) {
        await setOpenAIKey(key.trim());
        showInfo('OpenAI API key saved securely.');
        chatProvider.addMessage('system', 'OpenAI API key configured. Ready to use!');
    }
}

async function configureAnthropicKey(): Promise<void> {
    const existing = await getAnthropicKey();

    const items = [
        {
            label: '$(globe) Get API key from browser',
            description: 'Opens console.anthropic.com to generate/copy an API key',
            id: 'browser',
        },
        {
            label: '$(edit) Paste API key directly',
            description: 'Enter an API key you already have (sk-ant-api03-...)',
            id: 'paste',
        },
        ...(existing ? [{
            label: '$(trash) Remove API key',
            description: 'Delete the stored API key',
            id: 'remove',
        }] : []),
        { label: '$(close) Cancel', description: '', id: 'cancel' },
    ];

    const action = await vscode.window.showQuickPick(items, {
        title: `CRISTAL CODE — Anthropic API Key ${existing ? '(configured)' : '(not set)'}`,
        placeHolder: existing ? 'API key active. Update or remove?' : 'Add your Anthropic API key',
    });

    if (!action || action.id === 'cancel') { return; }

    if (action.id === 'remove') {
        await deleteAnthropicKey();
        showInfo('Anthropic API key removed.');
        chatProvider.addMessage('system', 'Anthropic API key removed.');
        return;
    }

    if (action.id === 'browser') {
        await vscode.env.openExternal(vscode.Uri.parse('https://console.anthropic.com/settings/keys'));
        showInfo('Browser opened — create or copy an API key, then paste it here.');
    }

    const key = await vscode.window.showInputBox({
        prompt: action.id === 'browser'
            ? 'Paste your API key from the browser (it starts with sk-ant-api03-...)'
            : 'Enter your Anthropic API key',
        placeHolder: 'sk-ant-api03-...',
        password: true,
        ignoreFocusOut: true,
    });

    if (key?.trim()) {
        await setAnthropicKey(key.trim());
        showInfo('Anthropic API key saved securely.');
        chatProvider.addMessage('system', 'Anthropic API key configured.');
    }
}

async function configure(): Promise<void> {
    const config = vscode.workspace.getConfiguration('cristalCode');

    const choice = await vscode.window.showQuickPick(
        [
            { label: '$(key) Anthropic API Key', id: 'anthropic-key' },
            { label: '$(key) OpenAI API Key', id: 'openai-key' },
            { label: '$(hubot) Configure Claude Model', id: 'claude-model' },
            { label: '$(server) Configure OpenAI Model', id: 'openai-model' },
            { label: '$(beaker) Configure Test Command', id: 'test-cmd' },
        ],
        { title: 'CRISTAL CODE — Configuration', placeHolder: 'What do you want to configure?' }
    );

    if (!choice) { return; }

    switch (choice.id) {
        case 'anthropic-key':
            await configureAnthropicKey();
            break;
        case 'openai-key':
            await configureOpenAIKey();
            break;
        case 'claude-model': {
            const model = await vscode.window.showQuickPick(
                ['claude-opus-4-6', 'claude-sonnet-4-5-20250929', 'claude-opus-4-5-20251101', 'claude-sonnet-4-20250514', 'claude-haiku-4-5-20251001'],
                { title: 'Select Claude model', placeHolder: config.get<string>('claudeModel', 'claude-opus-4-6') }
            );
            if (model) {
                await config.update('claudeModel', model, vscode.ConfigurationTarget.Global);
                showInfo(`Claude model set to ${model}.`);
            }
            break;
        }
        case 'openai-model': {
            const model = await vscode.window.showQuickPick(
                ['gpt-5.1', 'gpt-5.1-chat-latest', 'gpt-5.1', 'gpt-5-mini', 'gpt-4.1', 'gpt-4.1-mini', 'o3', 'o4-mini'],
                { title: 'Select OpenAI model', placeHolder: config.get<string>('openaiModel', 'gpt-5.1') }
            );
            if (model) {
                await config.update('openaiModel', model, vscode.ConfigurationTarget.Global);
                showInfo(`OpenAI model set to ${model}.`);
            }
            break;
        }
        case 'test-cmd': {
            const testCommand = await vscode.window.showInputBox({
                prompt: 'Test command (optional, e.g., pytest -q, npm test)',
                value: config.get<string>('testCommand', ''),
                ignoreFocusOut: true,
            });
            if (testCommand !== undefined) {
                await config.update('testCommand', testCommand, vscode.ConfigurationTarget.Global);
                showInfo('Test command updated.');
            }
            break;
        }
    }

    checkSetup();
}
