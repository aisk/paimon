# Paimon

![Paimon](https://automaton-media.com/wp-content/uploads/2020/10/20201019-140524-header.jpg)

[English](README.md) | 简体中文

Paimon 是一个终端里的 coding agent。它读写当前目录下的文件，执行命令。它也支持无头运行，还可以作为库导入，由更强的 agent 或者你自己的程序来驱动。

## 安装

```bash
uv tool install paimon   # 或者：pip install paimon
```

或者不安装直接运行：

```bash
uvx paimon
```

## 快速开始

```bash
paimon
```

首次启动会询问 provider 及其 API base 和 key，然后选择模型。列在第一位的 Kilo 无需账号：它的免费模型不用 key 就能用，但背后的模型提供方可能保留提示词和代码并用于训练，且每个 IP 每小时限 200 次请求。

之后输入要完成的任务即可。在提示中写 `@path/to/file` 可以把文件或图片提供给 agent。以 `!` 开头的一行不发给模型，而是直接在 shell 里执行。

运行时：`Shift+Tab` 切换 agent 的自主程度（**read** 只读，**auto** 放行工作目录内的编辑，其余操作交给另一次模型调用审批，**yolo** 不做任何检查，也是默认值），`Esc` 打断当前回合，`Ctrl+P` 打开命令面板，`Ctrl+T` 在新 pane 里打开另一个会话，`Ctrl+C` 退出。

## 常用命令

```bash
paimon -c                           # 恢复这个目录下最近的会话
paimon -r                           # 选择一个会话恢复
paimon --mode read                  # 以更谨慎的权限模式启动
paimon --model zai:glm-4.7          # 仅本次运行使用这个模型
paimon --web                        # 在浏览器里使用同一套 UI
paimon -p "what does cli.py do?"    # 不启动 UI，答案输出到 stdout
cat log.txt | paimon -p "summarize this"
paimon login zai --api-key-env ZAI_API_KEY
paimon install-skill                # 教 Claude Code 或 Codex 把 Paimon 当作 worker 使用
```

`-p` 不会停下来询问，所以在默认的 `yolo` 模式下它已经可以写文件、执行命令。权限模式是防止 agent 犯错的护栏，不是安全边界：需要真正的隔离时，请在容器或虚拟机里运行 Paimon。

## 文档

详细文档目前只有英文：

- [Guide](https://aisk.github.io/paimon/guide.html)：提示与图片、权限模式、多 pane 与并行 agent、会话、skills，以及由其他 agent 或 Python 程序驱动 Paimon。
- [Reference](https://aisk.github.io/paimon/reference.html)：全部命令行参数、子命令、快捷键、工具和 `config.json` 配置项，以及无头模式的输出格式。
- [Architecture](https://aisk.github.io/paimon/architecture.html)：代码的组织方式。

## 遥测

每次启动会向 Google Analytics 发送一条匿名事件，其中不包含会话、提示词、文件或凭据的任何内容；具体字段见 [Reference](https://aisk.github.io/paimon/reference.html#telemetry)。设置 `PAIMON_NO_TELEMETRY=1` 或 `DO_NOT_TRACK=1` 可以关闭。
