# Paimon

![Paimon](https://automaton-media.com/wp-content/uploads/2020/10/20201019-140524-header.jpg)

[English](README.md) | 简体中文

Paimon 是一个终端里的 coding agent。它读写当前目录下的文件，执行命令。它也支持无头运行，还可以作为库导入，由更强的 agent 或者你自己的程序来驱动。

## 安装

```bash
uv tool install paimon   # 或者：pip install paimon
```

## 快速开始

```bash
paimon
```

或者不安装直接运行：

```bash
uvx paimon
```

首次启动会询问 provider、模型、API base 和 key。之后输入要完成的任务即可。在提示中写 `@path/to/file` 可以把文件提供给 agent。

运行时：`Shift+Tab` 切换 agent 的自主程度（**read** 写文件或执行命令前先询问，**edit** 放行工作目录内的编辑，**yolo** 从不询问，也是默认值），`Esc` 打断当前回合，`Ctrl+P` 打开命令面板，`Ctrl+C` 退出。

`Ctrl+T` 在新 pane 里打开另一个会话，`Ctrl+W` 关闭当前 pane，`Ctrl+PageUp` 和 `Ctrl+PageDown` 在 pane 之间切换，`Ctrl+G` 跳到正在等待授权的 pane。Paimon 自己也能开 pane：让它同时做两件互不相干的事，它会在新 tab 里起第二个 agent。它也能把一条命令留在单独的 tab 里跑，比如开发服务器或者文件监视，不占着当前回合。

## 当作 subagent 使用

前沿模型擅长制定计划和验收结果，中间的执行步骤往往比较机械。让 Paimon 使用成本较低的模型执行，由 Claude Code 或 Codex 制定计划并检查结果。用一个 profile 单独保存该模型的账号：

```bash
paimon login --profile glm --model zai:glm-4.7 --api-key-env ZAI_API_KEY
paimon --profile glm -p "apply the plan in PLAN.md" --mode edit --output-format result
```

自带的 skill 会向调用方 agent 说明这套流程：

```bash
paimon install-skill                  # 安装到 Claude Code（~/.claude/skills/paimon）
paimon install-skill --target codex   # 安装到 Codex；--dest DIR 安装到任意目录
npx skills add aisk/paimon            # 通过 skills.sh 安装同一个 skill
```

## 当作库使用

agent 循环本身是可以导入的，Python 程序不必经过 CLI 就能驱动它。`Agent.open()` 新建或恢复一个会话，`agent.run()` 产出类型化的事件，每段文本、每次工具调用、每个回合结束各一个，调用方自己决定怎么渲染或过滤：

```python
import asyncio

from paimon.agent import Agent, TextDelta

async def main():
    agent = Agent.open(mode="edit")
    async for event in agent.run("summarize the tests in this directory"):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)

asyncio.run(main())
```

`Agent.open()` 还接受工作目录、用于权限确认的异步 `confirm` 回调，以及 `toolset`，可以只给模型一部分工具，或者换成你自己的工具。agent 被回收时会交还会话，想在确定的时刻交还就调 `close()`，或者把它当上下文管理器用。它写的会话文件和 CLI 一样，所以代码里跑出来的会话之后可以用 `paimon -r` 恢复。

## 会话

每次对话都会保存，很长的对话在接近上下文上限时会被原地总结。退出时 Paimon 会打印恢复该会话的命令：

```bash
paimon -r            # 从当前目录的会话中选择
paimon -r a1b2c3     # 按 id 恢复
paimon -c            # 恢复最近一个会话
paimon sessions      # 列出会话（--json 输出机器可读格式）
paimon log a1b2c3    # 查看会话做了什么，每个事件一行
```

## 其他运行方式

```bash
paimon --mode read                  # 以更谨慎的权限模式启动（默认为 yolo）
paimon --strict                     # 每条命令都先询问，包括只读命令
paimon --web                        # 在浏览器中使用同一套 UI（--port，默认 8000）
paimon -p "what does cli.py do?"    # 直接在 stdout 输出回答，不启动 UI
cat log.txt | paimon -p "summarize this"
paimon --model zai:glm-4.7          # 仅本次运行使用该模型
paimon --profile work               # 单独配置的另一个账号
```

`-p` 不会停下来询问，配合默认的 `yolo` 模式，它已经可以修改文件和执行命令。加 `--output-format result` 会输出一个包含结果的 JSON 对象，调用方程序读这个就够了。其余选项见 `paimon --help`。

## 配置

每个 profile 的模型设置保存在 `~/.config/paimon/<name>/config.json`，由首次启动或 `paimon login` 写入。会话存放在 `~/.local/share/paimon/sessions/`。安装 [delta](https://github.com/dandavison/delta) 后文件改动的展示效果更好。

read 和 edit 模式会不经询问执行一小组明确只读的命令（`ls`、`cat`、`git status` 等），`--strict` 可以关掉。**这是防止 agent 失误的护栏，不是安全边界。** 需要真正的隔离时，请在容器或虚拟机中运行 Paimon。
