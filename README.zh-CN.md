# Paimon

![Paimon](https://automaton-media.com/wp-content/uploads/2020/10/20201019-140524-header.jpg)

[English](README.md) | 简体中文

Paimon 是一个终端里的 coding agent。它读写当前目录下的文件、执行命令，在做任何改动前会先询问。它也支持无头运行，可以被更强的 agent 作为执行者调用。

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

首次启动会询问 provider、模型、API base 和 key，并保存到 `~/.config/paimon/default/config.json`。之后输入要完成的任务即可。

运行时：`Shift+Tab` 切换 agent 的自主程度（**read**：写文件或执行命令前先询问，`ls`、`git status` 这类明确只读的命令除外，会直接执行，**edit**：工作目录内的编辑直接执行，**yolo**：从不询问），`Esc` 打断当前回合，`Ctrl+P` 打开命令面板（切换 provider 或 profile、新建、分叉或恢复会话、显示模型思考、压缩上下文），`Ctrl+C` 退出。

`Ctrl+T` 在新 pane 里打开另一个会话，`Ctrl+W` 关闭当前 pane，`Ctrl+PageUp` 和 `Ctrl+PageDown` 在 pane 之间切换，`Ctrl+G` 跳到正在等待授权的 pane。tab 栏可以在命令面板里改成停靠在上方、左侧或右侧。

Paimon 自己也能开 pane：让它同时做两件互不相干的事，它会在新 tab 里起第二个 agent，工具、工作目录和权限模式都和当前会话一样。它的授权确认弹在它自己的 tab 里，用 `Ctrl+G` 过去处理。这些会话属于开它们的那个会话，不会出现在 `paimon sessions` 里，也随它一起结束。

它也能把一条命令留在单独的 tab 里跑，比如开发服务器、文件监视或者很长的构建，不占着当前回合。这类命令一律先确认，不管当前模式对只读命令怎么规定。tab 里流式显示输出，关掉 tab 或退出时命令随之停止。输出不是终端时很多程序会按块缓冲，所以 tab 里可能一阵子不出东西再一次性出来：这是用管道代替终端的代价，Paimon 不做终端模拟。

在提示中写 `@path/to/file` 可以把文件提供给 agent。

## 当作 subagent 使用

前沿模型擅长制定计划和验收结果，中间的执行步骤往往比较机械。让 Paimon 使用成本较低的模型执行，由 Claude Code 或 Codex 制定计划并检查结果，只在必要的环节为前沿模型付费。用一个 profile 单独保存该模型的账号：

```bash
paimon login --profile glm --model zai:glm-4.7 --api-key-env ZAI_API_KEY
paimon --profile glm -p "apply the plan in PLAN.md" --mode edit --output-format result
```

自带的 skill 会向调用方 agent 说明这套流程（先用 `paimon status --json` 检查、单次运行、读取唯一一行 result 对象、用其中的 `session_id` 续跑、用 `paimon log` 查看运行过程）：

```bash
paimon install-skill                  # 安装到 Claude Code（~/.claude/skills/paimon）
paimon install-skill --target codex   # 安装到 Codex；--dest DIR 安装到任意目录
npx skills add aisk/paimon            # 通过 skills.sh 安装同一个 skill
```

## 会话

每次对话都会保存。退出时 Paimon 会打印恢复该会话的命令：

```bash
paimon -r            # 从当前目录的会话中选择
paimon -r a1b2c3     # 按 id 恢复
paimon -c            # 恢复最近一个会话
paimon sessions      # 列出会话（--json 输出机器可读格式）
paimon log a1b2c3    # 查看会话做了什么，每个事件一行
```

`paimon log` 的每行输出都带一个稳定的序号；`--after SEQ`、`--turns N`、`--tail N` 缩小范围，`--json` 和 `--full` 输出原始记录。

## 其他运行方式

```bash
paimon --mode edit                  # 以更宽松的权限模式启动
paimon --strict                     # 每条命令都先询问，包括只读命令
paimon --web                        # 在浏览器中使用同一套 UI（--port，默认 8000）
paimon -p "what does cli.py do?"    # 直接在 stdout 输出回答，不启动 UI
cat log.txt | paimon -p "summarize this"
paimon --model zai:glm-4.7          # 仅本次运行使用该模型
paimon --profile work               # 单独配置的另一个账号
```

`-p` 不会停下来询问，当前模式需要确认的操作会被直接拒绝（识别为只读的命令在 read 模式下仍会执行）；如果本次运行需要修改文件，传 `--mode edit` 或 `--mode yolo`。加 `--output-format result` 输出一个包含结果的 JSON 对象（`json` 则每行输出一个事件），用 `--timeout`/`--max-tool-calls` 为无人值守的运行设置上限。

## 配置

`~/.config/paimon/<name>/config.json` 保存每个 profile 的模型设置（不传 `--profile` 时为 `default`）。两个可选配置项可以改变它的行为：自动放行只读命令，以及在接近上下文上限时原地总结长对话。

```json
{
  "safe_commands": false,
  "compaction": {
    "enabled": true,
    "context_window": 128000,
    "reserve_tokens": 16384,
    "keep_recent_tokens": 20000
  }
}
```

`safe_commands`（默认 `true`）允许 read 和 edit 模式不经询问执行一小组固定的、明确只读的命令（`ls`、`cat`、`git status` 等）；`--strict` 可在单次运行中关闭它。被识别的命令可以用 `&&`、`;` 或管道串联，也包括 `cd 目录 && …`（目录须留在工作目录内，且整条命令都以 `&&` 连接）。重定向、`$()`/反引号替换和后台 `&` 仍会询问。

**这是防止 agent 失误的护栏，不是安全边界。** 被识别的命令仍然通过 `PATH` 查找，仍可能顺着符号链接读到工作目录之外；而且哪怕是纯读取，也会把文件内容带进模型上下文，只读不等于保密安全。需要真正的隔离时，请在容器或虚拟机中运行 Paimon。

会话存放在 `~/.local/share/paimon/sessions/`（`PAIMON_DATA_HOME` 可覆盖）。安装 [delta](https://github.com/dandavison/delta) 后文件改动的展示效果更好。
