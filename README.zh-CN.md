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

运行时：`Shift+Tab` 切换 agent 的自主程度（**read**：写文件或执行命令前先询问，**edit**：工作目录内的编辑直接执行，**yolo**：从不询问），`Esc` 打断当前回合，`Ctrl+P` 打开命令面板（切换 provider 或 profile、新建或恢复会话、显示模型思考、压缩上下文），`Ctrl+C` 退出。

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
paimon --web                        # 在浏览器中使用同一套 UI（--port，默认 8000）
paimon -p "what does cli.py do?"    # 直接在 stdout 输出回答，不启动 UI
cat log.txt | paimon -p "summarize this"
paimon --model zai:glm-4.7          # 仅本次运行使用该模型
paimon --profile work               # 单独配置的另一个账号
```

`-p` 不会停下来询问，当前模式需要确认的操作会被直接拒绝；如果本次运行需要修改文件，传 `--mode edit` 或 `--mode yolo`。加 `--output-format result` 输出一个包含结果的 JSON 对象（`json` 则每行输出一个事件），用 `--timeout`/`--max-tool-calls` 为无人值守的运行设置上限。

## 配置

`~/.config/paimon/<name>/config.json` 保存每个 profile 的模型设置（不传 `--profile` 时为 `default`）。长对话可以在接近上下文上限时原地总结，添加如下配置：

```json
{
  "compaction": {
    "enabled": true,
    "context_window": 128000,
    "reserve_tokens": 16384,
    "keep_recent_tokens": 20000
  }
}
```

会话存放在 `~/.local/share/paimon/sessions/`（`PAIMON_DATA_HOME` 可覆盖）。安装 [delta](https://github.com/dandavison/delta) 后文件改动的展示效果更好。
