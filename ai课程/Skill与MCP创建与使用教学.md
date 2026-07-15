# Skill 与 MCP 创建与使用教学

## 概述

在 Trae IDE 中，**Skill（技能）** 和 **MCP（Model Context Protocol，模型上下文协议）** 是扩展 AI 智能体能力的两大核心机制。

| 概念 | 作用 | 形态 |
|------|------|------|
| **Skill** | 定义智能体的行为指令、工作流程和约束 | Markdown 文件（`SKILL.md`） |
| **MCP** | 为智能体提供可调用的外部工具（API、文件操作等） | JSON 描述符 + 服务端实现 |

---

## 一、Skill 详解

### 1.1 什么是 Skill

Skill 是一份结构化的 Markdown 指令文件，告诉 AI 智能体：

- 它擅长做什么（`description`）
- 执行步骤和流程
- 输入/输出规范
- 约束和注意事项

当用户触发某个 Skill 时，智能体加载该文件的内容作为系统提示的一部分，从而获得领域专业知识和工作流程指导。

### 1.2 Skill 文件结构

Skill 文件使用 **YAML front matter + Markdown** 格式：

```markdown
---
name: "my-skill"
description: 简短描述该技能的用途
---

## 执行步骤

1. 第一步：...
2. 第二步：...

## 注意事项

- 约束条件 1
- 约束条件 2
```

**front matter 字段说明：**

| 字段 | 必填 | 说明 |
|------|------|------|
| `name` | 是 | 技能唯一标识符，全小写字母 + 连字符 |
| `description` | 是 | 技能描述，用于让智能体判断何时调用该技能 |
| `disable-model-invocation` | 否 | 设为 `true` 禁用模型主动调用 |
| `user-invocable` | 否 | 设为 `false` 禁止用户手动触发 |

### 1.3 手动创建 Skill

#### 步骤一：找到 Skill 目录

Skill 目录位于 Trae IDE 配置目录下：

```
%USERPROFILE%\.trae-cn\builtin_skills\
```

#### 步骤二：创建 Skill 文件夹

```
%USERPROFILE%\.trae-cn\builtin_skills\my-skill\
  └── SKILL.md
```

#### 步骤三：编写 SKILL.md 文件

```markdown
---
name: "my-skill"
description: 用于执行数据分析任务的技能
---

## 执行步骤

**Step 1: 理解需求**
- 明确用户需要分析的数据范围和目标

**Step 2: 数据加载**
- 使用 Pandas 读取 CSV/Excel 文件
- 检查数据完整性

**Step 3: 分析与可视化**
- 生成描述性统计
- 使用 matplotlib 绘制图表

**Step 4: 输出报告**
- 汇总分析结果
- 给出结论与建议
```

#### 步骤四：注册 Skill

新建或编辑 `%USERPROFILE%\.trae-cn\skill-config.json` 文件：

```json
{
  "disabledSkills": [],
  "builtinSkillStatus": {},
  "managedSkills": {
    "my-skill": {
      "type": "file",
      "path": "%USERPROFILE%\\.trae-cn\\builtin_skills\\my-skill\\SKILL.md",
      "version": 1
    }
  },
  "deletedSkills": []
}
```

> **提示**：通常 IDE 会自动扫描 `builtin_skills` 目录，配置文件中注册后即可生效。

#### 步骤五：重启 IDE（或重新加载）

重启 Trae IDE 使新 Skill 生效。

### 1.4 让智能体创建 Skill

使用 Trae IDE 内置的 **skill-creator** 技能（在 `Skill` 工具列表中可以找到）。流程如下：

1. **触发条件**：在对话中描述你想创建的 Skill，例如：
   > "请帮我创建一个用于代码审查的 Skill"

2. **智能体自动执行**：
   - 调用 `skill-creator` 技能
   - 询问你 Skill 的名称、描述和具体需求
   - 生成 `SKILL.md` 文件并放置在正确路径
   - 更新 `skill-config.json` 注册

3. **使用方式**：
   - 创建完成后，在对话中提及该 Skill 的名称，智能体将自动加载并使用它
   - 例如：`"使用 my-skill 来分析这份数据"`

### 1.5 调用 Skill

智能体通过以下方式决定是否调用一个 Skill：

1. 分析用户输入是否匹配某个 Skill 的 `description`
2. 在 `<available_skills>` 列表中寻找匹配项
3. 调用 `Skill` 工具加载对应 Skill 文件
4. 加载后，Skill 中的指令会注入到系统提示中

---

## 二、MCP 详解

### 2.1 什么是 MCP

MCP（Model Context Protocol）是一种协议，允许 AI 智能体通过标准化的方式调用外部工具。每个 MCP 工具由两部分组成：

- **工具描述符（Tool Descriptor）**：一个 JSON 文件，描述工具的接口（名称、参数、返回值）
- **服务端实现（Server）**：实际执行工具逻辑的后端服务

### 2.2 MCP 的目录结构

MCP 配置文件位于 Trae IDE 配置目录下：

```
%USERPROFILE%\.trae-cn\mcps\
  └── <workspace-id>\
      └── solo_agent\
          └── <server-name>\
              ├── SERVER_METADATA.json    # 服务器元数据
              └── tools\
                  ├── tool1.json           # 工具描述符
                  ├── tool2.json
                  └── ...
```

**SERVER_METADATA.json 示例：**

```json
{
  "server_name": "my-mcp-server",
  "description": "我的自定义 MCP 服务器"
}
```

**工具描述符示例（`search_docs.json`）：**

```json
{
  "name": "search_docs",
  "description": "搜索文档内容",
  "arguments": {
    "type": "object",
    "properties": {
      "query": {
        "title": "Query",
        "type": "string"
      }
    },
    "required": ["query"],
    "title": "search_docsArguments"
  }
}
```

### 2.3 手动创建 MCP

#### 步骤一：找到工作区 MCP 目录

```
%USERPROFILE%\.trae-cn\mcps\<workspace-id>\solo_agent\
```

其中 `<workspace-id>` 是当前工作区的唯一标识。

#### 步骤二：创建 MCP 服务器目录

```
%USERPROFILE%\.trae-cn\mcps\<workspace-id>\solo_agent\my-mcp-server\
  ├── SERVER_METADATA.json
  └── tools\
      ├── tool1.json
      └── tool2.json
```

#### 步骤三：创建 SERVER_METADATA.json

```json
{
  "server_name": "my-mcp-server",
  "description": "提供数据查询和处理工具"
}
```

#### 步骤四：创建工具描述符

创建 `tools/tool1.json`：

```json
{
  "name": "query_data",
  "description": "根据条件查询数据",
  "arguments": {
    "type": "object",
    "properties": {
      "condition": {
        "title": "Condition",
        "type": "string"
      },
      "limit": {
        "title": "Limit",
        "type": "integer",
        "default": 10
      }
    },
    "required": ["condition"],
    "title": "query_dataArguments"
  }
}
```

#### 步骤五：实现服务端逻辑

根据协议实现对应的服务端，监听工具调用请求。服务端需要：

- 监听指定的传输通道（如 HTTP、stdio 等）
- 解析收到的工具调用请求
- 执行对应的业务逻辑
- 返回结果

#### 步骤六：注册 MCP 服务器

在 IDE 的 MCP 配置中添加服务器信息（通常在 IDE 设置中完成），使智能体能够发现并使用这些工具。

### 2.4 让智能体创建 MCP 工具

虽然 MCP 的创建通常需要手动设计工具接口和实现服务端逻辑，但你可以借助智能体来辅助：

1. **描述需求**：
   > "请帮我创建一个 MCP 工具，用于查询数据库中的用户信息"

2. **智能体会**：
   - 询问接口细节（参数、返回值等）
   - 生成工具描述符 JSON 文件
   - 提供服务端代码示例

3. **你需要做**：
   - 确认生成的描述符是否符合需求
   - 实现或部署服务端逻辑
   - 在 IDE 中注册 MCP 服务器

### 2.5 调用 MCP 工具

智能体通过 `run_mcp` 工具调用 MCP 工具：

```
执行流程：
1. 智能体在 <mcp_file_system_servers> 中发现可用的 MCP 服务器
2. 读取工具描述符 JSON 了解接口参数
3. 使用 run_mcp(server_name, tool_name, args) 调用工具
4. 处理返回结果
```

---

## 三、Skill vs MCP：对比与选择

| 维度 | Skill | MCP |
|------|-------|-----|
| **本质** | 指令/提示词模板 | 可执行的外部工具 |
| **修改方式** | 编辑 Markdown 即可 | 需要 JSON + 服务端实现 |
| **复杂程度** | 低 | 高 |
| **适用场景** | 流程标准化、知识注入 | 系统集成、数据操作 |
| **由智能体创建** | 完全支持（skill-creator） | 辅助生成接口代码 |

**选择建议：**

- 如果只是**告诉智能体怎么做**（流程、规范、知识），→ 用 **Skill**
- 如果需要**让智能体实际做**（读写文件、调用 API、操作数据库），→ 用 **MCP**

---

## 四、常见问题

### Q1: 创建 Skill 后智能体不识别？

- 检查 `skill-config.json` 是否正确注册
- 检查 `SKILL.md` 的 front matter 格式（`name` 和 `description` 必须正确填写）
- 重启 IDE

### Q2: MCP 工具返回错误？

- 检查工具描述符的 JSON 格式是否正确
- 确认服务端是否正常运行
- 检查参数是否满足 `required` 字段要求

### Q3: 如何在项目内共享 Skill 或 MCP？

Skill 和 MCP 配置目前属于 IDE 级别的配置，存放在用户目录下。可以在团队中共享 `.trae-cn` 目录下的相关配置，或将 Skill 文件纳入版本管理并在团队内分发。

### Q4: 修改后需要重启 IDE 吗？

- Skill：需要重启或重新加载
- MCP 工具描述符：需要重启
- MCP 服务端：如果支持热更新则不需要，否则需要重启
