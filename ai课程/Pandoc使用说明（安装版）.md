# Pandoc MD转Docx 使用说明

---

## 一、软件安装与环境配置

### 1.1 Pandoc 安装

Pandoc 是通用的文档转换工具，支持 Markdown 转 Word (Docx)。

**Windows 安装：**

下载安装包：https://pandoc.org/installing.html

或使用 winget 安装：

```batch
winget install pandoc
```

**验证安装：**

```bash
pandoc --version
```

### 1.2 PlantUML 环境配置

Pandoc 转换时，PlantUML 代码块需要先渲染为图片才能嵌入 Word。

**安装 Java 运行环境：**

PlantUML 依赖 Java 环境。下载路径（三选一）：

- Eclipse Temurin (JDK 17)：https://adoptium.net/temurin/releases/?version=17
- Oracle JDK：https://www.oracle.com/java/technologies/downloads/
- OpenJDK：https://openjdk.org/

或使用 winget 安装：

```batch
winget install EclipseAdoptium.Temurin17.JDK
```

**下载 PlantUML.jar：**

PlantUML 官方下载：https://plantuml.com/download

直接下载链接：
- 最新版：https://sourceforge.net/projects/plantuml/files/plantuml.jar/download
- Maven 仓库：https://repo1.maven.org/maven2/net/sourceforge/plantuml/plantuml/

下载后保存到 `H:\Dev\Standard-Doc\tools\plantuml.jar`

**配置环境变量：**

临时配置（当前命令行窗口有效）：

```batch
set PATH=H:\Dev\Standard-Doc\tools\jdk-26\bin;%PATH%
set PUMLJAR=H:\Dev\Standard-Doc\tools\plantuml.jar
set PLANTUML_BIN=java -jar %PUMLJAR%
```

永久配置（系统环境变量）：

1. 右键「此电脑」→「属性」→「高级系统设置」
2. 点击「环境变量」
3. 在「系统变量」中新建或编辑：
   - `PATH` 添加：`H:\Dev\Standard-Doc\tools\jdk-26\bin`
   - 新建 `PUMLJAR`：`H:\Dev\Standard-Doc\tools\plantuml.jar`
   - 新建 `PLANTUML_BIN`：`java -jar %PUMLJAR%`

**验证 PlantUML：**

```bash
java -jar $env:PUMLJAR -version
```

### 1.3 LaTeX 环境（可不安装）

如需转换 PDF，需安装 MiKTeX 或 TeX Live：

```batch
winget install MiKTeX
```

---

## 二、快速转换

### 2.1 MD转Docx

```bash
pandoc 输入文档.md --filter pandoc-plantuml --reference-doc=my-style.docx -o 输出文档.docx
```
若需要用到pandoc-plantuml，需要下载pandoc-plantuml-filter插件，可运行
```bash
# 先装pandoc本体 + 对接plantuml的过滤器
pip install pandoc pandoc-plantuml-filter
```
### 2.2 Docx转MD

```bash
pandoc 输入文档.docx -o 输出文档.md
```

**说明：** Docx转MD为单向转换，Word文档中的复杂格式可能无法完全保留。

### 2.3 转换参数说明

| 参数 | 说明 |
|------|------|
| `--filter pandoc-plantuml` | 自动将 PlantUML 代码块转换为图片 |
| `--reference-doc=my-style.docx` | Word 参考文档，控制样式 |

---

## 三、预期输出

转换后的 Word 文档应包含：

1. **封面**（元数据标题页）
2. **正文**（带正确的标题层级）
3. **表格**（带"表X"序号）
4. **图片/PlantUML**（带"图X"序号）
5. **代码块**（有语法高亮）
6. **公式**（正确渲染）
