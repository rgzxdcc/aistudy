# Docker 最小集：把 app.py 装进集装箱（保姆级）

> 目标：Docker 零基础 → 你的 LLM 服务在容器里跑起来，curl 能通。
> 预计耗时：2–3 小时。做完后 9.7 的 Docker 部分即达标，SQL 半天另算。

## 0. 心智模型（5 分钟，先读懂再动手）

- **镜像（Image）**＝ 安装光盘/模具：一个只读的打包模板，里面有 Python + 依赖 + 你的代码
- **容器（Container）**＝ 用光盘装出来的正在运行的机器：镜像是类，容器是实例
- **Dockerfile**＝ 配方：一行行描述"怎么做出这个镜像"
- 为什么需要它：你的 Mac 上"我能跑"靠的是 aistudy/.venv 里的各种环境；换一台机器（同事的电脑、云服务器）就没有这个 venv——容器把"运行所需的全部"打成一个包，做到**任何装了 Docker 的机器上行为一致**

## 1. 安装（10 分钟）

1. 打开 https://www.docker.com/products/docker-desktop/ → 下载 **Mac with Apple Silicon** 版（你是 M2）
2. 双击 `Docker.dmg` → 把鲸鱼图标拖进 Applications → 启动 Docker Desktop（同意服务条款，密码授权）
3. 菜单栏出现鲸鱼图标 = 后台服务已运行
4. 新开终端验证：
   ```bash
   docker --version        # 能打印版本号
   docker run hello-world  # 第一次会拉一个小镜像，看到 "Hello from Docker!" 即成功
   ```
5. **如果拉取镜像很慢/超时**（国内网络常见）：点菜单栏鲸鱼 → Settings → Docker Engine，在 JSON 里加一段后 Apply & Restart：
   ```json
   "registry-mirrors": ["https://docker.m.daocloud.io", "https://dockerproxy.net"]
   ```
   镜像源失效就搜"docker 国内镜像加速"换新的。

## 2. 读什么（20 分钟，不用多读）

- **本文就是主教程**，照着做完为止
- 做完后如果想要官方对照：https://fastapi.tiangolo.com/zh/deployment/docker/ 只读「Dockerfile」开始的两节（你做的时候命令和它是同一个套路，读起来会非常顺）
- 遇到具体命令不懂：`docker <命令> --help` 自带说明书，比搜索快

## 3. 写什么（30 分钟，三个新文件）

都在 `llm_app/01_api_basics/` 目录下操作。

### 3.1 `requirements.txt`——依赖清单（手写 4 行，不要用 freeze）

```text
openai
fastapi
uvicorn[standard]
python-dotenv
```

> **为什么不 `uv pip freeze` 一把梭？** 你的 venv 里有 torch、jupyter 等几十个包（那是给手写 GPT 和学习用的），全装进镜像会把体积撑到 2GB+。容器里只装 app.py 真正 import 的 4 个包——**最小依赖**是容器化的基本原则，镜像瘦、构建快、安全面小。

### 3.2 `Dockerfile`——配方（逐行都有注释，照抄后逐行理解）

```dockerfile
# 基础镜像：一个只装了 Python 3.12 的干净 Linux。slim = 精简版（约50MB）
FROM python:3.12-slim

# 容器内的工作目录（没有会自动创建）。之后所有相对路径都以 /code 为准
WORKDIR /code

# 先只复制依赖清单进去 —— 这是刻意的：只要 requirements.txt 没改，
# 重新 build 时这层直接用缓存，跳过最慢的装依赖步骤
COPY requirements.txt

# 安装依赖。--no-cache-dir 不存 pip 缓存，镜像更瘦
RUN pip install --no-cache-dir -r requirements.txt

# 再复制代码（app.py 变了只重烧这层，几秒钟）
COPY app.py

# 声明本服务监听 8000 端口（文档性声明，真正的映射靠运行时的 -p）
EXPOSE 8000

# 容器启动时执行的命令：用 uvicorn 跑 app.py 里的 app 对象
# 0.0.0.0 = 监听容器内所有网卡（必须，否则宿主机映射不进来）；容器内不需要 --reload
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3.3 `.dockerignore`——排除清单（防止垃圾和不该进镜像的东西）

```text
.venv
.env
__pycache__
.git
*.md
```

> `.env`（你的 API key）**绝不进镜像**——镜像可能被导出、分享，密钥只能运行时注入。`.venv` 不进是因为里面的包是 Mac 版的，装进 Linux 容器也不能用。

## 4. 构建、运行、验证（20 分钟）

在 `llm_app/01_api_basics/` 目录下：

```bash
# ① 构建：-t 给镜像起名；末尾的 . 表示"用当前目录的 Dockerfile 和文件"
docker build -t llm-chat .

# ② 查看做好的镜像
docker images    # 能看到 llm-chat，约 200MB 左右

# ③ 运行：把宿主机 .env 注入容器（key 不写进镜像），端口映射 宿主机8000→容器8000
docker run --rm --env-file /Users/xuwen/Documents/dcc_study/aistudy/.env -p 8000:8000 llm-chat
```

`docker run` 参数解释：
- `--rm`：容器停止后自动删除，不留垃圾
- `--env-file <路径>`：把 .env 里的 KEY=VALUE 作为环境变量注入容器（app.py 的 `os.getenv` 就是读它们）
- `-p 8000:8000`：把宿主机的 8000 端口映射到容器的 8000（左边是宿主机）
- **验证它活着**：另开一个终端 `docker ps` 能看到运行中的容器

```bash
# ④ 验证（新终端）：
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"用一句话解释什么是Docker"}]}'

curl -N -X POST http://127.0.0.1:8000/chat_stream \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"从1数到10"}]}'
```

`-N` 关掉 curl 缓冲，能亲眼看到流式逐字输出。

## 5. 一个必踩的坑（提前告诉你）：`/chat/local` 在容器里不通

你的 `client_local` 指向 `http://localhost:11434/v1`。但**容器里的 localhost 是容器自己**，不是你的 Mac——所以 Ollama 联不上。三种处理（选一即可）：

1. **最省事**：容器里不用管 /chat/local，知道原因就行（面试还能当细节讲）
2. **正确修法**：把 `client_local` 的 base_url 改成可配置——
   ```python
   client_local = OpenAI(api_key="ollama",
       base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
   ```
   然后 run 时加参数：`docker run ... -e OLLAMA_BASE_URL=http://host.docker.internal:11434/v1 llm-chat`
   （`host.docker.internal` 是 Docker Desktop 提供的特殊域名，从容器指向宿主机 Mac）
3. 动手意愿强的：改完重跑 `docker build` + `docker run`，curl 验证 /chat/local 也通——这才算全通

## 6. 常用命令速查（贴墙上）

| 命令 | 干什么 |
|---|---|
| `docker build -t llm-chat .` | 按 Dockerfile 构建镜像 |
| `docker images` | 列出本地镜像 |
| `docker run --rm -p 8000:8000 llm-chat` | 用镜像启动容器 |
| `docker ps` | 看运行中的容器（`-a` 含已停止的） |
| `docker stop <容器ID前3位>` | 停止容器（--rm 模式下自动删除） |
| `docker logs -f <容器ID前3位>` | 追看容器日志（报错先看这里） |
| `docker exec -it <容器ID> bash` | 钻进容器里看（exit 退出） |
| `docker rmi llm-chat` | 删除镜像 |

## 7. 完成标志（自测清单）

- [ ] `docker run hello-world` 成功
- [ ] 手写 requirements.txt / Dockerfile / .dockerignore 三个文件（合上本文能说清每行干嘛）
- [ ] `docker build` 成功，`docker images` 能看到 llm-chat
- [ ] 容器运行中，curl `/chat` 拿到 LLM 回答，`/chat_stream` 看到逐字输出
- [ ] 能口头回答：镜像和容器的区别？为什么先 COPY requirements 再 COPY app.py？.env 为什么不进镜像、怎么进去的？

## 8. 面试话术（做完就有）

> "我把 FastAPI 服务做了容器化：最小依赖的手写 requirements 加 slim 基础镜像控制体积；Dockerfile 里把依赖安装和代码复制分层，利用层缓存让改代码后的重建只要几秒；密钥用 `--env-file` 运行时注入，不进镜像；Mac 上容器访问宿主机的 Ollama 用 host.docker.internal 解决 localhost 隔离。"

四句话，每句都是你亲手做过的——这就是面试里"真做过"和"背概念"的区别。
