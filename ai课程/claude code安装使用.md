# 1 更新PowerShell（可选，先不做）

## 1.1 管理员打开Windows PowerShell，输入：$PSVersionTable

## 1.2 右键开始菜单->Windows PowerShell，执行：winget install --id Microsoft.PowerShell

# 2 安装Node.js

## 2.1 官网下载LTS版本：https://nodejs.org/，全程默认勾选

## 2.2 管理员终端验证：node --version \ npm --version

# 3 安装Git for Windows（已安装、跳过）

# 4 安装claude code

## 4.1 不要使用官网命令下载！

## 4.2 管理员终端输入：npm install -g @anthropic-ai/claude-code --registry=https://registry.npmmirror.com

## 4.3 验证安装，执行claude --version，一般显示最新版

# 5 安装cc switch（当前已安装新版，需要配置路由）

## 5.1 打开cc switch，创建API

## 5.2 打开设置-路由-本地路由，全部勾选。若出现报错"切换路由失败，claude配置文件不存在"，执行5.3

## 5.3 执行以下代码段：

```powershell
# 1. 定义用户主目录
$userHome = $env:USERPROFILE
$cfgJson = "$userHome\.claude.json"
$claudeDir = "$userHome\.claude"
$settingsFile = "$claudeDir\settings.json"

# 2. 创建 .claude.json 跳过初始化引导
if (-not (Test-Path $cfgJson)) {
    @"
{
    "hasCompletedOnboarding": true
}
"@ | Out-File $cfgJson -Encoding utf8
}

# 3. 创建 .claude 文件夹
if (-not (Test-Path $claudeDir)) {
    New-Item $claudeDir -ItemType Directory -Force
}

# 4. 生成空标准settings.json（CC Switch可自动写入路由配置）
if (-not (Test-Path $settingsFile)) {
    @"
{
    "env": {}
}
"@ | Out-File $settingsFile -Encoding utf8
}

Write-Host "配置目录&文件创建完成，重启CC Switch再切换路由"
```

## 5.4 执行完毕后关闭cc switch，重新打开，继续设置路由

# 6 启动claude code

## 6.1 此时所有安装步骤执行完毕，完成了claude code的命令行安装

## 6.2 在任意目录下，按下shift+右键，选择：在此处打开PowerShell窗口，输入claude，即可进入claude工作环境