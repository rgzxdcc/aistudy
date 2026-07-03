# -*- coding: utf-8 -*-
"""
模块1 · 第6课：模块、包、虚拟环境
=================================
适用对象：有其他语言基础。
聚焦 ML 工程必备：如何组织代码、如何隔离依赖、如何安装第三方库。
这是 Python 速通的最后一课，学完即可进入 NumPy/Pandas 阶段。
"""

import os
import sys


# ============================================================
# 一、模块 module：一个 .py 文件就是一个模块
# ============================================================
# 通过 import 引入其他模块，避免把所有代码塞进一个文件。

# 标准库模块：Python 自带，直接 import
import math                          # 整体导入
print("π =", math.pi)
print("向上取整:", math.ceil(2.3))

from math import sqrt, pi            # 只导入需要的内容（更省空间、调用更短）
print("根号 2 =", sqrt(2))

import math as m                     # 起别名（ML 惯例：import numpy as np）
print("e =", m.e)

# ML 中的经典别名约定（背下来！）
#   import numpy as np
#   import pandas as pd
#   import matplotlib.pyplot as plt
#   from sklearn import ...           # Scikit-learn 习惯按子模块导入


# ============================================================
# 二、from import 与 * 的注意事项
# ============================================================
# from x import * 会导入所有公开名字 —— 不推荐！容易名字冲突
# 推荐写明要导入什么：from math import sqrt, ceil

# 查看模块提供了哪些东西
print("math 的部分内容:", [n for n in dir(math) if not n.startswith("_")][:5])


# ============================================================
# 三、包 package：含 __init__.py 的目录
# ============================================================
# 把多个 .py 文件组织进一个带 __init__.py 的目录，就成了"包"。
# __init__.py 可以为空文件，存在即告诉 Python "这是一个包"。
#
#   mypkg/
#   ├── __init__.py
#   ├── data.py
#   └── model.py
#
# 在 model.py 里可以用： from . import data   (. 表示当前包)

# 你的 aistudy 项目目录结构其实就是多个包：
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
print("项目根目录:", os.path.basename(PROJECT))
print("包含的模块目录:", sorted([d for d in os.listdir(PROJECT)
                                  if os.path.isdir(os.path.join(PROJECT, d))
                                  and not d.startswith(".")]))


# ============================================================
# 四、__name__ == "__main__"：模块的"主入口"判断
# ============================================================
# 当一个 .py 被「直接运行」时，它的 __name__ 是 "__main__"
# 当它被 import 时，__name__ 是它的模块名
# 这让同一个文件既能独立运行，又能被别人复用，是 Python 的常见约定。

def main():
    print("这是主程序逻辑")

if __name__ == "__main__":
    # 只有直接运行本文件才会执行；被 import 时不会自动跑
    main()


# ============================================================
# 五、pip：Python 的包管理器（终端里用，不在 .py 里用）
# ============================================================
# 安装第三方库的命令（在 cmd/PowerShell 终端执行）：
#   pip install numpy pandas matplotlib scikit-learn
#
# 常用命令：
#   pip install <包名>            安装
#   pip install <包名>==<版本>    指定版本，如 pip install numpy==1.26.0
#   pip uninstall <包名>          卸载
#   pip list                      查看已安装的包
#   pip show <包名>               查看某个包的详细信息
#   pip install -r requirements.txt   批量按清单安装

# 国内加速（清华源，下载快很多）：
#   pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple


# ============================================================
# 六、虚拟环境：每个项目独立依赖（强烈推荐！）
# ============================================================
# 为什么需要？项目 A 用 numpy 1.x，项目 B 用 numpy 2.x，
# 装在全局会冲突。虚拟环境让每个项目有自己的"依赖隔离空间"。

# 方式 A：venv（Python 自带，最简单）
#   python -m venv .venv                 # 在项目根目录创建（.venv 是约定名）
#   # Windows 激活：
#   .venv\Scripts\activate
#   # Linux/Mac 激活：
#   source .venv/bin/activate
#   # 激活后再 pip install 就只影响这个环境

# 方式 B：conda（数据科学/ML 常用，适合装 numpy/torch 这种含 C 扩展的库）
#   conda create -n ml python=3.11        # 创建名为 ml 的环境，指定 Python 版本
#   conda activate ml                     # 激活
#   conda install numpy pandas scikit-learn
#
# 你当前看到的 TraeAI-3 就是一个 conda 环境名。

# 如何让 Trae 识别虚拟环境？
#   左下角状态栏点 Python 解释器 → 选择你创建的 .venv 或 conda 环境路径
#   之后右上角运行按钮就会用这个环境


# ============================================================
# 七、requirements.txt —— 锁定项目依赖（工程化必备）
# ============================================================
# 把项目依赖写成一个文本文件，别人 clone 后一键装齐：
#   numpy==1.26.0
#   pandas==2.1.4
#   matplotlib==3.8.0
#   scikit-learn==1.3.2
#
# 生成当前环境的依赖清单：
#   pip freeze > requirements.txt
# 按清单安装：
#   pip install -r requirements.txt

print("\n当前 Python 版本:", sys.version.split()[0])
print("当前 Python 路径:", sys.executable)
print("（这就是你运行脚本时实际使用的解释器）")


# ============================================================
# 八、import 搜索路径 sys.path（了解）
# ============================================================
# Python 找模块的顺序：
#   1. 当前脚本所在目录
#   2. 环境变量 PYTHONPATH 中的目录
#   3. 解释器安装目录的标准库
#   4. 第三方包目录 site-packages
print("\nimport 搜索路径（前 3 条）:")
for p in sys.path[:3]:
    print("  ", p)


# ============================================================
# 小结：本课你应掌握的"ML 工程"能力
# ============================================================
# 1. import / from import / import ... as 三种导入写法
# 2. 包 = 带 __init__.py 的目录，组织多文件项目
# 3. __name__ == "__main__" 的主入口判断
# 4. 用 venv 或 conda 创建虚拟环境隔离依赖
# 5. pip install / requirements.txt 管理第三方库
# 6. 国内用清华源加速 pip 下载
#
# === 模块 1 · Python 速通 结业 ===
# 恭喜！到这里你已经掌握了 ML 编程所需的全部 Python 基础。
# 下一阶段我们将进入模块 2：NumPy 数值计算。
