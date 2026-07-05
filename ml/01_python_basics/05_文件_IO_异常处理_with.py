# -*- coding: utf-8 -*-
"""
模块1 · 第5课：文件 I/O、异常处理、with 上下文管理
=================================================
适用对象：有其他语言基础。
聚焦 ML 中最常用的：读写 CSV/TXT、try/except、with、json。
"""

import os
import json


# ============================================================
# 一、基础文本文件读写
# ============================================================
# open(路径, 模式) 模式：r读 / w覆盖写 / a追加 / b二进制 / t文本(默认)
# 编码问题：Windows 默认 gbk，处理中文/ML 数据强烈建议显式指定 utf-8

# 当前脚本所在目录，便于跨机器运行
HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_TXT = os.path.join(HERE, "_temp_sample.txt")

# 1.1 写文件
f = open(SAMPLE_TXT, "w", encoding="utf-8")
f.write("第一行：机器学习\n")
f.write("第二行：深度学习\n")
f.close()                              # 不用 with 时必须手动 close！

# 1.2 读文件（全量）
f = open(SAMPLE_TXT, "r", encoding="utf-8")
content = f.read()                     # 一次性读全部
f.close()
print("全量读:\n", content, end="")

# 1.3 按行读（处理大文件推荐，类似生成器）
print("逐行读:")
with open(SAMPLE_TXT, "r", encoding="utf-8") as f:    # 见第三节 with
    for line in f:                                     # 文件对象本身可迭代
        print("  >", line.rstrip())                   # rstrip 去行尾换行符


# ============================================================
# 二、try / except / finally —— 异常处理
# ============================================================
# Python 没有 catch，用 except。可以针对不同异常类型分别处理。

def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:          # 捕获特定异常
        print("  [警告] 除以 0")
        return None
    except TypeError as e:             # 捕获并拿到异常对象
        print(f"  [警告] 类型错误: {e}")
        return None
    finally:                           # 无论是否出错都会执行（常用于清理）
        # print("  (finally 执行清理)")
        pass

print("10/2 =", safe_divide(10, 2))    # 5.0
print("10/0 =", safe_divide(10, 0))    # [警告] 除以 0 → None
print("10/'a' =", safe_divide(10, "a"))

# 抛出异常：raise
def check_score(score):
    if score < 0 or score > 100:
        raise ValueError(f"分数必须在 0~100，收到 {score}")   # 主动抛
    return score

try:
    check_score(150)
except ValueError as e:
    print(f"  捕获自定义异常: {e}")

# 自定义异常类（继承 Exception）
class DataError(Exception):
    """自定义：数据格式错误"""
    pass

try:
    raise DataError("CSV 列数不一致")
except DataError as e:
    print(f"  自定义异常: {e}")


# ============================================================
# 三、with 语句 —— 自动管理资源（ML 中必备写法）
# ============================================================
# with 块结束时，会自动调用对象的 __exit__，即使中间抛异常也能正确关闭。
# 读写文件、连接数据库、加锁，都强烈推荐用 with。

# 对比：不用 with 容易忘记 close；用 with 安全简洁
with open(SAMPLE_TXT, "a", encoding="utf-8") as f:
    f.write("第三行：with 追加\n")    # 离开 with 块自动 close
# f.write("xxx")                       # ← 这里再写会报错（已关闭）

# 同时打开多个文件
out_path = os.path.join(HERE, "_temp_copy.txt")
with open(SAMPLE_TXT, encoding="utf-8") as src, open(out_path, "w", encoding="utf-8") as dst:
    for line in src:
        dst.write(line)
print("复制完成")


# ============================================================
# 四、读写 JSON（配置文件、ML 结果保存常用）
# ============================================================
config = {
    "model": "RandomForest",
    "n_estimators": 100,
    "max_depth": None,
    "classes": ["cat", "dog", "bird"],
}
config_path = os.path.join(HERE, "_temp_config.json")

# 写 JSON：ensure_ascii=False 让中文不转义，indent=2 美化缩进
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

# 读 JSON
with open(config_path, "r", encoding="utf-8") as f:
    loaded = json.load(f)
print("读回 JSON:", loaded["model"], loaded["classes"])


# ============================================================
# 五、os.path 与路径处理（跨平台必备）
# ============================================================
# Windows 用 \，Linux/Mac 用 /，用 os.path 自动处理，避免硬编码

data_dir = os.path.join(HERE, "..", "06_datasets")
print("数据集目录:", os.path.normpath(data_dir))

# 常用函数
print("是否存在:", os.path.exists(SAMPLE_TXT))           # True
print("是否文件:", os.path.isfile(SAMPLE_TXT))           # True
print("文件大小:", os.path.getsize(SAMPLE_TXT), "字节")
print("扩展名:", os.path.splitext("data.csv"))           # ('data', '.csv')
print("文件名:", os.path.basename("/a/b/c.csv"))         # c.csv
print("目录名:", os.path.dirname("/a/b/c.csv"))          # /a/b

# 列出目录内容
if os.path.isdir(data_dir):
    print("数据集目录内容:", os.listdir(data_dir))


# ============================================================
# 六、清理临时文件（演示完即删）
# ============================================================
for p in [SAMPLE_TXT, out_path, config_path]:
    if os.path.exists(p):
        os.remove(p)
        print(f"已删除: {os.path.basename(p)}")


# ============================================================
# 小结：本课你应掌握的"ML 高频"写法
# ============================================================
# 1. open(..., encoding="utf-8") 是读写文本的标准姿势
# 2. with open(...) as f: 自动关闭，处理文件首选
# 3. try / except 捕获特定异常；raise 主动抛；自定义异常继承 Exception
# 4. json.dump / json.load 保存配置和结果
# 5. os.path.join 处理路径，避免跨平台问题
#
# 练习（可选）：写一个函数 load_csv_line(path)，逐行读取一个 CSV 文件，
#   按 "," 分割成列表返回。要求：文件不存在时打印友好提示而非崩溃。

import csv

def load_csv_line(path):
    if not os.path.exists(path):
        print("文件不存在")
        return
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for line in reader:    # csv.reader 已按逗号拆分，line 就是 list
            yield line

HERE = os.path.dirname(os.path.abspath(__file__))
data_csv = os.path.join(HERE, "..", "06_datasets/students_cleaned.csv")
for line in load_csv_line(data_csv):
    print(line)