# -*- coding: utf-8 -*-
"""
模块1 · 第2课：函数与 Lambda
==============================
适用对象：有其他语言基础，需要快速掌握 Python 函数特性。
聚焦 ML 代码中高频出现的写法：默认参数、可变参数、关键字参数、匿名函数、高阶函数。
"""


# ============================================================
# 一、函数基础与默认参数
# ============================================================
# Python 用 def 定义函数，用缩进表示函数体（不用大括号）。
# 默认参数：在定义时直接赋值，调用时可省略。

def greet(name, greeting="你好"):   # greeting 有默认值
    return f"{greeting}, {name}!"

print(greet("小明"))               # 省略默认参数 -> 你好, 小明!
print(greet("Alice", "Hi"))        # 覆盖默认值 -> Hi, Alice!

# 【ML 场景】很多 ML 函数都有大量默认参数，比如：
# model = SomeModel(learning_rate=0.01, max_depth=3)
# 你只需指定关心的参数，其余走默认。


# ============================================================
# 二、可变位置参数 *args —— 接收任意多个位置参数
# ============================================================
# 参数名习惯叫 *args，但真正起作用的是 * 号。
# 收集到的多个参数会被打包成一个「元组」。

def sum_all(*numbers):
    print("类型:", type(numbers))   # <class 'tuple'>
    return sum(numbers)

print("求和:", sum_all(1, 2, 3, 4, 5))   # 传入 5 个参数

# 反向操作：在调用时用 * 把列表/元组「拆开」逐个传入
nums = [10, 20, 30]
print("拆包传入:", sum_all(*nums))      # 等价于 sum_all(10, 20, 30)


# ============================================================
# 三、可变关键字参数 **kwargs —— 接收任意多个"键=值"
# ============================================================
# 收集到的多个键值对会被打包成一个「字典」。
# 这是 Python 里极其灵活的机制，几乎所有 ML 库都依赖它做配置透传。

def show_config(**options):
    print("配置字典:", options)
    for k, v in options.items():
        print(f"  {k} = {v}")

show_config(learning_rate=0.001, batch_size=32, optimizer="adam")

# 反向操作：用 ** 把字典拆成关键字参数传入
cfg = {"lr": 0.01, "epochs": 10}
show_config(**cfg)


# ============================================================
# 四、参数组合顺序（重要！）
# ============================================================
# 完整顺序必须是：位置参数 / 默认参数 / *args / 关键字仅参数 / **kwargs
# 其中「关键字仅参数」是指必须用 key=value 传递的参数。

def train(model, lr=0.01, *, verbose=True, **extras):
    #       ^^^^^  ^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^
    #       位置    默认      * 之后全是「必须用键传递」的参数
    print(f"训练 {model}, lr={lr}, verbose={verbose}, 其他={extras}")

train("ResNet")                              # verbose 走默认
train("ResNet", lr=0.05, verbose=False)      # verbose 必须用键传
# train("ResNet", True)  ← 这行会报错！因为 verbose 必须用键传

# 【ML 场景】Scikit-learn 大量使用这种设计：核心参数按位置传，
# 可选项必须写清楚参数名，避免记不住参数顺序导致误用。


# ============================================================
# 五、Lambda 匿名函数 —— 一行写完的小函数
# ============================================================
# 语法：lambda 参数列表: 表达式（只能是单个表达式，不能写语句）
# 适合"用一次就丢"的简单逻辑，ML 数据处理时极其常用。

# 普通写法
def square(x):
    return x ** 2

# Lambda 等价写法（赋给变量也能用，但更常见是直接传入）
sq = lambda x: x ** 2
print("Lambda 平方:", sq(5))           # 25

# 带 multiple 参数
add = lambda a, b: a + b
print("Lambda 相加:", add(3, 4))        # 7

# 【ML 场景】配合 map / filter / sorted 使用（见下一节）


# ============================================================
# 六、高阶函数：map / filter / sorted —— 配合 Lambda 食用最佳
# ============================================================
# 高阶函数 = 接收函数作为参数的函数。ML 里常用来批量变换数据。

# 6.1 map(函数, 可迭代对象)：对每个元素应用函数
prices = [100, 200, 300]
discounted = list(map(lambda p: p * 0.8, prices))   # 全场 8 折

print("打折后:", discounted)   # [80.0, 160.0, 240.0]
# 注意：map 在 Python3 返回迭代器，要用 list() 才能看到结果

# 6.2 filter(函数, 可迭代对象)：保留使函数返回 True 的元素
scores = [55, 82, 90, 47, 73]
passed = list(filter(lambda s: s >= 60, scores))
print("及格的:", passed)        # [82, 90, 73]

# 6.3 sorted(可迭代对象, key=函数, reverse=是否降序)
students = [("Alice", 88), ("Bob", 75), ("Carol", 95)]
# 按分数从小到大排
by_score = sorted(students, key=lambda s: s[1])
print("按分数升序:", by_score)
# 按分数从大到小排
by_score_desc = sorted(students, key=lambda s: s[1], reverse=True)
print("按分数降序:", by_score_desc)

# 【对比】以上三种用列表推导式也能做（第1课学的）：
#  map:     [p * 0.8 for p in prices]
#  filter:  [s for s in scores if s >= 60]
# Python 社区普遍推荐用推导式，可读性更好；但读别人代码时仍需看懂 map/filter。


# ============================================================
# 七、变量作用域 LEGB（了解即可）
# ============================================================
# Python 查找变量的顺序：Local → Enclosing → Global → Builtin
#   Local     : 函数内部
#   Enclosing: 外层嵌套函数
#   Global    : 模块顶层
#   Builtin   : Python 内置（如 len、print、sum）

x = "全局"          # Global
def outer():
    x = "外层"       # Enclosing    
    def inner():
        x = "内层"   # Local 优先  
        print("inner 看到:", x)
    inner()
    print("outer 看到:", x)
outer()
print("模块看到:", x)

# 函数内若想修改"全局"变量，需用 global 声明（一般不推荐，容易出 bug）
counter = 0
def increment():
    global counter
    counter += 1
increment()
print("counter:", counter)   # 1


# ============================================================
# 小结：本课你应掌握的"ML 高频"写法
# ============================================================
# 1. 默认参数：def f(x, lr=0.01)
# 2. *args 收集任意位置参数（元组）、**kwargs 收集任意关键字参数（字典）
# 3. 「* 之后必须用 key=value 传」—— Scikit-learn API 的核心设计
# 4. Lambda + map / filter / sorted 做批量数据变换
# 5. 优先用列表推导式（更 Pythonic），但要能读懂 map/filter 写法
#
# 练习（可选）：给定 data = [1, -2, 3, -4, 5]
#   (1) 用 filter + lambda 筛出正数
#   (2) 用 map + lambda 取每个数的平方
#   (3) 改用列表推导式再写一遍

data = [1, -2, 3, -4, 5]
positive = list(filter(lambda x: x > 0, data))
print(positive)
square = list(map(lambda d: d ** 2, data)) # type: ignore
print(square)

positive = [x for x in data if x > 0]
print(positive)
square = [x ** 2 for x in data] # type: ignore
print(square)
