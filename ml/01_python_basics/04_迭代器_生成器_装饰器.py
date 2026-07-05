# -*- coding: utf-8 -*-
"""
模块1 · 第4课：迭代器、生成器、装饰器
=====================================
适用对象：有其他语言基础。
这三个特性是 Python 的"高级玩家"标志，ML 框架源码中大量使用。
"""


# ============================================================
# 一、可迭代对象 Iterable 与 迭代器 Iterator
# ============================================================
# 可迭代对象：能放在 for 循环里的东西（list/dict/str/file 等）
# 迭代器：实现了 __next__() 的对象，能逐个吐出元素
# 关系：iter(可迭代对象) → 迭代器；next(迭代器) → 下一个元素

nums = [10, 20, 30]
it = iter(nums)         # 列表本身不是迭代器，要先转
print(next(it))         # 10
print(next(it))         # 20
print(next(it))         # 30
# print(next(it))       # ← 再调会抛 StopIteration（迭代结束的信号）

# for 循环的本质：自动调 iter() 拿迭代器，反复调 next() 直到 StopIteration
for x in [1, 2, 3]:
    pass                # 等价于手动 iter + next


# ============================================================
# 二、生成器 generator —— 用函数生成迭代器（最常用！）
# ============================================================
# 含 yield 关键字的函数就是生成器函数。调用它不会立刻执行，
# 而是返回一个生成器对象，每次 next() 执行到 yield 处"暂停"并返回值。

def count_up_to(n):
    """生成 1, 2, ..., n 的生成器"""
    i = 1
    while i <= n:
        yield i                 # 每次执行到这里暂停，把 i 抛出去
        i += 1                  # 下次 next() 从这里继续

gen = count_up_to(3)
print(next(gen))                # 1
print(next(gen))                # 2
print(next(gen))                # 3

# 直接用 for 消费生成器（最常见）
for x in count_up_to(5):
    print("生成:", x, end="  ")
print()


# ------------------------------------------------------------
# 为什么 ML 要用生成器？——「惰性求值，省内存」
# ------------------------------------------------------------
# 比如要处理 1 亿条数据，一次性读进内存会爆。
# 生成器是"按需生成"，每次只在内存里存一条。

def read_large_data(n):
    """模拟逐条产生海量数据"""
    for i in range(n):
        yield [i, i * 2, i * 3]      # 一次只在内存里有一条

# 用法：for 循环逐条处理，内存占用恒定
total = 0
for row in read_large_data(1000000):
    total += row[0]
print("前 100 万项之和:", total)


# ============================================================
# 三、生成器表达式 —— 列表推导式的"惰性版"
# ============================================================
# 语法：把列表推导式的 [] 换成 () 即可
# 好处：不立刻构造整个列表，省内存

# 列表推导式：立即生成所有元素，占内存
squares_list = [i ** 2 for i in range(10)]
# 生成器表达式：惰性，几乎不占内存
squares_gen = (i ** 2 for i in range(10))

print("生成器对象:", squares_gen)    # <generator object ...>
print("逐个取:", next(squares_gen), next(squares_gen))   # 0 1

# 【ML 场景】sum / max / min 等函数可以直接吃生成器，省内存
print("平方和:", sum(i ** 2 for i in range(101)))   # 不需要再加 []


# ============================================================
# 四、装饰器 decorator —— 在不修改原函数的前提下"加功能"
# ============================================================
# 装饰器本质：接收一个函数，返回一个新函数。
# 语法糖：@decorator_name 写在 def 上面，等价于 f = decorator(f)

import time

def timer(func):
    """计时装饰器：测量函数运行耗时"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)        # 调用原函数
        elapsed = time.time() - start
        print(f"  [timer] {func.__name__} 耗时 {elapsed:.4f}s")
        return result                          # 别忘了把原结果返回
    return wrapper

@timer                          # 等价于：slow_func = timer(slow_func)
def slow_func():
    time.sleep(0.1)
    return "完成"

print(slow_func())              # 会先打印耗时，再返回"完成"


# ============================================================
# 五、带参数的装饰器（三层嵌套）
# ============================================================
# 如果装饰器自己也要参数（如 @repeat(3)），需要再包一层函数

def repeat(n):
    """让被装饰函数重复执行 n 次"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)                      # greet 会被调用 3 次
def greet():
    print("Hi!")

greet()


# ============================================================
# 六、实战：用装饰器给 ML 训练函数加日志
# ============================================================
def log_training(func):
    def wrapper(*args, **kwargs):
        print(f"=== 开始 {func.__name__} ===")
        result = func(*args, **kwargs)
        print(f"=== 结束 {func.__name__}, 结果: {result} ===")
        return result
    return wrapper

@log_training
def train_model(epochs):
    print(f"  训练 {epochs} 轮...")
    return {"loss": 0.05, "acc": 0.97}

train_model(epochs=10)


# ============================================================
# 七、迭代器协议：自己实现一个可迭代对象（了解）
# ============================================================
class NumberRange:
    """自定义可迭代对象：生成 [start, stop) 的整数"""
    def __init__(self, start, stop):
        self.cur = start
        self.stop = stop

    def __iter__(self):                 # for 循环会调这个拿迭代器
        return self

    def __next__(self):                 # 每次迭代会调这个
        if self.cur >= self.stop:
            raise StopIteration()       # 结束信号
        v = self.cur
        self.cur += 1
        return v

for x in NumberRange(5, 8):
    print("自定义迭代:", x, end="  ")
print()


# ============================================================
# 小结：本课你应掌握的"ML 高频"写法
# ============================================================
# 1. iter() / next() / StopIteration 的关系
# 2. yield 写生成器 —— 处理大数据必备，省内存
# 3. 生成器表达式 (...) 替代 [...]，配合 sum/max/min 用
# 4. 装饰器 @xxx 的本质：函数变换函数
# 5. 看懂三层嵌套的带参装饰器
#
# 练习（可选）：写一个生成器 fibonacci()，产生斐波那契数列前 n 项。
# 提示：a, b = 0, 1; for _ in range(n): yield a; a, b = b, a + b

def fibonacci(n):
	""" 生成斐波那契数列 """
	a = 0 
	b = 1
	for _ in range(n):
		yield a
        #a = b
        #b = a + b
		a, b = b, a + b

print(list(fibonacci(10)))