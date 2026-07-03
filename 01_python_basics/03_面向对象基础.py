# -*- coding: utf-8 -*-
"""
模块1 · 第3课：面向对象基础
==========================
适用对象：有其他语言基础，需要快速掌握 Python 的类与对象写法。
聚焦 ML 代码中高频出现的特性：类定义、继承、魔法方法、@property、@classmethod。
"""


# ============================================================
# 一、最简单的类：用 class 关键字定义
# ============================================================
# Python 类的方法第一个参数必须是 self（指向实例本身），调用时不显式传。

class Dog:
    # __init__ 是构造函数（初始化方法），相当于其他语言的 constructor
    def __init__(self, name, age):
        self.name = name      # 实例属性
        self.age = age

    def bark(self):           # 实例方法
        return f"{self.name} 汪汪叫！"

d = Dog("旺财", 3)             # 实例化，不需要 new 关键字
print(d.bark())                # 旺财 汪汪叫！
print(d.name, d.age)           # 直接访问属性


# ============================================================
# 二、类属性 vs 实例属性（容易踩坑！）
# ============================================================
class Cat:
    species = "猫科"           # 类属性：所有实例共享，定义在方法外

    def __init__(self, name):
        self.name = name       # 实例属性：每个实例独立

c1 = Cat("Tom")
c2 = Cat("Kitty")
print(c1.species, c2.species)  # 猫科 猫科（共享类属性）
print(c1.name, c2.name)        # Tom Kitty（各自独立）

Cat.species = "哺乳动物"        # 改类属性，所有实例都受影响
print(c1.species)              # 哺乳动物


# ============================================================
# 三、继承：在括号里写父类
# ============================================================
class Animal:
    def __init__(self, name):
        self.name = name

    def eat(self):
        return f"{self.name} 在吃饭"

class Bird(Animal):            # Bird 继承 Animal
    def fly(self):
        return f"{self.name} 在飞"

b = Bird("小鸟")
print(b.eat())                 # 继承自 Animal 的方法
print(b.fly())                 # 自己的方法

# 调用父类方法：用 super()
class Parrot(Animal):
    def __init__(self, name, color):
        super().__init__(name)        # 调用父类构造函数
        self.color = color

    def info(self):
        return f"{self.color}色的 {self.name}"

p = Parrot("Polly", "绿")
print(p.info(), p.eat())


# ============================================================
# 四、魔法方法（dunder methods）—— 运算符重载的本质
# ============================================================
# 形如 __xxx__ 的方法叫魔法方法。它们让自定义类能使用内置操作符。
# ML 中自定义数据结构、模型对象时常用。

class Vector:
    """二维向量：演示 __repr__、__add__、__len__"""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):                 # 控制 print / 调试输出
        return f"Vector({self.x}, {self.y})"

    def __add__(self, other):           # 重载 + 运算符
        return Vector(self.x + other.x, self.y + other.y)

    def __eq__(self, other):            # 重载 == 运算符
        return self.x == other.x and self.y == other.y

    def __len__(self):                  # 控制 len()
        return 2

v1 = Vector(1, 2)
v2 = Vector(3, 4)
print(v1)                               # Vector(1, 2)
print(v1 + v2)                          # Vector(4, 6)  ← 用 + 自动触发 __add__
print(v1 == Vector(1, 2))               # True
print(len(v1))                          # 2

# 常用魔法方法速查：
#   __init__    构造      __repr__   打印/调试
#   __str__     str()     __len__    len()
#   __add__     +         __mul__    *
#   __eq__      ==        __lt__     <
#   __getitem__ obj[k]    __iter__   for x in obj


# ============================================================
# 五、@property —— 把方法伪装成属性
# ============================================================
# 让"取值"像属性访问一样自然，但内部可以加逻辑（如校验、计算）。

class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius        # 约定：_ 开头表示"内部使用"

    @property
    def fahrenheit(self):              # 像属性一样被访问，但实际是方法
        return self._celsius * 9 / 5 + 32

t = Temperature(100)
print(t.fahrenheit)                    # 212.0  ← 不加括号！像访问属性


# ============================================================
# 六、@classmethod 与 @staticmethod
# ============================================================
class DateHelper:
    @staticmethod
    def is_leap_year(year):            # 不需要 self/cls，纯工具函数
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    @classmethod
    def from_string(cls, date_str):    # cls 指向类本身，常用作"替代构造函数"
        y, m, d = map(int, date_str.split("-"))
        return f"{y}年{m}月{d}日"

print(DateHelper.is_leap_year(2024))   # True
print(DateHelper.from_string("2026-7-2"))


# ============================================================
# 七、dataclass —— 用最少代码定义数据类（Python 3.7+，ML 里很常用）
# ============================================================
from dataclasses import dataclass

@dataclass
class Sample:                          # 自动生成 __init__ / __repr__ / __eq__
    feature: list                      # 类型注解（运行时不强制，但 IDE/工具会读）
    label: int
    weight: float = 1.0                # 同样支持默认值

s = Sample(feature=[1.7, 65], label=1)
print(s)                               # Sample(feature=[1.7, 65], label=1, weight=1.0)


# ============================================================
# 小结：本课你应掌握的"ML 高频"写法
# ============================================================
# 1. class 定义、__init__ 构造、self 指向实例
# 2. 继承用 class 子类(父类)，super() 调父类方法
# 3. __repr__ / __add__ / __eq__ 等魔法方法 = 运算符重载
# 4. @property 把方法变属性，@classmethod / @staticmethod 的区别
# 5. @dataclass 一行定义数据类 —— ML 自定义数据结构首选
#
# 练习（可选）：用 @dataclass 定义一个 Student 类，包含 name、score，
#   并添加一个 @property 方法 passed（score >= 60 返回 True）。
