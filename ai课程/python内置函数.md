# Python 知识分享：62 个常用内置函数

## 1. print()
**功能：** 打印输出给定的内容

```python
print("Hello, World!")  # 输出：Hello, World!
```

---

## 2. len()
**功能：** 返回对象的长度或元素个数

```python
string = "Hello, World!"
length = len(string)
print(length)  # 输出：13
```

---

## 3. input()
**功能：** 接收用户输入并返回作为字符串

```python
name = input("请输入您的姓名：")
print("您的姓名是：" + name)
```

---

## 4. range()
**功能：** 生成一个指定范围内的整数序列

```python
for num in range(1, 5):
    print(num)  # 输出：1 2 3 4
```

---

## 5. str()
**功能：** 将对象转换为字符串

```python
number = 42
string = str(number)
print(string)  # 输出：'42'
```

---

## 6. int()
**功能：** 将对象转换为整数

```python
string = "42"
number = int(string)
print(number)  # 输出：42
```

---

## 7. float()
**功能：** 将对象转换为浮点数

```python
string = "3.14"
number = float(string)
print(number)  # 输出：3.14
```

---

## 8. type()
**功能：** 返回对象的类型

```python
number = 42
print(type(number))  # 输出：int
```

---

## 9. list()
**功能：** 将可迭代对象转换为列表

```python
string = "Hello"
char_list = list(string)
print(char_list)  # 输出：['H', 'e', 'l', 'l', 'o']
```

---

## 10. tuple()
**功能：** 将可迭代对象转换为元组

```python
list_data = [1, 2, 3]
tuple_data = tuple(list_data)
print(tuple_data)  # 输出：(1, 2, 3)
```

---

## 11. dict()
**功能：** 创建一个字典对象

```python
person = dict(name='Alice', age=25)
print(person)  # 输出：{'name': 'Alice', 'age': 25}
```

---

## 12. set()
**功能：** 创建一个集合对象

```python
numbers = [1, 2, 3, 2, 1]
unique_numbers = set(numbers)
print(unique_numbers)  # 输出：{1, 2, 3}
```

---

## 13. sum()
**功能：** 返回可迭代对象的总和

```python
numbers = [1, 2, 3, 4, 5]
total = sum(numbers)
print(total)  # 输出：15
```

---

## 14. max()
**功能：** 返回可迭代对象的最大值

```python
numbers = [1, 2, 3, 4, 5]
maximum = max(numbers)
print(maximum)  # 输出：5
```

---

## 15. min()
**功能：** 返回可迭代对象的最小值

```python
numbers = [1, 2, 3, 4, 5]
minimum = min(numbers)
print(minimum)  # 输出：1
```

---

## 16. abs()
**功能：** 返回数值的绝对值

```python
number = -42
absolute = abs(number)
print(absolute)  # 输出：42
```

---

## 17. replace()
**功能：** 替代字符串中的某一些子串为另一些字符

```python
st = "i want a apple"
st = st.replace("apple", "mice")
print(st)  # 输出：mice i want a
```

---

## 18. round()
**功能：** 返回一个数值的四舍五入值

```python
number = 3.14159
rounded = round(number, 2)
print(rounded)  # 输出：3.14
```

---

## 19. strip()
**功能：** 去除字符串前面和后面的空格

```python
st = "  hello  "
st = st.strip()
print(st + "end")  # 输出：helloend
```

---

## 20. sorted()
**功能：** 返回一个排序后的可迭代对象

```python
numbers = [5, 2, 4, 1, 3]
sorted_numbers = sorted(numbers)
print(sorted_numbers)  # 输出：[1, 2, 3, 4, 5]
```

---

## 21. reversed()
**功能：** 返回一个反转后的可迭代对象

```python
numbers = [1, 2, 3, 4, 5]
reversed_numbers = list(reversed(numbers))
print(reversed_numbers)  # 输出：[5, 4, 3, 2, 1]
```

---

## 22. zip()
**功能：** 将多个可迭代对象按索引位置组合成元组

```python
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]
zipped = list(zip(names, ages))
print(zipped)  # 输出：[('Alice', 25), ('Bob', 30)]
```

---

## 23. enumerate()
**功能：** 返回可迭代对象中元素的索引和值

```python
names = ['Alice', 'Bob', 'Charlie']
for index, name in enumerate(names):
    print(f"Name at index {index}: {name}")
```

---

## 24. any()
**功能：** 判断可迭代对象中是否存在任何为真的元素

```python
numbers = [0, 1, 2, 3]
print(any(numbers))  # 输出：True
```

---

## 25. all()
**功能：** 判断可迭代对象中所有元素是否都为真

```python
numbers = [0, 1, 2, 3]
print(all(numbers))  # 输出：False
```

---

## 26. slice()
**功能：** 返回一个切片对象，用于切片操作

```python
numbers = [0, 1, 2, 3, 4, 5]
sliced = numbers[slice(2, 5)]
print(sliced)  # 输出：[2, 3, 4]
```

---

## 27. isinstance()
**功能：** 检查对象是否为指定类型的实例

```python
number = 42
print(isinstance(number, int))  # 输出：True
```

---

## 28. callable()
**功能：** 检查对象是否可调用（函数、方法等）

```python
def say_hello():
    print("Hello!")
print(callable(say_hello))  # 输出：True
```

---

## 29. getattr()
**功能：** 返回对象的属性值

```python
class Person:
    name = "Alice"
person = Person()
name = getattr(person, "name")
print(name)  # 输出：Alice
```

---

## 30. setattr()
**功能：** 设置对象的属性值

```python
class Person:
    name = ""
person = Person()
setattr(person, "name", "Alice")
print(person.name)  # 输出：Alice
```

---

## 31. delattr()
**功能：** 删除对象的属性

```python
class Person:
    name = "Alice"
person = Person()
delattr(person, "name")
print(hasattr(person, "name"))  # 输出：False
```

---

## 32. pow()
**功能：** 返回数值的指定次幂

```python
result = pow(2, 3)
print(result)  # 输出：8
```

---

## 33. divmod()
**功能：** 返回两个数值的商和余数

```python
quotient, remainder = divmod(10, 3)
print(quotient, remainder)  # 输出：3 1
```

---

## 34. filter()
**功能：** 使用函数过滤可迭代对象中的元素

```python
numbers = [1, 2, 3, 4, 5]
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(even_numbers)  # 输出：[2, 4]
```

---

## 35. map()
**功能：** 使用函数对可迭代对象中的每个元素进行映射

```python
numbers = [1, 2, 3, 4, 5]
squared_numbers = list(map(lambda x: x ** 2, numbers))
print(squared_numbers)  # 输出：[1, 4, 9, 16, 25]
```

---

## 36. reduce()
**功能：** 使用函数对可迭代对象中的元素进行累积计算

```python
from functools import reduce
numbers = [1, 2, 3, 4, 5]
product = reduce(lambda x, y: x * y, numbers)
print(product)  # 输出：120
```

---

## 37. open()
**功能：** 打开文件并返回文件对象

```python
file = open("example.txt", "r")
content = file.read()
print(content)
file.close()
```

---

## 38. close()
**功能：** 关闭文件

```python
file = open("example.txt", "r")
content = file.read()
file.close()
```

---

## 39. read()
**功能：** 读取文件内容

```python
file = open("example.txt", "r")
content = file.read()
print(content)
file.close()
```

---

## 40. write()
**功能：** 将内容写入文件

```python
file = open("example.txt", "w")
file.write("Hello, World!")
file.close()
```

---

## 41. append()
**功能：** 在列表末尾添加元素

```python
numbers = [1, 2, 3]
numbers.append(4)
print(numbers)  # 输出：[1, 2, 3, 4]
```

---

## 42. extend()
**功能：** 将可迭代对象中的元素添加到列表末尾

```python
numbers = [1, 2, 3]
more_numbers = [4, 5, 6]
numbers.extend(more_numbers)
print(numbers)  # 输出：[1, 2, 3, 4, 5, 6]
```

---

## 43. insert()
**功能：** 在指定索引处插入元素

```python
numbers = [1, 2, 3]
numbers.insert(1, 4)
print(numbers)  # 输出：[1, 4, 2, 3]
```

---

## 44. remove()
**功能：** 移除列表中第一个匹配的元素

```python
numbers = [1, 2, 3, 2, 4]
numbers.remove(2)
print(numbers)  # 输出：[1, 3, 2, 4]
```

---

## 45. pop()
**功能：** 移除并返回指定索引处的元素

```python
numbers = [1, 2, 3]
popped = numbers.pop(1)
print(popped)   # 输出：2
print(numbers)  # 输出：[1, 3]
```

---

## 46. index()
**功能：** 返回第一个匹配元素的索引

```python
numbers = [1, 2, 3, 2, 4]
index = numbers.index(2)
print(index)  # 输出：1
```

---

## 47. count()
**功能：** 返回元素在列表中的出现次数

```python
numbers = [1, 2, 3, 2, 4]
count = numbers.count(2)
print(count)  # 输出：2
```

---

## 48. sort()
**功能：** 对列表进行排序

```python
numbers = [5, 2, 4, 1, 3]
numbers.sort()
print(numbers)  # 输出：[1, 2, 3, 4, 5]
```

---

## 49. reverse()
**功能：** 反转列表中的元素顺序

```python
numbers = [1, 2, 3, 4, 5]
numbers.reverse()
print(numbers)  # 输出：[5, 4, 3, 2, 1]
```

---

## 50. random.random()
**功能：** 来生成随机数

```python
import random
print(random.random())  # 输出：0.2203627
```

---

## 51. time.sleep()
**功能：** 让程序停止一段时间

```python
import time
time.sleep(5)
print('hello')  # hello 会延迟 5 秒后输出
```

---

## 52. listdir()
**功能：** 显示当前目录下的文件

```python
path = r'D:/images'
dirs = os.listdir(path)
for file in dirs:
    print(file)  # 输出 images 下所有文件列表
```

---

## 53. chr()
**功能：** 返回指定 Unicode 代码的字符

```python
char = chr(65)
print(char)  # 输出：'A'
```

---

## 54. ord()
**功能：** 返回字符的 Unicode 代码

```python
code = ord('A')
print(code)  # 输出：65
```

---

## 55. bin()
**功能：** 将整数转换为二进制字符串

```python
binary = bin(10)
print(binary)  # 输出：'0b1010'
```

---

## 56. hex()
**功能：** 将整数转换为十六进制字符串

```python
hexadecimal = hex(16)
print(hexadecimal)  # 输出：'0x10'
```

---

## 57. oct()
**功能：** 将整数转换为八进制字符串

```python
octal = oct(8)
print(octal)  # 输出：'0o10'
```

---

## 58. frozenset()
**功能：** 创建一个可变的字节数组对象

```python
my_array = bytearray([0, 1, 2, 3])
print(my_array)  # 输出：bytearray(b'\x00\x01')
```

---

## 59. bytes()
**功能：** 创建一个不可变的字节数组对象

```python
my_bytes = bytes([0, 1, 2, 3])
print(my_bytes)  # 输出：b'\x00\x01\x02\x03'
```

---

## 60. ascii()
**功能：** 返回一个表示对象的可打印字符串

```python
text = "Hello, 你好"
ascii_text = ascii(text)
print(ascii_text)  # 输出：'Hello, 你好'
```

---

## 61. exec()
**功能：** 执行动态生成的 Python 代码

```python
code = '''
for i in range(5):
    print(i)
'''
exec(code)  # 输出：0 1 2 3 4
```

---

## 62. format()
**功能：** 根据指定的格式进行字符串格式化

```python
name = "Alice"
age = 25
formatted = format("Name: {}, Age: {}", name, age)
print(formatted)  # 输出："Name: Alice, Age: 25"
```

---

**-END-**
