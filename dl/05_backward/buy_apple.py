from layer_native import MulLayer

apple_price = 100
apple_num = 2
tax = 1.1

# layer
mul_layer_apple = MulLayer()
mul_layer_tax = MulLayer()

# forward
apple_price = mul_layer_apple.forward(apple_price, apple_num)
price = mul_layer_tax.forward(apple_price, tax)

print(price)

#backward
dprice = 1
dapple_price, dtax = mul_layer_tax.backward(dprice)
dapple, dnum = mul_layer_apple.backward(dapple_price)

print(dapple, dnum, dtax)

# import numpy as np

# x = np.array([[10, -0.5], [-2.0, 3.0]])
# print(x)
# mask = (x <= 0)
# print (mask)

