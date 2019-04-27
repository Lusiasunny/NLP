import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import random
# 问题：为什么我得到的结果很随机，有的时候能得到很小的损失，有时很大？

content = pd.read_csv('train.csv')
content = content.dropna()
age_with_frares = content[(content['fare'] < 400) & (content['fare'] > 130) & (content['age'] >22)]
# print(age_with_frares.head())

sub_fare = age_with_frares['fare']
sub_age = age_with_frares['age']

def func(age, k ,b):
    return k * age + b

def loss(y, yhat):
    """
    y：真实的fare
    yhat: 估计的fare estimated
    返回值：估计的fare值有多好
    """
    # return np.mean(np.abs(y - yhat))
    # return np.mean(np.square(y - yhat))
    return np.mean(np.sqrt(y - yhat)) # 开根号，减小点的影响
    # 不同损失的区别：平方后拉大距离较远的两个向量的距离。简称：直接绝对值：L1，平方L2，立方L3


min_error_rate = float('inf')
print(min_error_rate)


loop_times = 800

losses = []

change_directions = [
# k,b值增大还是减小
(+1, -1),
(+1, +1),
(-1, +1),
(-1, -1)
]

k_hat = random.random() * 20 -10
b_hat = random.random() * 20 -10
best_k, best_b = k_hat, b_hat
best_direction = None

def step():
    # return random.random() * 2 -1 # 目的是生成一个-1到+1的随机数，这里也是在改方向，而且是随机改。。。。
    # 所以，并不需要负数。
    return random.random()


direction = random.choice(change_directions)

# 2时15分
def derivate_k(y, yhat, x):
    abs_values = [1 if (y_i - yhat_i) > 0 else -1 for y_i, yhat_i in zip(y, yhat)]
    return np.mean([a * x_i for a, x_i in zip(abs_values, x)])

def derivate_b(y, yhat):
    abs_values = [1 if (y_i - yhat_i) > 0 else -1 for y_i, yhat_i in zip(y, yhat)]
    return np.mean([a * -1 for a in abs_values])

learning_rate = 1e-3 # 可以写成一个函数，loss越大，变化越快，loss越小，变化越慢

while loop_times > 0:
    k_delta = -1 * learning_rate * derivate_k(sub_fare, func(sub_age, k_hat, b_hat),sub_age)
    b_delta = -1 * learning_rate * derivate_b(sub_fare, func(sub_age, k_hat, b_hat))


    # k_delta_directions, b_delta_directions = direction

    # k_delta = k_delta_directions * step()
    # b_delta = b_delta_directions * step()

    # new_k = k_hat + k_delta
    # new_b = b_hat + b_delta
    
    k_hat += k_delta
    b_hat += b_delta



    # k_hat = random.randint(-10,10)
    # b_hat = random.randint(-10,10)
    
    estimated_fares = func(sub_age, k_hat, b_hat)
    error_rate = loss(y = sub_fare, yhat = estimated_fares)

    print('loop == {}'.format(loop_times))
    print('f(age) = {}*age + {}, with error rate: {}'.format(best_k, best_b, error_rate))
    

    # 如果min_error_rate变得更好，把方向记录下来
    # if error_rate < min_error_rate:

    #   min_error_rate = error_rate
    #   best_k, best_b = k_hat, b_hat
    #   direction = (k_delta_directions, b_delta_directions)

    #   losses.append(min_error_rate)
    #   print(loop_times)

    #   # print('min_error_rate:',min_error_rate)
    #   print('f(age) = {} * age + {},with error rate: {}'.format(best_k,best_b,min_error_rate))
    # else:
    #   # direction = random.choice(change_directions)
    #   direction = random.choice(list(set(change_directions) - {(k_delta_directions, b_delta_directions)})) # 除去老方向

    loop_times -= 1


plt.scatter(sub_age, sub_fare)
# plt.plot(sub_age,estimated_fares, c = 'r')
plt.plot(sub_age, func(sub_age, best_k,best_b), c = 'r')
plt.show()


# 收敛,说明这个模型是可学习的
# plt.plot(range(len(losses)), losses)
# plt.show()



