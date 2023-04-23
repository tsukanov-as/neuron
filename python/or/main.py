import random
from datetime import datetime

random.seed(datetime.now().timestamp())

cassette_size = 10 # размер пачки спайков (количество испытаний)

ap = 0.5 # вероятность спайка на входящей связи 1
bp = 0.5 # вероятность спайка на входящей связи 2

total_fired = 0 # сумма срабатываний нейрона
for i in range(1000): # подсчет вероятности срабатывания OR нейрона путем эксперимента
    stress = 0 # стресс нейрона
    for i in range(cassette_size):
        total_spike = 0 # сумма спайков
        total_spike += 1 if random.random() < ap else 0
        total_spike += 1 if random.random() < bp else 0
        if total_spike > 0: # пришел хотя бы один спайк (функция OR)
            stress += 1

    god_dice = random.random()
    np = stress / cassette_size # вероятность срабатывания нейрона
    if god_dice < np:
        total_fired += 1

print("вероятность срабатывания OR нейрона из эксперимента:", total_fired / 1000)
print("вероятность срабатывания OR нейрона расчетная:", ap+bp-ap*bp)
