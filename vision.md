# Гипотеза о зрении
Зрительная кора реализует как минимум два алгоритма распознавания образов, которые работают в тандеме и компенсируют недостатки друг друга:
1. Быстрое детектирование образов аналогом сверточной сети прямого распространения.
2. Медленное детектирование, автоматом, работающим последовательно поверх признаков.

Оба алгоритма используют для своей работы признаки/образы, добытые "конвейером" зрительной коры.
Например, первый этап конвейера детектирует линии, а второй образы, составленные из линий (треугольники, к примеру)
Признаки/образы любого этапа доступны автомату, как алфавит для распознавания сложных образов.
В то же время эти признаки/образы являются частью аналога сверточной сети.

Кроме того, образы с любого этапа конвейера возможно доступны непосредственно вниманию.

Обучение конвейера возможно идет от общего к частному. Сначала обучается условно последний этап по первичным признакам, и только потом образ дробится на детали в промежуточных этапах. Это ускоряет и упрощает процедуру обучения, так как сначала путем простого накопления статистики на ходу обучается двуслойная сеть, которая потом дает обратную связь для выделения признаков и при этом уже частично решает задачу распознавания образов.