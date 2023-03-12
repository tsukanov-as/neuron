# neuron

## Содержимое репозитория
* Модель вероятностного импульсного нейрона (черновик)
* Классификатор, моделирующий ассоциативную вероятностную импульсную нейронную сеть прямого распространения.  
В сети два слоя:
  * входной - признаки
  * выходной - классы

  Каждая связь в сети пропускает импульс с вероятностью, рассчитанной по теореме Байеса:  
  `W[n,m]=Pr(Class[n]|Feature[m])`  
  Выходной нейрон срабатывает с вероятностью:  
  `Pr(Class[n]|W[n,1] + W[n,2] + ... + W[n,m])`  
  рассчитанной по теореме сложения вероятностей совместных независимых событий:  
  `Pr(A+B)=Pr(A)+Pr(B)-Pr(A*B)`

## Модель вероятностного нейрона

Предполагается, что сработавший входной нейрон ассоциативной сети выдает пачку импульсов фиксированного размера. Дальше каждый импульс распространяется по связям с соответствующими вероятностями. Выходной нейрон суммирует пришедшие импульсы за некоторый промежуток времени (пришедшие одновременно считаются как один) и срабатывает с вероятностью сумма/порог, где порог равен размеру пачки. Обучение сети гипотетически может происходить путем увеличения на константу веса сработавших связей целевого класса (прямолинейный сбор статистики), при этом периодически вес каждой связи сети должен уменьшаться на ту же константу (забывание). Нейрон тоже должен копить статистику своих срабатываний, отношение веса исходящей связи к весу нейрона дает искомую вероятность пропуска импульса по теореме Байеса. Вес нейрона аналогично связям периодически уменьшается.

Полный логический базис возможен если допустить, что существуют следующие модификации нейрона:
* OR - засчитывает импульс если сработал хоть один провод
* XOR - засчитывает импульс если сработал только один провод
* AND - засчитывает импульс если сработали одновременно все провода
* NOT - инвертирует вероятность срабатывания нейрона, может сочетаться с другими модификациями

Обучение (или предотвращение забывания полезного) многослойной сети гипотетически возможно путем обратного распространения награды (увеличение веса сработавших связей).

Гипотезы:
* Сеть делится на слои и каналы. Слои обеспечивают последовательные этапы обработки информации. Например, в слое 1 детектируются первичные признаки, а в слое 2 их кобинации. Каналы представляют собой параллельные независимые друг от друга вероятностные пространства. Например, это могут быть цветовые каналы в отдельных областях сети.
* Пачки импульсов, которыми разряжаются нейроны, могут иметь разные частоты. Каждый провод реагирует только на свою частоту. На какой частоте нейрон принял сигнал, на такой и разряжается. Таким образом на одних и тех же нейронах может быть построена сеть имеющая несколько конфигураций (режимов). Например, гипотетически так может быть реализована "коробка передач" для переключения аллюров.
* Нейроны-детекторы могут быть реализованы как XNOR и/или AND.
* Ассоциативные нейроны могут быть реализованы как OR.
* Детектирование сложных признаков может быть выполнено поэтапно слой за слоем в нескольких параллельных каналах. Первичные признаки раскидываются по каналам таким образом, чтобы в одном канале не было похожих. Детектирование комбинаций в следующих слоях продолжается по тому же принципу. Все каналы выполняют свою работу независимо до самого конца. Каждый канал дает свою оценку сигналов, потом все оценки взвешиваются ассоциативной сетью (классификатор по OR) для получения общих оценок по классам.

```sh
$ git clone https://github.com/tsukanov-as/neuron.git && cd neuron
$ go test -v -count=1 ./...
```
