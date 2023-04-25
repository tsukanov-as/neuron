from matplotlib import pyplot as plt
from matplotlib import animation
import random
from datetime import datetime

PF1 = 0.2 # вероятность срабатывания нейрона F1 (например, внешний сигнал с рецептора)
PF2 = 0.9 # вероятность срабатывания нейрона F2 (например, внешний сигнал с рецептора)

PW1 = 0.8 # вероятность срабатывания связи F1 -> C1 (результат деления веса нейрона F1 на вес связи с C1)
PW2 = 0.2 # вероятность срабатывания связи F1 -> C2 (результат деления веса нейрона F1 на вес связи с C2)
PW3 = 0.6 # вероятность срабатывания связи F2 -> C1 (результат деления веса нейрона F2 на вес связи с C1)
PW4 = 0.4 # вероятность срабатывания связи F2 -> C2 (результат деления веса нейрона F2 на вес связи с C2)

random.seed(datetime.now().timestamp())
rnd = random.random

fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(7, 6.5)

ax = plt.axes(xlim=(0, 100), ylim=(0, 100))

# нейроны AND
F1 = plt.Circle((20, 80), 5, zorder=2)
F2 = plt.Circle((20, 20), 5, zorder=2)

# нейроны OR
C1 = plt.Circle((80, 80), 5, zorder=2)
C2 = plt.Circle((80, 20), 5, zorder=2)

# связи
W1 = plt.Line2D((F1.center[0], C1.center[0]), (F1.center[1], C1.center[1]), lw=2, alpha=0.5, zorder=1)
W2 = plt.Line2D((F1.center[0], C2.center[0]), (F1.center[1], C2.center[1]), lw=2, alpha=0.5, zorder=1)
W3 = plt.Line2D((F2.center[0], C1.center[0]), (F2.center[1], C1.center[1]), lw=2, alpha=0.5, zorder=1)
W4 = plt.Line2D((F2.center[0], C2.center[0]), (F2.center[1], C2.center[1]), lw=2, alpha=0.5, zorder=1)

ax.add_patch(F1)
ax.add_patch(F2)

ax.add_patch(C1)
ax.add_patch(C2)

ax.add_line(W1)
ax.add_line(W2)
ax.add_line(W3)
ax.add_line(W4)

ax.text(x=10, y=80, s='F1')
F1_total = 0
F1_total_text = ax.text(x=10, y=70, s="0")

ax.text(x=10, y=20, s='F2')
F2_total = 0
F2_total_text = ax.text(x=10, y=10, s="0")

pf1 = PW1*PF1
pf2 = PW3*PF2
ax.text(x=87, y=80, s=f'C1 ({pf1+pf2-pf1*pf2:.2f})')
C1_total = 0
C1_total_text = ax.text(x=77, y=70, s="0")

pf1 = PW2*PF1
pf2 = PW4*PF2
ax.text(x=87, y=20, s=f'C2 ({pf1+pf2-pf1*pf2:.2f})')
C2_total = 0
C2_total_text = ax.text(x=77, y=10, s="0")

total = 0
total_text = ax.text(x=40, y=90, s="total: 0")

clock = 0

def animate(i):
    global clock
    global total
    global F1_total
    global F2_total
    global C1_total
    global C2_total
    clock += 1
    if clock % 2 == 0:
        total += 1
        total_text.set_text(f"total: {total}")
        spike1 = False
        spike2 = False
        if rnd() < PF1: # сработал нейрон F1
            F1_total += 1
            F1_total_text.set_text(f"{F1_total} ({F1_total/total:.2f})")
            F1.set_color("r")
            if rnd() < PW1: # сработала связь F1 -> C1
                spike1 = True
                W1.set_color("r")
            if rnd() < PW2: # сработала связь F1 -> C2
                spike2 = True
                W2.set_color("r")
        spike3 = False
        spike4 = False
        if rnd() < PF2: # сработал нейрон F2
            F2_total += 1
            F2_total_text.set_text(f"{F2_total} ({F2_total/total:.2f})")
            F2.set_color("r")
            if rnd() < PW3: # сработала связь F2 -> C1
                spike3 = True
                W3.set_color("r")
            if rnd() < PW4: # сработала связь F2 -> C2
                spike4 = True
                W4.set_color("r")
        if spike1 or spike3: # сработал нейрон C1
            C1_total += 1
            C1_total_text.set_text(f"{C1_total} ({C1_total/total:.2f})")
            C1.set_color("r")    
        if spike2 or spike4: # сработал нейрон C2
            C2_total += 1
            C2_total_text.set_text(f"{C2_total} ({C2_total/total:.2f})")
            C2.set_color("r")
    else:
        F1.set_color("C0")
        F2.set_color("C0")
        C1.set_color("C0")
        C2.set_color("C0")
        W1.set_color("C0")
        W2.set_color("C0")
        W3.set_color("C0")
        W4.set_color("C0")
    
    return W1, W2, W3, W4, F1, F2, C1, C2, total_text, F1_total_text, F2_total_text, C1_total_text, C2_total_text

anim = animation.FuncAnimation(fig, animate, frames=1000, interval=10, blit=True)

# mywriter = animation.FFMpegWriter(fps=30)
# anim.save('spike_animation.mp4', writer=mywriter)

plt.show()