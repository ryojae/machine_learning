import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 최적화할 함수 (2D)
def func(x, y):
    return x**2 + y**2

# 기울기 계산 (2D)
def gradient(x, y):
    return 2*x, 2*y

# Adagrad 애니메이션
def adagrad_animation():
    fig, ax = plt.subplots()
    x, y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
    z = func(x, y)
    ax.contour(x, y, z, levels=50)

    # 초기 위치
    pos = np.array([1.5, 1.5])
    learning_rate = 1.0
    epsilon = 1e-8

    # 기울기 제곱 합 초기화
    grad_squared_sum = np.zeros_like(pos)

    point, = ax.plot([], [], 'ro')
    path, = ax.plot([], [], 'r-', alpha=0.5)

    positions = [pos.copy()]

    def update(i):
        nonlocal pos, grad_squared_sum
        grad = np.array(gradient(pos[0], pos[1]))

        # 기울기 제곱 합 업데이트
        grad_squared_sum += grad**2

        # 학습률 조정
        adjusted_lr = learning_rate / (np.sqrt(grad_squared_sum) + epsilon)

        # 위치 업데이트
        pos -= adjusted_lr * grad
        positions.append(pos.copy())

        # pos 값을 시퀀스로 전달
        point.set_data([pos[0]], [pos[1]])
        path.set_data(*zip(*positions))
        return point, path

    ani = animation.FuncAnimation(fig, update, frames=50, interval=200, blit=True)
    plt.show()

# 애니메이션 실행
adagrad_animation()
