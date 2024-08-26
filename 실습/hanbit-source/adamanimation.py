import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 최적화할 함수 (2D)
def func(x, y):
    return x**2 + y**2

# 기울기 계산 (2D)
def gradient(x, y):
    return 2*x, 2*y

# Adam 애니메이션
def adam_animation():
    fig, ax = plt.subplots()
    x, y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
    z = func(x, y)
    ax.contour(x, y, z, levels=50)

    # 초기 위치
    pos = np.array([1.5, 1.5])
    learning_rate = 0.1
    beta1 = 0.9  # 모멘텀을 위한 지수 평균 가중치
    beta2 = 0.999  # 기울기 제곱의 지수 평균 가중치
    epsilon = 1e-8  # 분모 안정화를 위한 작은 값

    m = np.zeros_like(pos)  # 모멘텀(1차 모멘트) 초기화
    v = np.zeros_like(pos)  # 스케일링(2차 모멘트) 초기화
    t = 0  # 타임스텝

    point, = ax.plot([], [], 'ro')
    path, = ax.plot([], [], 'r-', alpha=0.5)

    positions = [pos.copy()]

    def update(i):
        nonlocal pos, m, v, t
        t += 1
        grad = np.array(gradient(pos[0], pos[1]))

        # 모멘텀과 스케일링 업데이트
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        # 편향 보정
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        # 위치 업데이트
        pos -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        positions.append(pos.copy())

        point.set_data([pos[0]], [pos[1]])
        path.set_data(*zip(*positions))
        return point, path

    ani = animation.FuncAnimation(fig, update, frames=50, interval=200, blit=True)
    plt.show()

# 애니메이션 실행
adam_animation()
