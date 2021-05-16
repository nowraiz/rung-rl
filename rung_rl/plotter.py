import matplotlib.pyplot as plt

fig = plt.figure()
plt.ion()


def plot(x, radiant, dire):
    plt.cla()
    plot_radiant(x, radiant)
    if dire:
        plot_dire(x, dire)
    plot_trendline(x)
    plt.legend()
    plt.grid(True, axis='y')
    plt.draw()
    plt.pause(0.001)

def plot_reward(x, average_r, average_d = None):
    plt.cla()
    if average_d:
        plt.plot(x, average_d, 'b', label="Running Average DQN")
    plt.plot(x, average_r, 'r', label="Running Average Random")
    plt.legend()
    plt.grid(True, axis='y')
    plt.draw()
    plt.pause(0.001)


def savefig():
    plt.savefig("graph.png")

def plot_radiant(x, y):
    plt.plot(x, y, '--or', label="Against DQN")


def plot_dire(x, y):
    plt.plot(x, y, '--xb', label="Against Random")

def plot_trendline(x):
    y = [50 for _ in x]
    plt.plot(x, y, 'k', label='Baseline (50%)')