import matplotlib.pyplot as plt

fig = plt.figure()
plt.ion()


def plot(x, radiant, dire):
    plt.cla()
    plot_radiant(x, radiant)
    plot_dire(x, dire)
    plot_trendline(x)
    plt.legend()
    plt.draw()
    plt.pause(1)

def savefig():
    plt.savefig("graph.png")

def plot_radiant(x, y):
    plt.plot(x, y, '--or', label="Radiant")


def plot_dire(x, y):
    plt.plot(x, y, '--xb', label="Dire")

def plot_trendline(x):
    y = [50 for _ in x]
    plt.plot(x, y, 'k', label='Baseline (50%)')