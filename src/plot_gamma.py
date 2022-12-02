import numpy as np
import matplotlib.pyplot as plt

data = np.load("quan.npz", allow_pickle=True)
results = data['data'].reshape(-1)[0]

fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1,2,1)
for k, v in results.items():
    #print("{} {}".format(k, v))
    ax.plot(v[0], label=k)
ax.set_xlabel("bits")
ax.set_ylabel("accuracy")
ax.set_title("without retrain gamma")
plt.legend(ncol=3, loc="lower right")

ax = fig.add_subplot(1,2,2)
for k, v in results.items():
    #print("{} {}".format(k, v))
    ax.plot(v[1], label=k)
ax.set_xlabel("bits")
ax.set_title("with retrain gamma")
plt.legend(ncol=3, loc="lower right")
plt.tight_layout()
plt.savefig('gamma.pdf', format='pdf')
plt.show()
