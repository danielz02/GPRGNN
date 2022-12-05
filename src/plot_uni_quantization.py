import numpy as np
import matplotlib.pyplot as plt

data = np.load("quan.npz", allow_pickle=True)
results = data['data'].reshape(-1)[0]

fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1,2,1)
for k, v in results.items():
    #print("{} {}".format(k, v))
    ax.plot([i+1 for i in range(len(v[0])-1)], np.array(v)[0][1:], label=k)
ax.set_xlabel("bits")
ax.set_ylabel("accuracy")
ax.set_title("without retrain gamma")
plt.legend(ncol=3, loc="lower right")

ax = fig.add_subplot(1,2,2)
for k, v in results.items():
    #print("{} {}".format(k, v))
    ax.plot([i+1 for i in range(len(v[0])-1)], np.array(v)[1][1:], label=k)
ax.set_xlabel("bits")
ax.set_title("with retrain gamma")
plt.legend(ncol=3, loc="lower right")
plt.tight_layout()
plt.savefig('figures/uni_quan_accuracy.pdf', format='pdf')
plt.show()

fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1,2,1)
for k, v in results.items():
    #print("{} {}".format(k, v))
    ax.plot([i+1 for i in range(len(v[0])-1)], np.array(v)[2][1:], label=k)
ax.set_xlabel("bits")
ax.set_ylabel("MSE")
ax.set_title("layer 1 weight")
plt.legend(ncol=3, loc="upper right")

ax = fig.add_subplot(1,2,2)
for k, v in results.items():
    #print("{} {}".format(k, v))
    ax.plot([i+1 for i in range(len(v[0])-1)], np.array(v)[4][1:], label=k)
ax.set_xlabel("bits")
ax.set_title("layer 2 weight")
plt.legend(ncol=3, loc="upper right")
plt.tight_layout()
plt.savefig('figures/uni_quan_mse_weight.pdf', format='pdf')
plt.show()

fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1,2,1)
for k, v in results.items():
    #print("{} {}".format(k, v))
    ax.plot([i+1 for i in range(len(v[0])-1)], np.array(v)[3][1:], label=k)
ax.set_xlabel("bits")
ax.set_ylabel("MSE")
ax.set_title("layer 1 bias")
plt.legend(ncol=3, loc="upper right")

ax = fig.add_subplot(1,2,2)
for k, v in results.items():
    #print("{} {}".format(k, v))
    ax.plot([i+1 for i in range(len(v[0])-1)], np.array(v)[5][1:], label=k)
ax.set_xlabel("bits")
ax.set_title("layer 2 bias")
plt.legend(ncol=3, loc="upper right")
plt.tight_layout()
plt.savefig('figures/uni_quan_mse_bias.pdf', format='pdf')
plt.show()