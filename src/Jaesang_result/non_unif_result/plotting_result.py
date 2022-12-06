import numpy as np
import matplotlib.pyplot as plt

Dataset_list=['Cora','CiteSeer','PubMed','Computers','Photo','Chameleon','Squirrel','Texas','Cornell']

# for dataset in Dataset_list:
#     with open(dataset+'.txt') as f:
#         for line
experiment_result=np.zeros([len(Dataset_list),10,6])
i=0
for dataset in Dataset_list:
    experiment_result[i]=np.loadtxt(dataset+'.txt')
    
    i+=1
    
    
fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1,2,1)
for i in range(9):
    ax.plot(experiment_result[i,1:,0], label=Dataset_list[i])
ax.set_xlabel("bits")
ax.set_ylabel("accuracy")
ax.set_title("without retrain gamma")
plt.legend(ncol=3, loc="lower right")
    

ax = fig.add_subplot(1,2,2)
for i in range(9):
    ax.plot(experiment_result[i,1:,1], label=Dataset_list[i])
ax.set_xlabel("bits")
ax.set_title("with retrain gamma")
plt.legend(ncol=3, loc="lower right")
plt.tight_layout()
plt.savefig('figures/non_uni_quan_accuracy.pdf', format='pdf')
plt.show()

fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1,2,1)
for i in range(9):
    #print("{} {}".format(k, v))
    ax.plot(experiment_result[i,1:,2], label=Dataset_list[i])
ax.set_xlabel("bits")
ax.set_ylabel("MSE")
ax.set_title("layer 1 weight")
plt.legend(ncol=3, loc="upper right")

ax = fig.add_subplot(1,2,2)
for i in range(9):
    #print("{} {}".format(k, v))
    ax.plot(experiment_result[i,1:,3], label=Dataset_list[i])
ax.set_xlabel("bits")
ax.set_title("layer 2 weight")
plt.legend(ncol=3, loc="upper right")
plt.tight_layout()
plt.savefig('figures/non_uni_quan_mse_weight.pdf', format='pdf')
plt.show()

fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1,2,1)
for i in range(9):
    #print("{} {}".format(k, v))
    ax.plot(experiment_result[i,1:,4], label=Dataset_list[i])
ax.set_xlabel("bits")
ax.set_ylabel("MSE")
ax.set_title("layer 1 bias")
plt.legend(ncol=3, loc="upper right")

ax = fig.add_subplot(1,2,2)
for i in range(9):
    #print("{} {}".format(k, v))
    ax.plot(experiment_result[i,1:,5], label=Dataset_list[i])
ax.set_xlabel("bits")
ax.set_title("layer 2 bias")
plt.legend(ncol=3, loc="upper right")
plt.tight_layout()
plt.savefig('figures/non_uni_quan_mse_bias.pdf', format='pdf')
plt.show()