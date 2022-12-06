import numpy as np
import matplotlib.pyplot as plt

Nrun=20

experiment_result=np.loadtxt('Sensitivity_with_20_run.txt')
mean=experiment_result[0,:]
std=experiment_result[1,:]

labels=['Original','Quantize lin1 weights','Quantize lin1 bias','Quantize lin2 weights','Quantize lin2 bias']
plt.errorbar(range(5),mean,yerr=1.96*std/np.sqrt(Nrun),fmt='o')
plt.xticks(range(5),labels,rotation='vertical')
plt.title('Accuracy with 95% confidence interval')
plt.savefig('Sensitivity_analysis', bbox_inches="tight")
plt.show()