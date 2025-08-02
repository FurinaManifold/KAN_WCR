import torch
import numpy as np
import matplotlib.pyplot as plt
import json
plt.rcParams['font.size'] = 24

with open('1d_poly5_DNN_constdif_drifterror.json','r') as f:
    data = f.read()
    dnn = json.loads(data)

with open('1d_poly5_KAN2_2_30_constdif_drifterror.json','r') as f:
    data = f.read()
    kan = json.loads(data)

with open('chen_poly5_training_a_error.json','r') as f:
    data = f.read()
    chen = json.loads(data)

# x = torch.linspace(10, 1000).unsqueeze(1)
plt.figure(figsize=(8,7))
plt.plot(np.linspace(10,1000,100), dnn,linewidth = 3,label = 'FNN')
plt.plot(np.linspace(10,1000,100), kan,linewidth = 3,label = 'KAN')
plt.plot(np.linspace(10,10000,1000), chen,linewidth = 3,label = 'Chen')
plt.xticks(np.linspace(0,10000,3))
plt.yticks(np.linspace(0,0.5,6))
plt.xlim(0,10000)
plt.ylim(0.0,0.5)
plt.xlabel("epochs")
plt.ylabel("drift-error")
plt.legend()
plt.show()
plt.savefig("drift-error-compare_poly5.png")
plt.close()