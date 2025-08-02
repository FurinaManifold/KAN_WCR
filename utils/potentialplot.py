import torch
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 24
x = torch.linspace(-1.2, 1.2, 100).unsqueeze(1)
pot = x**2-2*x**4+1.1*x**6
plt.figure(figsize=(8,7))
plt.plot(x.numpy(), pot,linewidth = 3)
plt.xticks(np.linspace(-1.2,1.2,3))
plt.yticks(np.linspace(0,0.6,3))
plt.xlim(-1.5,1.5)
plt.ylim(0.0,0.6)
plt.xlabel("x")
plt.ylabel("V(x)")
# plt.legend()
plt.show()
plt.savefig("save_files/potential.png")
plt.close()