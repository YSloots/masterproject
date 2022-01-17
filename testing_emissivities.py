import numpy as np
import matplotlib.pyplot as plt

path = 'figures/'
#%%

def calc_emissivity_alphaDep(v, alpha):
	return v**2 + Ecre**2

def calc_emissivity_energyDep(v, Ecre):
	return v**2 + Ecre**2

#%%

v     = np.linspace(-2,2,4)
Ecre  = np.linspace(0,4,4)
alpha = np.linspace(-2, -4, 4)

vv, EE  = np.meshgrid(v, Ecre, sparse=True)
vv, aa  = np.meshgrid(v, alpha, sparse=True)
print(vv)
print(EE)
print(aa)
#J_E     = calc_emissivity_energyDep(vv,EE)
#J_alpha = calc_emissivity_alphaDep(vv,aa)

J_E = vv**2 + EE**2
J_alpha = J_E

plt.contourf(v, Ecre, J_E, 40, cmap='YlOrBr')
plt.savefig(path+'energyDepEmissivity.png')
plt.close('all')

plt.contourf(v,alpha,J_alpha, 40, cmap='YlOrBr')
plt.savefig(path+'alpohaDepEmissivitiy.png')
plt.close('all')

