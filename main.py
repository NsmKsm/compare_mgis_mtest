# mfront --obuild --interface=generic MericCailletaudSingleCrystalViscoPlasticity.mfront 

import mgis.behaviour as mgisbv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os 

path_bv_lib=  Path('src/libBehaviour.so')
path_bv_test=Path('Test.txt')

if not path_bv_lib.exists():
    os.system("mfront --obuild --interface=generic MericCailletaudSingleCrystalViscoPlasticity.mfront")
else:
    pass

if not path_bv_test.exists():
    os.system("mtest Test.mtest")
else:
    pass


data_mtest= np.loadtxt('Test.txt') # The results of mtest are saved in Test.txt

h=mgisbv.Hypothesis.Tridimensional # Hypothesis 
b= mgisbv.load('src/libBehaviour.so','MericCailletaudSingleCrystalViscoPlasticity',h)

N_gauss= 3  # the numbre of gauss points 
m = mgisbv.MaterialDataManager(b, N_gauss) # for details see: https://comet-fenics.readthedocs.io/en/latest/demo/plasticity_mfront/plasticity_mfront.py.html 

Ndt= 200  # the number of time laps 
time= np.linspace(0,10,Ndt+1)

EZZ0= np.linspace(0,0.02,Ndt+1) # Gauss point 0
EZZ1= np.linspace(0,0.09,Ndt+1) # Gauss point 1
EZZ2= np.linspace(0,1.4,Ndt+1) # Gauss point 2


E = np.array([2.08e5, 1.04e5 , 3.16e5] ) # YoungModulus for points 0,1 and 2. 
nu = np.array([0.3,0.3,0.3] ) # PoissonRatio for points 0,1 and 2. 
mu = np.array([8e4 , 4e4, 12e4] ) # ShearModulus  


storage=mgisbv.MaterialStateManagerStorageMode.LocalStorage
for s in [m.s0, m.s1]:  # m.s0  is the material state at time = ti ;;; m.s1  is the material state at time = ti+1
    mgisbv.setMaterialProperty(s, "YoungModulus1" , E , storage)
    mgisbv.setMaterialProperty(s, "YoungModulus2" , E , storage)
    mgisbv.setMaterialProperty(s, "YoungModulus3" , E , storage)
    mgisbv.setMaterialProperty(s, "PoissonRatio12", nu, storage)
    mgisbv.setMaterialProperty(s, "PoissonRatio23", nu, storage)  # material field on all points
    mgisbv.setMaterialProperty(s, "PoissonRatio13", nu, storage)
    mgisbv.setMaterialProperty(s, "ShearModulus12", mu, storage)
    mgisbv.setMaterialProperty(s, "ShearModulus23", mu, storage)
    mgisbv.setMaterialProperty(s, "ShearModulus13", mu, storage)
    mgisbv.setExternalStateVariable(s, "Temperature", 330)


# for s in [m.s0, m.s1]:
#     mgisbv.setMaterialProperty(s, "YoungModulus1" , 2.08e5)
#     mgisbv.setMaterialProperty(s, "YoungModulus2" , 2.08e5)
#     mgisbv.setMaterialProperty(s, "YoungModulus3" , 2.08e5)
#     mgisbv.setMaterialProperty(s, "PoissonRatio12", 0.3)
#     mgisbv.setMaterialProperty(s, "PoissonRatio23", 0.3)      # constant material field on all points
#     mgisbv.setMaterialProperty(s, "PoissonRatio13", 0.3)
#     mgisbv.setMaterialProperty(s, "ShearModulus12", 8e4)
#     mgisbv.setMaterialProperty(s, "ShearModulus23", 8e4)
#     mgisbv.setMaterialProperty(s, "ShearModulus13", 8e4)
#     mgisbv.setExternalStateVariable(s, "Temperature", 330)

eps_tot=np.zeros((3,201,6)) #                 |
eps_tot[:,:,2]=np.array([EZZ0, EZZ0, EZZ2 ])# | the imposed deformation


sig0=m.s0.thermodynamic_forces # stress at t=0
intvar0=m.s0.internal_state_variables # internale variables at t=0




Mtest_ind=1 # the indexe of gauss point that corresponds to MTEST (for comparing)
eps_to0= eps_tot[Mtest_ind,0] # strain at t=0
data_mgs= np.array([np.concatenate((np.array([0]),eps_to0, sig0[Mtest_ind], intvar0[Mtest_ind]))]) # all data in time (strain, stress and internal variables) are stored here.

mit = mgisbv.IntegrationType.IntegrationWithoutTangentOperator # Integration methode

for tn in range(1,Ndt+1):
    print('time = ', time[tn])

    eps_to=eps_tot[:,tn] # totat strain at t=ti+1

    m.s1.gradients[:, :]= eps_to # updating totat deformation at t= ti+1

    dt = time[tn] - time[tn - 1] # time lap
    mgisbv.integrate(m, mit, dt, 0, m.n) # integration

    sig=m.s1.thermodynamic_forces # internale variables calculated at t=ti+1
    intvar=m.s1.internal_state_variables # internale variables at t=ti+1

    para=np.array([np.concatenate((np.array([time[tn]]),eps_to[Mtest_ind], sig[Mtest_ind], intvar[Mtest_ind]))])  
    data_mgs=np.concatenate((data_mgs, para )) # all data in time (strain, stress and internal variables) are stored here.
 
    mgisbv.update(m)   # the results, obtained at ti+1, become now at ti for next iteration (they are strored in m.s0) 


Nvar=9 # the index of variable

ylbs=['time','Exx','Eyy','Ezz','Exy','Exz','Eyz','Sxx','Syy','Szz','Sxy','Sxz','Syz','elast_Exx','elast_Eyy','elast_Ezz','elast_Exy','elast_Exz','elast_Eyz',\
      'EVS0','EVS1','EVS2','EVS3','EVS4','EVS5','EVS6','EVS7','EVS8','EVS9','EVS10','EVS11',
      'BS0','BS1','BS2','BS3','BS4','BS5','BS6','BS7','BS8','BS9','BS10','BS11'] # EVS = EquivalentViscoplasticSlip,, BS= BackStrain

plt.plot( time, data_mgs[:,Nvar],'r-',label='MGIS')
plt.plot(time,data_mtest[:,Nvar],'b--',label='MTEST')
plt.xlabel('time')
plt.ylabel(ylbs[Nvar])
plt.legend(loc="lower right")
plt.show()




