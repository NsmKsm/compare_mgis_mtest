# mfront --obuild --interface=generic MericCailletaudSingleCrystalViscoPlasticity.mfront 
# import mgis
import mgis.behaviour as mgisbv
import numpy as np
import matplotlib.pyplot as plt

data_mtest= np.loadtxt('Test.txt')

h=mgisbv.Hypothesis.Tridimensional



b=mgisbv.load('src/libBehaviour.so','MericCailletaudSingleCrystalViscoPlasticity',h)
m = mgisbv.MaterialDataManager(b, 1)

# r = np.array([1,0,0,0,1,0,0,0,1], dtype=np.double)


Ndt= 200  # the number of time laps

time= np.linspace(0,10,Ndt+1)
EZZ= np.linspace(0,0.02,Ndt+1)

for s in [m.s0, m.s1]:
    mgisbv.setExternalStateVariable(s, "Temperature", 330)

eps_to0= np.array( [0,0,EZZ[0],0,0,0])
sig0=m.s0.thermodynamic_forces.flatten()
intvar0=m.s0.internal_state_variables.flatten()


data_mgs= np.array([np.concatenate((np.array([0]),eps_to0, sig0, intvar0))])
mit = mgisbv.IntegrationType.IntegrationWithoutTangentOperator

for tn in range(1,Ndt+1):
    print('time = ', time[tn])
    eps_to= np.array( [0,0,EZZ[tn],0,0,0])
    m.s1.gradients[:, :]= eps_to
    dt = time[tn] - time[tn - 1]
    mgisbv.integrate(m, mit, dt, 0, m.n)
    sig=m.s1.thermodynamic_forces.flatten()
    intvar=m.s1.internal_state_variables.flatten()

    para=np.array([np.concatenate((np.array([time[tn]]),eps_to, sig, intvar))])
    data_mgs=np.concatenate((data_mgs, para ))

    mgisbv.update(m)


Nvar=9
ylbs=['time','Exx','Eyy','Ezz','Exy','Exz','Eyz','Sxx','Syy','Szz','Sxy','Sxz','Syz','elast_Exx','elast_Eyy','elast_Ezz','elast_Exy','elast_Exz','elast_Eyz',\
      'EVS0','EVS1','EVS2','EVS3','EVS4','EVS5','EVS6','EVS7','EVS8','EVS9','EVS10','EVS11',
      'BS0','BS1','BS2','BS3','BS4','BS5','BS6','BS7','BS8','BS9','BS10','BS11'] # EVS = EquivalentViscoplasticSlip,, BS= BackStrain

plt.plot( time, data_mgs[:,Nvar],'r-',label='MGIS')
plt.plot(time,data_mtest[:,Nvar],'b--',label='MTEST')
plt.xlabel('time')
plt.ylabel(ylbs[Nvar])
plt.legend(loc="lower right")
plt.show()




