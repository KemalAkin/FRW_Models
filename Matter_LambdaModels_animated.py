# -*- coding: utf-8 -*-
'''
==================================
= # Date: Aug 05, 2018           =   
= # Author: Kemal Akin           =
= # E - mail: akinkem@itu.edu.tr =
==================================
'''
    
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
# =================================
# = Some unit conversion factors: =
# =================================
H0 =  70               #Hubble constant H = 70 km/s/Mpc
Mpc = 3.085677581e19   # km
km =  1.0
Gyr = 3.1536e16        #second

H_0 = (H0 * km * Gyr) /  Mpc    #Here H_0 is in the units of 1/Gyr

plt.rc('text',fontsize = 11 ,usetex=True)  #In order to use LaTeX
plt.rc('font', family='serif') #In order to use Serif (mathced font with LaTeX)



# Universe containing Matter and Negative Cosmological constant:
def Matter_Negative_Lambda(Omega_0,t):
    # a_max: Scale factor at the time of maximum expansion
    a_max = (Omega_0 / (Omega_0 - 1))**(1./3) 
    
    # t_crunch: Lifetime of such a universe
    t_crunch = (2*np.pi / (3*H_0)) * (1/(np.sqrt(Omega_0 - 1)))
    
    # a: Evolution of the scale factor with time
    a = (np.sqrt(Omega_0 / (Omega_0 - 1))*np.sin((3./2)*H_0*t*np.sqrt(Omega_0 -1 )))**(2./3)
    
    # age: Present age of such a universe (t_0)
    age = (2. / (3*H_0*np.sqrt(Omega_0 - 1))) * np.arcsin(np.sqrt((Omega_0 - 1)/(Omega_0)))
    
    # Print properties of Matter - Negative Lambda model
    print ' For Matter = %.2f, Lambda = %.2f :'%(Omega_0, 1-Omega_0)
    print '-------------------------------------'
    print ' a_max= %.3f \n t_crunch= %.3f \n age= %.3f'%(a_max,t_crunch,age)
    
    # Return to array of scale factor, and various properties 
    return [a,a_max,t_crunch,age]

# Universe containing Matter and Positive Cosmological constant:
# ** Since in a flat universe Lambda = 1 - Matter, only one input (Omega_0)
# will be enough to describe ingredients **
def Matter_Positive_Lambda(Omega_0,a):
    # a_ml: Scale factor at the time of matter - lambda equality
    a_ml = (Omega_0 / (1 - Omega_0))**(1./3)
    
    # ** t: After integrating Friedmann equations, we deduce cosmic time as a function
    # of scale factor, i.e. t(a). In previous models we were able to revert
    # t(a) to a(t) analytically, however in this model t(a) is complicated
    # thus it is better to revert numerically. We will plot (t,a) as always. **
    t = ( (2 / (3*np.sqrt(1-Omega_0))) * np.log(  (a/a_ml)**(3/2) + np.sqrt( 1 + (a/a_ml)**(3))))/H_0
    
    # age: Present age of such a universe model
    age = (2. / (3*H_0*np.sqrt(1 - Omega_0 ))) * np.log( (np.sqrt(1 - Omega_0) + 1)/np.sqrt(Omega_0) )
    # Print properties of Matter - Positive Lambda model
    print ' For Matter = %.2f, Lambda = %.2f :'%(Omega_0, 1-Omega_0)
    print '-------------------------------------'
    print ' a_ml_equality= %.3f \n age= %.3f'%(a_ml,age)
    
    # Return to array of scale factor, and various properties 
    return [t, age, a_ml]

# =================
# = GENERATE DATA =
# =================

# Time scale 
t = np.linspace(0,100,1000)

# ** As decribed in lines 59-62 we will numerically invert t(a) to a(t) for 
# matter - positive cosmological constant model **
a = np.linspace(0,30,1000)

# ** Construct arrays to be plotted **:
# Negative Lambda (Flat FRW Universe with M = 1.1, L = -0.1)
# * 'nl' stands for negative lambda *
[a_nl,a_max_nl,t_c_nl,age_nl] = Matter_Negative_Lambda(1.1,t)

# Negative Lambda (Flat FRW Universe with M = 1.3, L = -0.3)
[a_nl1,a_max_nl1,t_c_nl1,age_nl1] = Matter_Negative_Lambda(1.3,t)

# Positive Lambda (Flat FRW Universe with M = 0.7, L = 0.3)
# * 'pl' stands for positive lambda *
[t_pl, age_pl, a_ml_pl] = Matter_Positive_Lambda(0.7,a)

# Positive Lambda (Flat FRW Universe with M = 0.3, L = 0.7) 
# * 'pl' stands for positive lambda *
# ** This model with M = 0.3, L= 0.7 is really a good approximation of 
# standard model of the cosmology: Lambda - Cold Dark Matter (L - CDM) **
[t_pl1, age_pl1, a_ml_pl1] = Matter_Positive_Lambda(0.3,a)

# =============
# = ANIMATION =
# =============

# Figure setup:
scale = 1920./1080
fig1 = plt.figure(figsize=(4.75*scale, 4.75), dpi=200, facecolor='white', edgecolor='w')
ax1 = fig1.add_subplot(111)

# Axis label, limits, title etc.
ax1.set_xlabel('$t-t_0$ (Gyr)')
ax1.set_ylabel('$a(t)$')

ax1.set_xlim(-20,100)
ax1.set_ylim(0,8)

plt.suptitle('Evolution of Matter + $\Lambda$ Universe Models',fontsize = 11)

'''
ax1.plot(t - age_nl,a_nl, color = 'crimson', label = '$\Omega_m = 1.1, \Omega_\Lambda = -0.1$')
ax1.plot(t- age_nl1,a_nl1, color = 'navy',   label = '$\Omega_m = 1.3, \Omega_\Lambda = -0.3$')

ax1.plot(t_pl- age_pl ,a, color = 'forestgreen', label = '$\Omega_m = 0.7, \Omega_\Lambda = 0.3$')
ax1.plot(t_pl1- age_pl1,a, color = 'orange', label = '$\Omega_m = 0.3, \Omega_\Lambda = 0.7$')
'''
plt.legend(['$\Omega_m = 1.1$','$\Omega_m = 1.3$','$\Omega_m= 0.7','$\Omega_m= 0.3$'],
           bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)

ax1.scatter(0,1,color = 'black',zorder = 10,s = 13)
plt.hlines(1,-20,0,color = 'black',linestyle = ':',linewidth = 0.5)
plt.vlines(0,0,1,color = 'black',linestyle = ':',linewidth = 0.5)

low = 2 
# Table:
ax1.plot([50,90],[6-low,6-low], color = 'black' , lw = 0.75)
ax1.plot([50,90],[5.9-low,5.9-low], color = 'black' , lw = 0.75)

ax1.plot([50,54],[5.7-low,5.7-low], color = 'crimson' , lw = 1)
ax1.plot([50,54],[5.3-low,5.3-low], color = 'navy' , lw = 1)
ax1.plot([50,54],[4.9-low,4.9-low], color = 'forestgreen' , lw = 1)
ax1.plot([50,54],[4.5-low,4.5-low], color = 'orange' , lw = 1)


ax1.text(57,6.1-low, '$\Omega_m$',fontsize = 10)
ax1.text(63,6.1-low, '$\Omega_\Lambda$',fontsize = 10)
ax1.text(70,6.1-low, '$t_0$',fontsize = 10)
ax1.text(76,6.1-low, '$t_c$',fontsize = 10)
ax1.text(82,6.1-low, '$a_{max}$',fontsize = 10)

ax1.text(-18,7.6, '$H_0 = 70$ km/s/Mpc')

ax1.text(57,5.6-low, '$1.1 \; \,  -0.1 \quad \;  9.0 \quad  92.6 \quad \quad 2.2$',fontsize = 9)
ax1.text(57,5.2-low, '$1.3 \; \,  -0.1 \quad \;  8.5 \quad  53.5 \quad \quad 1.6$',fontsize = 9)
ax1.text(57,4.8-low, '$0.7 \quad \; \;  0.3 \quad \;  10.5 \quad  - \quad \quad -$',fontsize = 9)
ax1.text(57,4.4-low, '$0.3 \quad \; \; 0.7 \quad \;  13.5 \quad  - \quad \quad -$',fontsize = 9)


ax1.text(57,6.5, r'$t_{crunch} = \displaystyle\frac{2\pi}{H_0} \frac{1}{\sqrt{\Omega_m - 1}} $')
ax1.text(57,5.1, r'$a_{max} = \displaystyle\left(\frac{\Omega_m}{\Omega_m - 1}\right)^{1/3}$')

ax1.scatter(t_c_nl/2 - age_nl, a_max_nl, color = 'crimson' , s = 15)
ax1.scatter(t_c_nl1/2 - age_nl1, a_max_nl1, color = 'navy' , s = 15)

# Horizontal lines for closed models:
n = 1500
t_d1 = np.linspace(-15,t_c_nl/2 ,n)
a_d1 = np.ones(n)*a_max_nl

t_d2 = np.linspace(-15,t_c_nl1/2 ,n)
a_d2 = np.ones(n)*a_max_nl1

# Vertical lines for closed models:
t_d3 = np.ones(n)*(t_c_nl/2)
a_d3 = np.linspace(0,a_max_nl,n)

t_d4 = np.ones(n)*(t_c_nl1/2)
a_d4 = np.linspace(0,a_max_nl1,n)


lines = []

for i in range(0,n):
    if i < len(t):
        # Negative Lambda 
        line1, = ax1.plot(t[:i] - age_nl ,a_nl[:i], color = 'crimson', label = '$\Omega_m = 1.1$')
        line2, = ax1.plot(t[:i] - age_nl1,a_nl1[:i], color = 'navy',   label = '$\Omega_m = 1.3$')
    
        # Positive Lambda
        line3, = ax1.plot(t_pl[:i] - age_pl ,a[:i], color = 'forestgreen', label = '$\Omega_m = 0.7$')
        line4, = ax1.plot(t_pl1[:i]- age_pl1,a[:i], color = 'orange', label = '$\Omega_m = 0.3$')
        
        
    
    # Dashed lines for a_max
    line5, = ax1.plot( t_d1[:i] - age_nl,a_d1[:i], color = 'crimson' , lw = 0.75,ls=':' ,zorder = 0)
    line6, = ax1.plot( t_d2[:i] - age_nl1,a_d2[:i], color = 'navy' , lw = 0.75,ls=':',zorder = 0)
    line7, = ax1.plot( t_d3[:i] - age_nl,a_d3[:i], color = 'crimson' , lw = 0.75,ls=':',zorder = 0)
    line8, = ax1.plot( t_d4[:i] - age_nl1,a_d4[:i], color = 'navy' , lw = 0.75,ls=':',zorder = 0)
    
    lines.append([line1,line2,line3,line4,line5,line6,line7,line8])
#plt.legend(['$\Omega_m = 1.1$','$\Omega_m = 1.3$','$\Omega_m= 0.7','$\Omega_m= 0.3$'],
#           bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)  

plt.legend(['$\Omega_m = 1.1$','$\Omega_m = 1.3$','$\Omega_m= 0.7$','$\Omega_m= 0.3$'],handles=[line1,line2,line3,line4],
           bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)

ani = animation.ArtistAnimation(fig1,lines,interval=10,blit=True)
#ani.save('Matter_LambdaModels_animated.mp4',extra_args=['-vcodec', 'libx264'])












