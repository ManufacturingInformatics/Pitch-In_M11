import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
import warnings
# paths to different data sets
p100 = 'C:/Users/DB/Desktop/BEAM/Project Documentation/InSituAbsorb-autoemris-v100.csv'
p500 = 'C:/Users/DB/Desktop/BEAM/Project Documentation/InSituAbsorb-autoemris-v500.csv'
p1500 = 'C:/Users/DB/Desktop/BEAM/Project Documentation/InSituAbsorb-autoemris-v1500.csv'

# get data from file
# first column is power, second ccolumn is absorbtivity
b_v100 = np.genfromtxt(p100,delimiter=',')
b_v500 = np.genfromtxt(p500,delimiter=',')
b_v1500 = np.genfromtxt(p1500,delimiter=',')

# generate power data to test fitting with
p_test_data = np.linspace(0,500,b_v100.shape[0]*3)

# generalized sigmoid fn
def sigmoid(x,lower,upper,growth,v,Q):
    """ Generalized sigmoid function

        lower : lower asymptote
        upper : upper asymptote
        growth: growth factor
        v     : affects which asymptote the max growth occurs near
        Q     : related to y(0)
    """
    return lower + (upper-lower)/((1 + Q*np.exp(-growth*x))**(1.0/v))

def exponential(t,A,B,C,D):
    """ Generic Exponential Function

        y(t) = A + B*exp(C*t)

        Return result
    """
    from numpy import exp
    return A + B*exp((C*t)+D)

# fn for measuring error of gen_p0
def sumOfSquaredError(parameterTuple,xData,yData):
    warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
    # generate values
    val = sigmoid(xData, *parameterTuple)
    # returns sum sq error
    return np.sum((yData - val) ** 2.0)

def gen_p0(x,y):
    """ Estimate initial parameters for curve_fit using differential_evolution for data x,y"""
    maxX = np.max(x)
    maxY = np.max(y)
    minX = np.min(x)
    minY = np.min(y)

    # search areas for the parameters
    # in order of parameters for sigmoid fn
    parameterBounds = []
    # lower
    parameterBounds.append([minY,minY+0.1])
    # upper
    parameterBounds.append([maxY,maxY+0.1])
    # growth
    parameterBounds.append([0.1,200.0])
    # v
    parameterBounds.append([minY,maxY])
    # Q
    parameterBounds.append([minY,maxY])

    # "seed" the numpy random number generator for repeatable results
    # use differentiable evolution algorithm to generate initial parameter estimates for curve fit
    result = differential_evolution(sumOfSquaredError, parameterBounds,args=(x,y), seed=3,maxiter=3000)
    return result.x

def gen_p1(x,y):
    """ Estimate initial parameters for curve_fit using differential_evolution for data x,y"""
    maxX = np.max(x)
    maxY = np.max(y)
    minX = np.min(x)
    minY = np.min(y)

    # search areas for the parameters
    # in order of parameters for sigmoid fn
    parameterBounds = []
    # lower
    parameterBounds.append([minY,minY+0.1])
    # upper
    parameterBounds.append([maxY,maxY+0.1])
    # growth
    parameterBounds.append([0.001,10.0])
    # v
    parameterBounds.append([minY,maxY])
    # Q
    parameterBounds.append([minY,maxY])

    # "seed" the numpy random number generator for repeatable results
    # use differentiable evolution algorithm to generate initial parameter estimates for curve fit
    result = differential_evolution(sumOfSquaredError, parameterBounds,args=(x,y), seed=3,maxiter=5000)
    return result.x

def gen_p2(x,y):
    """ Estimate initial parameters for curve_fit using differential_evolution for data x,y"""
    maxX = np.max(x)
    maxY = np.max(y)
    minX = np.min(x)
    minY = np.min(y)

    # search areas for the parameters
    # in order of parameters for sigmoid fn
    parameterBounds = []
    # lower
    parameterBounds.append([minY,minY+0.1])
    # upper
    parameterBounds.append([maxY,maxY+0.1])
    # growth
    parameterBounds.append([0.0001,1.0])
    # v
    parameterBounds.append([minY,maxY])
    # Q
    parameterBounds.append([minY,maxY])

    # "seed" the numpy random number generator for repeatable results
    # use differentiable evolution algorithm to generate initial parameter estimates for curve fit
    result = differential_evolution(sumOfSquaredError, parameterBounds,args=(x,y), seed=3,maxiter=3000)
    return result.x

## fit sigmoids
# generate initial parameters
p0 = gen_p0(b_v100[:,0],b_v100[:,1])
popt_v100 = curve_fit(sigmoid,b_v100[:,0],b_v100[:,1],p0=p0,method='dogbox')[0]
print('v100:\n',p0)
print(popt_v100)
with open("sigmoid-v100-popt.txt","w") as f:
    for p in popt_v100:
        f.write(str(p)+',')

p0 = gen_p1(b_v500[:,0],b_v500[:,1])
popt_v500 = curve_fit(sigmoid,b_v500[:,0],b_v500[:,1],p0=p0,method='trf',maxfev=10000)[0]
print('\nv500:\n',p0)
print(popt_v500)
with open("sigmoid-v500-popt.txt","w") as f:
    for p in popt_v500:
        f.write(str(p)+',')

p0 = gen_p2(b_v1500[:,0],b_v1500[:,1])
popt_v1500 = curve_fit(sigmoid,b_v1500[:,0],b_v1500[:,1],p0=p0,method='trf',maxfev=10000)[0]
print('\nv1500:\n',p0)
print(popt_v1500)
with open("sigmoid-v1500-popt.txt","w") as f:
    for p in popt_v1500:
        f.write(str(p)+',')


## relationship between velocity and parameters
# veocity data in mm/s
vel = [100,500,1500]
# parse parameters
lower =  [popt_v100[0],popt_v500[0],popt_v1500[0]]
upper =  [popt_v100[1],popt_v500[1],popt_v1500[1]]
growth = [popt_v100[2],popt_v500[2],popt_v1500[2]]
v =      [popt_v100[3],popt_v500[3],popt_v1500[3]]
Q =      [popt_v100[4],popt_v500[4],popt_v1500[4]]

# set figure size to scaled version of default fig size
scale_factor = 2
fig_size = [f*scale_factor for f in plt.rcParams["figure.figsize"]]
## Sigmoid parameters
# label axis
f,((ax1,ax2,ax3),(ax4,ax5,_)) = plt.subplots(2,3,figsize=fig_size,tight_layout=True)
ax1.set(xlabel='Velocity (mms-1)',ylabel='Absorbtivity (Lower)',title='Lower Asymptote of Sigmoid Fitting')
ax2.set(xlabel='Velocity (mms-1)',ylabel='Absorbtivity (Upper)',title='Upper Asymptote of Sigmoid Fitting')
ax3.set(xlabel='Velocity (mms-1)',ylabel='Absorbtivity (Growth)',title='Growth of Sigmoid Fitting')
ax4.set(xlabel='Velocity (mms-1)',ylabel='Absorbtivity (v)',title='v-factor of Sigmoid Fitting')
ax5.set(xlabel='Velocity (mms-1)',ylabel='Absorbtivity (Q)',title='Q-factor of Sigmoid Fitting')

# plot data
ax1.plot(vel,lower,'bo-',markersize=10)
ax2.plot(vel,upper,'bo-',markersize=10)
ax3.plot(vel,growth,'bo-',markersize=10)
ax4.plot(vel,v,'bo-',markersize=10)
ax5.plot(vel,Q,'bo-',markersize=10)
# save figure
f.savefig("absorbtivity-sigmoid-params.png")


print("Fitting polynomial and writing coeffs to file")
## fit quadratics to data and evaluate for the current data
vel_new = np.linspace(min(vel),max(vel),50)
poly_coeffs_lower = poly.polyfit(vel,lower,2)
poly_fit_lower = poly.Polynomial(poly_coeffs_lower)
with open("absorb-poly-lower-params.txt","w") as f:
    for c in poly_coeffs_lower:
        f.write(str(c)+",")


poly_coeffs_upper = poly.polyfit(vel,upper,2)
poly_fit_upper = poly.Polynomial(poly_coeffs_upper)
with open("absorb-poly-upper-params.txt","w") as f:
    for c in poly_coeffs_upper:
        f.write(str(c)+",")


poly_coeffs_growth = poly.polyfit(vel,growth,2)
poly_fit_growth = poly.Polynomial(poly_coeffs_growth)
with open("absorb-poly-growth-params.txt","w") as f:
    for c in poly_coeffs_growth:
        f.write(str(c)+",")


poly_coeffs_Q = poly.polyfit(vel,Q,2)
poly_fit_Q = poly.Polynomial(poly_coeffs_Q)
with open("absorb-poly-Q-params.txt","w") as f:
    for c in poly_coeffs_Q:
        f.write(str(c)+",")

poly_coeffs_v = poly.polyfit(vel,v,2)
poly_fit_v = poly.Polynomial(poly_coeffs_v)
with open("absorb-poly-v-params.txt","w") as f:
    for c in poly_coeffs_v:
        f.write(str(c)+",")


## plot polyfit results
print("Plotting polyfit parameters")
f,((ax1,ax2,ax3),(ax4,ax5,_)) = plt.subplots(2,3,figsize=fig_size,tight_layout=True)
ax1.set(xlabel='Velocity (mms-1)',ylabel='Absorbtivity (Lower)',title='Lower Asymptote of Sigmoid Fitting')
ax2.set(xlabel='Velocity (mms-1)',ylabel='Absorbtivity (Upper)',title='Upper Asymptote of Sigmoid Fitting')
ax3.set(xlabel='Velocity (mms-1)',ylabel='Absorbtivity (Growth)',title='Growth of Sigmoid Fitting')
ax4.set(xlabel='Velocity (mms-1)',ylabel='Absorbtivity (v)',title='v-factor of Sigmoid Fitting')
ax5.set(xlabel='Velocity (mms-1)',ylabel='Absorbtivity (Q)',title='Q-factor of Sigmoid Fitting')

# plot data
ax1.plot(vel,lower,'bo',vel_new,poly_fit_lower(vel_new),'r-',markersize=10)
ax2.plot(vel,upper,'bo',vel_new,poly_fit_upper(vel_new),'r-',markersize=10)
ax3.plot(vel,growth,'bo',vel_new,poly_fit_growth(vel_new),'r-',markersize=10)
ax4.plot(vel,v,'bo',vel_new,poly_fit_v(vel_new),'r-',markersize=10)
ax5.plot(vel,Q,'bo',vel_new,poly_fit_Q(vel_new),'r-',markersize=10)
f.savefig("sigmoid-params-polyfit.png")

## plot fitting with extended velocity data set
print("Evaluating polynomial for extended data set")
v_ext = np.linspace(min(vel),2500,120)
f,((ax1,ax2,ax3),(ax4,ax5,_)) = plt.subplots(2,3,figsize=fig_size,tight_layout=True)
ax1.set(xlabel='Velocity (mms-1)',ylabel='Absorbtivity (Lower)',title='Lower Asymptote of Sigmoid Fitting (Ext)')
ax2.set(xlabel='Velocity (mms-1)',ylabel='Absorbtivity (Upper)',title='Upper Asymptote of Sigmoid Fitting (ext)')
ax3.set(xlabel='Velocity (mms-1)',ylabel='Absorbtivity (Growth)',title='Growth of Sigmoid Fitting (Ext)')
ax4.set(xlabel='Velocity (mms-1)',ylabel='Absorbtivity (v)',title='v-factor of Sigmoid Fitting (Ext)')
ax5.set(xlabel='Velocity (mms-1)',ylabel='Absorbtivity (Q)',title='Q-factor of Sigmoid Fitting (Ext)')

# plot data
ax1.plot(vel,lower,'bo',v_ext,poly_fit_lower(v_ext),'r-',markersize=10)
ax2.plot(vel,upper,'bo',v_ext,poly_fit_upper(v_ext),'r-',markersize=10)
ax3.plot(vel,growth,'bo',v_ext,poly_fit_growth(v_ext),'r-',markersize=10)
ax4.plot(vel,v,'bo',v_ext,poly_fit_v(v_ext),'r-',markersize=10)
ax5.plot(vel,Q,'bo',v_ext,poly_fit_Q(v_ext),'r-',markersize=10)
f.savefig("sigmoid-params-polyfit-vext.png")

## plotting sigmoid for extended range of velocity
print("Evaluating for different velocities")
vel_step = np.linspace(0,max(v_ext),10)
# power range
power_range = np.linspace(0,np.max(b_v1500[:,0])*2.0,b_v1500[:,0].shape[0])
f,ax = plt.subplots(1,1,tight_layout=True)
ax.set(xlabel="Power (W)",ylabel="Absorbtivity",title="Absorbtivity for Velocities =[{0:.2f},{1:.2f}]".format(np.min(vel_step),np.max(vel_step)),
                                                                                                                        yscale="log")
# for each velocity in 
for vs in vel_step:
    # calculate absorbtivity for the power range
    # evaluate parameters for specified velocity
    beta = sigmoid(power_range,poly_fit_lower(vs),poly_fit_upper(vs),poly_fit_growth(vs),poly_fit_v(vs),poly_fit_Q(vs))
    # add to plot
    ax.plot(power_range,beta,label="v={:.2f}mms-1".format(vs))
# add legend 
ax.legend()
f.savefig("absorbtivity-power-v-range.png")


#plt.plot(b_v100[:,0],b_v100[:,1],'bo',p_test_data,sigmoid(p_test_data,*popt_v100),'m^')
#ax_1 = plt.gca()
#ax_1.set(xlabel='Power (W)',ylabel='Absorbtivity',title='Absorbtivity Profile for v=100mm.s-1')
#plt.gcf().savefig("sigmoidfit-absorb-v100.png")

#plt.figure()
#plt.plot(b_v500[:,0],b_v500[:,1],'bo',p_test_data,sigmoid(p_test_data,*popt_v500),'m^')
#ax_2 = plt.gca()
#ax_2.set(xlabel='Power (W)',ylabel='Absorbtivity',title='Absorbtivity Profile for v=500mm.s-1')
#plt.gcf().savefig("sigmoidfit-absorb-v500.png")

#plt.figure()
#plt.plot(b_v1500[:,0],b_v1500[:,1],'bo',p_test_data,sigmoid(p_test_data,*popt_v1500),'m^')
#ax_3 = plt.gca()
#ax_3.set(xlabel='Power (W)',ylabel='Absorbtivity',title='Absorbtivity Profile for v=1500mm.s-1')
#plt.gcf().savefig("sigmoidfit-absorb-v1500.png")

print("Finished")
plt.show()