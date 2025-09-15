import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def gen_sig():
    T = 20.0 # ms
    delta = 5. # ms
    A = 100.0
    return gen_data(A,T,delta,0.25)

def gen_bg():
    T = 20.0 # ms
    delta = 5. # ms
    A = 0.0
    return gen_data(A,T,delta,0.25)

def gen_data(A,T,delta,noise):
    data = []
    for t in np.linspace(0,100,num=100): #ms
        h = A * np.sin(2*np.pi*t / T) * scipy.stats.norm.pdf(t,loc=50,scale=20) + np.random.normal(scale=noise)
        l = A * np.sin(2*np.pi*(t+delta) / T)* scipy.stats.norm.pdf(t,loc=50+delta,scale=20) + np.random.normal(scale=noise)
        data.append( [ t,h,l, h+l, h-l])

    return np.array(data)


sdata = gen_sig()
bdata = gen_bg()

plt.plot(sdata[:,0],sdata[:,1],label="H")
plt.plot(sdata[:,0],sdata[:,2],label="L")
plt.plot(sdata[:,0],sdata[:,1]+sdata[:,2],label="H+L",linestyle=":")
plt.plot(sdata[:,0],sdata[:,1]-sdata[:,2],label="H-L",linestyle=":")
plt.xlabel("Time [ms]")
plt.ylabel("Strain")
plt.legend()
plt.savefig("sigs.pdf")
plt.clf()

plt.plot(bdata[:,0],bdata[:,1],label="H")
plt.plot(bdata[:,0],bdata[:,2],label="L")
plt.plot(bdata[:,0],bdata[:,1]+bdata[:,1],label="H+L",linestyle=":")
plt.plot(bdata[:,0],bdata[:,1]-bdata[:,1],label="H-L",linestyle=":")
plt.legend()
plt.xlabel("Time [ms]")
plt.ylabel("Strain")
plt.savefig("bgs.pdf")
