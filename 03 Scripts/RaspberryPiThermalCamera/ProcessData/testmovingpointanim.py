import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

data = np.random.randn(100)
print("Created {} data points".format(data.shape[0]))
f,ax = plt.subplots()
dp, = ax.plot([],[],'ro')
ax.plot(data,'b-')

def update(ff):
    dp.set_data(ff,data[ff])
    return (dp,)

ani = animation.FuncAnimation(f,update,data.shape[0],interval=1,blit=True)
ani.save('test-movingpoint.gif',writer='imagemagick',fps=60)
