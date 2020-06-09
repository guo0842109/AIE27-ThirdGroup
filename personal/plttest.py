import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0, 100)
#新建figure对象
fig=plt.figure()
#新建子图1
ax1=fig.add_subplot(3,3,1)
ax1.plot(x, x)

ax1=fig.add_subplot(3,3,2)
ax1.plot(x, x)

#新建子图3
ax3=fig.add_subplot(3,3,3)
ax3.plot(x, x ** 2)
ax3.grid(color='r', linestyle='--', linewidth=1,alpha=0.3)
#新建子图4
ax4=fig.add_subplot(3,3,4)
ax4.plot(x, np.log(x))
plt.show()