import cv2
import numpy as np
import matplotlib.pyplot as plt

# Kinect
camera = 'kinect'
# distance = 41.8
# width = 65
# height = 56
# tan_width = (width/2)/41.8
# tan_height = (height/2)/41.8

# helios2
camera = 'helios'
# distance = 41.8
# width = 118
# height = 72
# tan_width = (width/2)/41.8
# tan_height = (height/2)/41.8

depth_img = cv2.imread('/media/jyh/Extreme SSD/unloading/depth_test/' + camera + '_'+'plane_32.2.png', cv2.IMREAD_ANYDEPTH)

numpy_depth = np.array(depth_img)
np.savetxt('/home/jyh/catkin_ws/src/ransac_comparison/data/depth.txt',numpy_depth, fmt = '%3d', delimiter=',', header='test')

np.savetxt('/home/jyh/catkin_ws/src/ransac_comparison/data/depth_t.txt',numpy_depth.T, fmt = '%3d', delimiter=',', header='test')


box_size = [38, 48]

# Kinect
# img_size = [576, 640]

# Helios
img_size = [480, 640]

wx, hy = numpy_depth.shape
XB = np.arange(wx)
YB = np.arange(hy)
X,Y = np.meshgrid(XB,YB)
Z = numpy_depth.T
# plt.imshow(Z,interpolation='none')
# plt.show()

# X, Y = np.meshgrid(xx, yy)
# Z = np.asarray(zz, dtype=np.uint8).reshape((hy, wx))
# # Z = f(X, Y)

# Plotting 3D graph
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', \
                        edgecolor='none', antialiased=True, \
                        linewidth=0)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_zlim(0, 30)
ax.set_title('3D surface plot', fontsize=20)

# set view angles to get better plot
ax.azim = 70   # z rotation (default=270)
ax.elev = 50     # x rotation (default=0)
ax.dist = 10    # zoom (define perspective)

fig.colorbar(surf, shrink=0.5, aspect=15, pad = -0.05)
plt.tight_layout()
plt.savefig('/media/jyh/Extreme SSD/unloading/depth_test/' + camera + '_'+'plane_32.2_3d.png')
plt.show()
