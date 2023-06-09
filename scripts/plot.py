#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Get coefficient data

P = pd.read_csv('/home/jyh/catkin_ws/src/ransac_comparison/data/ransac_pcl_1.txt', sep="\t", header=None)
Y = pd.read_csv('/home/jyh/catkin_ws/src/ransac_comparison/data/ransac_pcl_2.txt', sep="\t", header=None)
Z = pd.read_csv('/home/jyh/catkin_ws/src/ransac_comparison/data/ransac_pcl_3.txt', sep="\t", header=None)

# X = Y[Y[3]<0.15]
# X = X[X[3]<0.45]

# W = Z[Z[3]<0.15]
# W = W[W[3]>0.35]
# X = pd.concat([X, W])

# Q = P[P[3]<0.25]
# print(X.size)
# Q = Q[Q[3]>0.35]
# X = pd.concat([X, Q])

X1 = P[P[3] < 0.25]
X2 = Y[Y[3] < 0.25]
X3 = Z[Z[3] < 0.25]

# X1 = X1[X1[3] < 0.35]
# X2 = X2[X2[3] < 0.35]
# X3 = X3[X3[3] < 0.35]


X = pd.concat([X1, X2])
X = pd.concat([X, X3])

# X.to_csv('/home/jyh/catkin_ws/src/ransac_comparison/data/ransac3_new_0.txt', sep = '\t')
# X = pd.read_csv('/media/jyh/Extreme SSD/230321/3/ransac6_2.txt', sep="\t", header=None)

# coefficient 추출
a_ = X.iloc[:,0]
b_ = X.iloc[:,1]
c_ = X.iloc[:,2]
d_ = X.iloc[:,3]
a = X.iloc[:,0].round(4)
b = X.iloc[:,1].round(4)
c = X.iloc[:,2].round(4)
d = X.iloc[:,3].round(4)

#a = a[a<0.5]

a_hist = a.value_counts()
b_hist = b.value_counts()
c_hist = c.value_counts()
d_hist = d.value_counts()

a_mean = np.mean(a)
a_std = np.std(a, ddof=1)
print("Mean :", a_mean.round(3), "St.dev :", a_std.round(3))
b_mean = np.mean(b)
b_std = np.std(b, ddof=1)
print("Mean :", b_mean.round(3), "St.dev :", b_std.round(3))
c_mean = np.mean(c)
c_std = np.std(c, ddof=1)
print("Mean :", c_mean.round(3), "St.dev :", c_std.round(3))
d_mean = np.mean(d)
d_std = np.std(d, ddof=1)
print("Mean :", d_mean.round(3), "St.dev :", d_std.round(3))

fig, axs = plt.subplots(2,2)
plt.xlim([-1.2, 1.2])
axs[0,0].hist(a, bins=a_hist.size)
#axs[0,0].axis([-1.2, 1.2, 0, 1000])
axs[0,0].set_title('a')
axs[0,1].hist(b, bins=b_hist.size)
#axs[0,1].axis([-1.2, 1.2, 0, 500])
axs[0,1].set_title('b')
axs[1,0].hist(c, bins=c_hist.size)
#axs[1,0].axis([-1.2, 1.2, 0, 500])
axs[1,0].set_title('c')
axs[1,1].hist(d, bins=d_hist.size)
#axs[1,1].axis([-1.2, 1.2, 0, 500])
axs[1,1].set_title('d')

print(X.size)

plt.show()