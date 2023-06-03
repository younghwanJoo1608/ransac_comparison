import cv2
import numpy as np

# Kinect
camera = 'kinect'
distance = 41.8
width = 65
height = 56
tan_width = (width/2)/41.8
tan_height = (height/2)/41.8

# helios2
# camera = 'helios'
# distance = 41.8
# width = 118
# height = 72
# tan_width = (width/2)/41.8
# tan_height = (height/2)/41.8
box_distance = 500

depth_img = cv2.imread('/media/jyh/Extreme SSD/unloading/depth_test/' + camera + '_'+str(box_distance)+'.png', cv2.IMREAD_ANYDEPTH)

numpy_depth = np.array(depth_img)
np.savetxt('/home/jyh/catkin_ws/src/ransac_comparison/data/depth.txt',numpy_depth, fmt = '%3d', delimiter=',', header='test')

np.savetxt('/home/jyh/catkin_ws/src/ransac_comparison/data/depth_t.txt',numpy_depth.T, fmt = '%3d', delimiter=',', header='test')

box_size = [38, 48]

# Kinect
img_size = [576, 640]

# Helios
# img_size = [480, 640]

view_size = [box_distance * tan_height * 2, box_distance * tan_width * 2]
pixel_size = [view_size[0]/img_size[0], view_size[1]/img_size[1]]
box_pixel = [box_size[0]/pixel_size[0], box_size[1]/pixel_size[1]]

# print(tan_width)
# print(tan_height)
# print(pixel_size)
depth_exist = 0

row_pixel = []
col_pixel = []

# for i in range(int(img_size[0]*2/5), int(img_size[0]*3/5)):
#     for j in range(int(img_size[1]*2/5), int(img_size[1]*3/5)):
#         if (depth_img[i][j] >= 38) and (depth_img[i][j] <= 39):
#             row_pixel.append(i)
#             col_pixel.append(j)
#             depth_exist = depth_exist+1

pixels = 0
for i in range(291, 323):
    for j in range(295, 340):
        pixels = pixels+1
        if (depth_img[i][j] >= 1) and (depth_img[i][j] <= 128):
            row_pixel.append(i)
            col_pixel.append(j)
            depth_exist = depth_exist+1

#print(box_pixel[0] * box_pixel[1])

print (np.max(row_pixel))
print (np.min(row_pixel))
print (np.max(col_pixel))
print (np.min(col_pixel))

pixel2 = (np.max(row_pixel) - np.min(row_pixel))*(np.max(col_pixel) - np.min(col_pixel))
# print(pixel2)
print(depth_exist)
print(depth_exist/pixels * 100)

#print(depth_exist / (box_pixel[0] * box_pixel[1]) * 100)
# print(depth_exist / pixel2 * 100)

