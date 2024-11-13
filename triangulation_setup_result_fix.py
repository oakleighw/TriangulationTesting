import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pandas as pd
from sklearn.metrics import root_mean_squared_error


cXs344 = [] #centroids to plot
cYs344 = []

cXs346 = [] #centroids to plot
cYs346 = []


labelsdir = r"labels"

fig, ax = plt.subplots(1,2,figsize=(10, 4))

#read labels in yolo format
for lab in sorted(os.listdir(labelsdir)):
    p = os.path.join(labelsdir,lab)
    f = open(p, "r")
    lines = f.readlines()
    #txt = np.loadtxt(p, delimiter=' ')
   
    for line in lines:
        lineS = line.split(" ")

        #denormalise centre of ball bearing point
        xcent = float(lineS[1]) * 1920 #width
        ycent = float(lineS[2]) * 1200 #height

        
        #print("xcent", xcent, "ycent", ycent)
        #append to arrays
        if lab[0:3] == "344":
            cXs344.append(xcent)
            cYs344.append(ycent)
            ax[0].scatter(xcent, ycent) #plot colours check that points are annotated correctly
            
        else:
            cXs346.append(xcent)
            cYs346.append(ycent)
            ax[1].scatter(xcent, ycent)

        
        
        
# fig, ax = plt.subplots(1,2,figsize=(10, 4))
# ax[0].set_xlim(0,1920)
# ax[0].set_ylim(0,1200)
# ax[0].scatter(cXs344, cYs344)
# ax[0].set_title("Cam 344")
# ax[1].set_xlim(0,1920)
# ax[1].set_ylim(0,1200)
# ax[1].scatter(cXs346, cYs346)
# ax[1].set_title("Cam 346")
# plt.show()



#Cam344
mtx344p = r"PATH\cmtx.yml"

#Cam346
mtx346p = r"PATH\cmtx.yml"


#load camera mtxs and distortion coefficients for both cameras
fs = cv.FileStorage(mtx344p, cv.FILE_STORAGE_READ)
mtx344 = np.array(fs.getNode("cameraMatrix0").mat())
dist344 = np.array(fs.getNode("distCoeffs").mat())


t = np.array(fs.getNode("tranVec").mat())
fun = np.array(fs.getNode("fundMat").mat())
r = np.array(fs.getNode("rotM").mat())
E = np.array(fs.getNode("essMat").mat())
#vals = cv.calibrationMatrixValues(mtx344, (1920,1200), 8.8,6.60)

#print(vals)

# vals = cv.calibrationMatrixValues(mtx344, (1920,1200), 4,4)
# print(vals)

fs = cv.FileStorage(mtx346p, cv.FILE_STORAGE_READ)
mtx346 = np.array(fs.getNode("cameraMatrix1").mat())
dist346 = np.array(fs.getNode("distCoeffs").mat())
# tran346 = np.array(fs.getNode("tranVec").mat())

# r346 = np.array(fs.getNode("rotM").mat())

#read in ground truth points
points344 = np.array([[x,y] for x,y in zip(cXs344,cYs344)],dtype="float64")
points346 = np.array([[x,y] for x,y in zip(cXs346,cYs346)],dtype="float64")


#projection matrix
# proj344 = np.dot(mtx344, np.hstack((np.eye(3), np.zeros((3, 1)))))
# proj346 = np.dot(mtx346, np.hstack((r, t)))  

R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(mtx344, dist344, mtx346, dist346,
                                              (1200,1920), r, t, flags = 0) #alpha=0,flags = 0 changes from conf review report




undistort_points344 = cv.undistortPoints(points344, mtx344, dist344, P = P1, R = R2).squeeze() #have to swap R's here

undistort_points346 = cv.undistortPoints(points346, mtx346, dist346, P = P2, R = R1).squeeze()


triangulated_points_4d_homogeneous = cv.triangulatePoints(P1, P2, undistort_points344.T, undistort_points346.T)

# Convert from homogeneous to 3D so you can plot with the correct perspective
triangulated_points_3d = triangulated_points_4d_homogeneous[:3] / triangulated_points_4d_homogeneous[3]
triangulated_points_3d = triangulated_points_3d.T

# Create the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') #as opposed to 3d



x_points = triangulated_points_3d[:, 0]
y_points = triangulated_points_3d[:, 1]
z_points = triangulated_points_3d[:, 2]
# cam344_position = R1.T



# Plot the points using scatter3D
ax.scatter(x_points[0:4], y_points[0:4], z_points[0:4] ,c='b', marker='o')  # Adjust color (c) and marker style as desired
ax.scatter(x_points[4:8], y_points[4:8], z_points[4:8] ,c='r', marker='o')  # Adjust color (c) and marker style as desired
ax.scatter(x_points[8:12], y_points[8:12], z_points[8:12] ,c='g', marker='o')  # Adjust color (c) and marker style as desired
ax.scatter(x_points[12:16], y_points[12:16], z_points[12:16] ,c='y', marker='o')  # Adjust color (c) and marker style as desired
ax.scatter(x_points[16:20], y_points[16:20], z_points[16:20] ,c='purple', marker='o')  # Adjust color (c) and marker style as desired
ax.scatter(x_points[20:24], y_points[20:24], z_points[20:24] ,c='orange', marker='o')  # Adjust color (c) and marker style as desired
ax.scatter(x_points[24:28], y_points[24:28], z_points[24:28] ,c='pink', marker='o')  # Adjust color (c) and marker style as desired
ax.scatter(x_points[28:], y_points[28:], z_points[28:] ,c='black', marker='o')  # Adjust color (c) and marker style as desired

cam344_position = (0,0,0)
# Cam 344 is the origin because of stereoCalibrate is relative
ax.scatter(*cam344_position, c='gray', marker='^', s=100, label='Camera 344')

# # cam346_position = -r.T @ P2[:, 3]
cam346_position = -R1.T @ t
ax.scatter(*cam346_position, c='olive', marker='^', s=100, label='Camera 346')

leg = ["5cm","10cm","15cm","20cm","25cm","30cm","35cm","Strawberry", "Cam1","Cam2",]
#leg = ["5cm","10cm","15cm","20cm","25cm","30cm","35cm","Strawberry"]
ax.legend(leg)
ax.set_aspect('equal')

# def euc(point1,point2):
#     return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))





# Set labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

ax.set_title('3D Plot from Triangulated Points')

plt.show()

fig = plt.figure()

# plt.scatter(x_points, z_points)
# plt.show()


#checking distances between points
def euc(point1,point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))


#for colleating distances (for error metric)
inners = []
outers = []

#create ground truth distances to compare (in millimetres)
inner_gt_base =  [50,25,25,50]
outer_gt_base = [50] * 4

inner_gt = []
outer_gt = []

for i in range(0, len(x_points), 4):
    #get 2 adjacent triangle patterns
    temp_l = [x_points[i:i+4],y_points[i:i+4],z_points[i:i+4]] #first 3 points (vertical)
    temp_l2 = [x_points[i+4:i+8],y_points[i+4:i+8],z_points[i+4:i+8]] #next 3 points (behind)

    # distances go 
    # 1.first vertice to 3rd (d1) 
    # 2. point splitting a triangle side (d1 & d2) 
    # 3. second vertex to third (d3)
    # 4. first vertex for current triangle and first vertex next triangle (d4)

    # 5. side-middle for current triangle to  side-middle of next triangle (d5)
    # 6. second vertex for current triangle and second vertex next triangle (d6)
    # 7. third vertex for current triangle and third vertex next triangle (d7)


    #within one pattern
    print(f"inter {leg[i//4]}")
    
    d1 = euc([temp_l[0][0],temp_l[1][0],temp_l[2][0]],[temp_l[0][3],temp_l[1][3],temp_l[2][3]])  
    d21 = euc([temp_l[0][0],temp_l[1][0],temp_l[2][0]],[temp_l[0][1],temp_l[1][1],temp_l[2][1]]) 
    d22 = euc([temp_l[0][1],temp_l[1][1],temp_l[2][1]],[temp_l[0][2],temp_l[1][2],temp_l[2][2]]) 
    d3 = euc([temp_l[0][2],temp_l[1][2],temp_l[2][2]],[temp_l[0][3],temp_l[1][3],temp_l[2][3]]) 


    print(d1,d21,d22,"\n",d3)

    inners.extend([d1,d21,d22,d3])
    inner_gt.extend(inner_gt_base)

    if (i >= (len(x_points)-8)): #don't include camera distance or strawberry distances
        break

    # pattern-to-pattern
    d4 = euc([temp_l[0][0],temp_l[1][0],temp_l[2][0]],[temp_l2[0][0],temp_l2[1][0],temp_l2[2][0]])
    d5 = euc([temp_l[0][1],temp_l[1][1],temp_l[2][1]],[temp_l2[0][1],temp_l2[1][1],temp_l2[2][1]])
    d6 = euc([temp_l[0][2],temp_l[1][2],temp_l[2][2]],[temp_l2[0][2],temp_l2[1][2],temp_l2[2][2]])
    d7 = euc([temp_l[0][3],temp_l[1][3],temp_l[2][3]],[temp_l2[0][3],temp_l2[1][3],temp_l2[2][3]])
    #np.sqrt(np.sum((np.array([x_points[0][1],y_points[0][1]]) - np.array(x_points[0][2],y_points[0][2]))**2))
    # print(i%3)
    

    print(f"outer {leg[i//4]} -> {leg[(i+4)//4]}", d4, d5, d6 , d7)

    
    outers.extend([d4,d5,d6,d7])
    outer_gt.extend(outer_gt_base)



#print metrics
inner_rmse = root_mean_squared_error(inner_gt,inners)
outer_rmse = root_mean_squared_error(outer_gt,outers)
inner_gt.extend(outer_gt)
inners.extend(outers)
total_rmse = root_mean_squared_error(inner_gt,inners)

print("Inner RMSE:" ,inner_rmse,"\n",
      "Outer RMSE: ", outer_rmse,"\n",
      "Total RMSE:", total_rmse)