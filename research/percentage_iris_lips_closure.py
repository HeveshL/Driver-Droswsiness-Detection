import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from shapely.geometry import Polygon
import time

# Video Stream
cap = cv2.VideoCapture(0)

showPlots = False # may be CPU intensive

# Drawing Utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
drawSpec= mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
mp_face_mesh = mp.solutions.face_mesh

# Initializing Plots
if showPlots:
    fig = plt.figure("Realtime plots",figsize=(5,10))
    ax11 = plt.subplot(4,2,1, projection='3d')
    ax12 = plt.subplot(4,2,2, projection='3d')
    ax21 = plt.subplot(4,2,3)
    ax22 = plt.subplot(4,2,4)
    ax31 = plt.subplot(4,2,5)
    ax32 = plt.subplot(4,2,6)
    ax41 = plt.subplot(4,2,7, projection='3d')
    ax42 = plt.subplot(4,2,8)

    ax11.title.set_text("3D Left Eye Plot")
    ax12.title.set_text("3D Right Eye Plot")
    ax21.title.set_text("2D Projected Left Eye Plot")
    ax22.title.set_text("2D Projected Right Eye Plot")
    ax31.title.set_text("Left Eye Visibility Graph")
    ax32.title.set_text("Right Eye Visibility Graph")
    ax41.title.set_text("3D Lips Plot")
    ax42.title.set_text("Lips Ratio Graph")

    ax11.view_init(elev=10, azim=10)
    ax12.view_init(elev=10, azim=10)
    ax41.view_init(elev=170, azim=0)
    
    plt.subplots_adjust(
        left=0.1,
        bottom=0.1, 
        right=0.9, 
        top=0.9, 
        wspace=0.4, 
        hspace=0.4)

# Indices of landmarks
rIrisIdx = [469, 470, 471, 472]
lIrisIdx = [474, 475, 476, 477]
rEyeIdx = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
lEyeIdx = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 263]
lipsIdx = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]


# Queues for Visibility Percentage Plot
lQueue = []
rQueue = []
lipsQueue = []

lastUpdate = time.time()
updateFreq = 0.1 # in secs

# Blink State Variables
blinks = 0
isBlinking = False
blinkThreshold = 30 # %
blinkTimes = []

# Yawn State Variables
yawns = 0
isYawning = False
yawnStartTime = None
yawnThreshold = 1
yawnDuration = 1 # in secs

def getNormal(points):
    '''Given 3D points, returns the 3D normal vector'''

    b = np.array(points[0])
    r = np.array(points[1])
    s = np.array(points[2])
    qr = r - b
    qs = s - b
    normal = np.cross(qr,qs)
    normal = normal / sum(normal**2)**0.5
    return normal


def plotAndConnect(points,ax,color,is2d = False):
    '''Plots the points and connects them with lines'''

    if is2d:
        # Plotting 2D points
        for p in points:
            ax.scatter(*p, color=color)
        
        # Connecting 2D points
        for i in range(len(points)):
            ax.plot(
                [points[i][0],points[(i+1)%len(points)][0]],
                [points[i][1],points[(i+1)%len(points)][1]],
                color="black"
            )
    else:
        # Plotting 3D points
        for p in points:
            ax.scatter3D(xs=[p[0]], ys=[p[1]], zs=[p[2]], color=color)

        # Connecting 3D points
        for i in range(len(points)):
            ax.plot3D(
                xs=[points[i][0],points[(i+1)%len(points)][0]],
                ys=[points[i][1],points[(i+1)%len(points)][1]],
                zs=[points[i][2],points[(i+1)%len(points)][2]],
                color="black"
            )


def projection(point, normal):
    '''Projects a 3D point onto a 2D plane defined by the normal vector and the origin'''

    u = np.array(point)
    x = sum(normal**2)
    dot = np.dot(u,normal)
    alpha = -dot/x
    u = u + normal*alpha
    return list(u)


def getVisibility(iris, eye):
    '''Returns the visibility percentage of the iris'''

    try:
        irisPolygon = Polygon(iris)
        eyePolygon = Polygon(eye)
        intersection = irisPolygon.intersection(eyePolygon)
        return intersection.area / irisPolygon.area *100
    except:
        return 0


def convert3Dto2D(point,normal):
    '''Converts a 3D point to 2D'''

    # normalising normal vector
    modN = sum(map(lambda x: x**2,normal))**0.5
    normal = tuple(map(lambda x: x/ modN, normal))

    # calculating z rotation matrix
    modNxy = (normal[0]**2 + normal[1]**2)**0.5
    rz = [
        [normal[0]/modNxy, normal[1]/modNxy, 0],
        [-normal[1]/modNxy, normal[0]/modNxy, 0],
        [0, 0, 1]
    ]
    # calculating y rotation matrix
    normal1 = np.dot(rz,normal)
    ry = [
        [normal1[2], 0, -normal1[0]],
        [0, 1 ,0],
        [normal1[0], 0, normal1[2]]
    ]
    # calculating x rotation matrix (-90 degree)
    rx = [
        [0, 1],
        [-1, 0]
    ]
    # applying rotations
    return np.dot(rx,np.dot(ry,np.dot(rz,point))[:-1])


def calculateRatio(leftP, rightP, topP, bottomP):
    '''Calculates the ratio lips-vertical:lips-horizontal'''
    
    horizontal = sum((rightP-leftP)**2)**0.5
    vertical = sum((topP-bottomP)**2)**0.5
    return vertical/horizontal

def plotAll(points):
    '''Ploting 6 plots and returns the visibility percentage'''

    # Getting coordinates of landmarks
    lIrisCord = [(-points[idx].z, points[idx].x, -points[idx].y) for idx in lIrisIdx]
    rIrisCord = [(-points[idx].z, points[idx].x, -points[idx].y) for idx in rIrisIdx]
    lEyeCord  = [(-points[idx].z, points[idx].x, -points[idx].y) for idx in lEyeIdx]
    rEyeCord  = [(-points[idx].z, points[idx].x, -points[idx].y) for idx in rEyeIdx]
    lipsCord = [(-points[idx].z, points[idx].x, -points[idx].y) for idx in lipsIdx]
    # print(lipsCord)

    # Getting normal vectors
    lIrisNor = getNormal(lIrisCord)
    rIrisNor = getNormal(rIrisCord)

    # Projecting landmarks onto 2D plane
    lIrisPro = [projection(x,lIrisNor) for x in lIrisCord]
    rIrisPro = [projection(x,rIrisNor) for x in rIrisCord]
    lEyePro  = [projection(x,lIrisNor) for x in lEyeCord]
    rEyePro  = [projection(x,rIrisNor) for x in rEyeCord]

    # Converting 3D points to 2D
    lIrisPro2D = [convert3Dto2D(point,lIrisNor) for point in lIrisPro]
    rIrisPro2D = [convert3Dto2D(point,rIrisNor) for point in rIrisPro]
    lEyePro2D  = [convert3Dto2D(point,lIrisNor) for point in lEyePro]
    rEyePro2D  = [convert3Dto2D(point,rIrisNor) for point in rEyePro]

    # Lips 4 points
    leftP = np.array((-points[78].z, points[78].x, -points[78].y))
    rightP = np.array((-points[308].z, points[308].x, -points[308].y))
    topP = np.array((-points[13].z, points[13].x, -points[13].y))
    bottomP = np.array((-points[14].z, points[14].x, -points[14].y))

    # Getting visibility percentage
    lVisibility = getVisibility(lIrisPro2D,lEyePro2D)
    rVisibility = getVisibility(rIrisPro2D,rEyePro2D)
    
    # Getting lips ratio
    ratio = calculateRatio(leftP, rightP, topP, bottomP)

    currTime = time.time()
    global lastUpdate, lQueue, rQueue, lipsQueue

    # Checking if plot has to be updated
    if currTime - lastUpdate > updateFreq and showPlots:
        
        # Adding left iris visiblity to the lQueue
        lQueue.append(lVisibility)
        lQueue = lQueue[-100:]

        # Adding right iris visiblity to the rQueue
        rQueue.append(rVisibility)
        rQueue = rQueue[-100:]

        # Adding lips ratio to the lipsQueue
        lipsQueue.append(ratio)
        lipsQueue = lipsQueue[-100:]

        # Plotting left eye on 3D plot
        ax11.clear()
        plotAndConnect(lIrisCord,ax11,"blue")
        plotAndConnect(lEyeCord,ax11,"green")

        # Plotting right eye on 3D plot
        ax12.clear()
        plotAndConnect(rIrisCord,ax12,"blue")
        plotAndConnect(rEyeCord,ax12,"red")

        # Plotting left eye projection on 2D plot
        ax21.clear()
        plotAndConnect(lIrisPro2D,ax21,"blue",True)
        plotAndConnect(lEyePro2D,ax21,"green",True)

        # Plotting right eye projection on 2D plot
        ax22.clear()
        plotAndConnect(rIrisPro2D,ax22,"blue",True)
        plotAndConnect(rEyePro2D,ax22,"red",True)
        
        # Plotting left eye visibility on 2D plot
        ax31.clear()
        ax31.plot(lQueue,color="green")
        ax31.set_ylim(0,100)

        # Plotting right eye visibility on 2D plot
        ax32.clear()
        ax32.plot(rQueue,color="red")
        ax32.set_ylim(0,100)

        # Plotting lips on 3D plot
        ax41.clear()
        plotAndConnect(lipsCord,ax41,"green")

        # Plotting lips ratio
        ax42.clear()
        ax42.plot(lipsQueue,color="red")

        lastUpdate = currTime

        plt.pause(0.001)

    return lVisibility,rVisibility,ratio


# Main loop
with mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detecting landmarks
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style()
                )

                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style()
                )

            # Getting percentage of iris visibility
            lVisibility,rVisibility,ratio = plotAll(results.multi_face_landmarks[0].landmark)

            # If working with high FPS i.e. no plots are drawn
            if not showPlots:
                # updating blink status
                if lVisibility < blinkThreshold and rVisibility < blinkThreshold and not isBlinking:
                    isBlinking = True
                elif lVisibility >= blinkThreshold and rVisibility >= blinkThreshold and isBlinking:
                    isBlinking = False
                    blinks += 1
                    blinkTimes.append(time.time())
                
                curTime = time.time()
                i=0
                while i<len(blinkTimes):
                    if curTime - blinkTimes[i]>60:
                        blinkTimes.pop(i)
                    else:
                        break
                
                # updating yawn status
                if ratio > yawnThreshold and not isYawning:
                    yawnStartTime = time.time()
                    isYawning = True
                elif ratio <= yawnThreshold and isYawning:
                    yawnEndTime = time.time()
                    isYawning = False
                    if yawnEndTime - yawnStartTime >= yawnDuration:
                        yawns += 1

            # Adding percentage text to the image
            image = cv2.flip(image, 1)
            image = cv2.putText(image, f"Left: {int(lVisibility)}%", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            image = cv2.putText(image, f"Right: {int(rVisibility)}%", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            image = cv2.putText(image, f"Lips Ratio: {(ratio)}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            if not showPlots:
                image = cv2.putText(image, f"Blinks: {(blinks)}  Freq(Blinks/min): {(len(blinkTimes))}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
                image = cv2.putText(image, f"Yawns: {(yawns)}", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
              
        # Showing image
        cv2.imshow('Video', image)

        # Press ESC on keyboard to exit
        if cv2.waitKey(5) & 0xFF == 27:
            break

# When everything done, release the video capture object
cap.release()