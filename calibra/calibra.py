import numpy as np
import cv2 as cv
import glob

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,6,0)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.jpg')

for fname in images:
    img = cv.imread(fname)
    b = cv.cvtColor(img, cv.COLOR_BGR2LAB)[:, :, 2]
    b = cv.resize(b, None, fx=0.5, fy=0.5)
    b = cv.GaussianBlur(b, (9, 9), 0)
    b = cv.normalize(b, None, 0, 255, cv.NORM_MINMAX)

    ret, corners = cv.findChessboardCornersSB(b, (7,7), cv.CALIB_CB_EXHAUSTIVE)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners * 2)          # volta para 2560x1440

        cv.drawChessboardCorners(b, (7,7), corners, ret)   # desenha na imagem pequena
        cv.imshow('img', b)
        cv.waitKey(1000)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
print("Imagens usadas:", len(objpoints), "| erro (px):", ret)

print("Camera matrix : \n")
print(mtx)
img = cv.imread('9.jpg')   # uma foto em que o tabuleiro foi detectado
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image (rw, rh para nao sobrescrever w, h da imagem)
x, y, rw, rh = roi
dst = dst[y:y+rh, x:x+rw]
cv.imwrite('calibresult.png', dst)



# undistort
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
dst = dst[y:y+rh, x:x+rw]
cv.imwrite('calibresult2.png', dst)


cv.destroyAllWindows()

