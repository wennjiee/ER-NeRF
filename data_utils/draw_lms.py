import cv2
import face_alignment
import numpy as np
def cv_draw_landmark(img_ori, pts, box=None, color=(0, 0, 255), size=2):
    img = img_ori.copy()
    n = pts.shape[0]
    for i in range(n):
        cv2.circle(img, (int(round(pts[i, 0])), int(round(pts[i, 1]))), size, color, -1)
        # cv2.putText(img, str(i), (int(round(pts[i, 0])-3), int(round(pts[i, 1])-3)), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0,255,0), 1)
    return img

img_path = './data/wwj/ori_imgs/999.jpg'
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
input = cv2.imread(img_path)
preds = './data/wwj/ori_imgs/999.lms'
pred = np.loadtxt(preds)
output = cv_draw_landmark(input, pred)
cv2.imshow('result', output)
cv2.waitKey(0)
	
