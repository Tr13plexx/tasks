import numpy as np
import cv2

def detect_by_color(
        image_path: str,
        lower_hsv: tuple[int,int,int],
        upper_hsv: tuple[int,int,int],
        min_area: int = 500,
        output_path: str = 'result.jpg'
) -> list[dict]:
        image = cv2.imread(image_path)
        if image is None:
                raise ValueError("Изображение не загружено")
        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv,lower_hsv,upper_hsv)
        kernel = np.ones((5,5),np.uint8)
        mask1 = cv2.morphologyEx(color_mask,cv2.MORPH_OPEN,kernel)
        mask2 = cv2.morphologyEx(color_mask,cv2.MORPH_DILATE,kernel)
        

