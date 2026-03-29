import numpy as np
import cv2

def detect_by_color(
        image_path: str,
        lower_hsv: tuple[int,int,int],
        upper_hsv: tuple[int,int,int],
        min_area: int = 2000,
        output_path: str = 'result.jpg'
) -> list[dict]:
        image = cv2.imread(image_path)
        print(image is None)
        if image is None:
                raise ValueError("Изображение не загружено")
        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        lower_hsv = np.array(lower_hsv)
        upper_hsv = np.array(upper_hsv)
        color_mask = cv2.inRange(hsv,lower_hsv,upper_hsv)
        kernel = np.ones((5,5),np.uint8)
        mask1 = cv2.morphologyEx(color_mask,cv2.MORPH_OPEN,kernel)
        mask2 = cv2.morphologyEx(mask1,cv2.MORPH_DILATE,kernel)
        contours,_ = cv2.findContours(mask2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        print(len(contours))
        output = image.copy()
        result = []
        for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                        continue
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(
                        output,
                        f'Площадь:{int(area)}',
                        (x,y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0,255,0),
                        2
                )
                result.append({
                        "x":x,
                        "y":y,
                        "w":w,
                        "h":h,
                        "area":area
                })
        cv2.imwrite(output_path,output)
        cv2.imshow("mask", mask2)
        cv2.waitKey(0)
        return result

if __name__ == "__main__":
        result = detect_by_color(
                "input.jpeg",
                (0,0,0),
                (180,255,255)
        )
        print("Найденные обьекты:")
        for obj in result:
                print(obj)

