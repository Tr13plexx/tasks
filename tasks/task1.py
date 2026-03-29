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
        

