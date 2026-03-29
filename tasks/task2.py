import cv2
import numpy as np
from typing import Callable,List,Tuple

class ImagePreprocessor:
    def __init__(self) -> None:
        self._steps: List[Tuple[str,Callable]] = []
    def add_step(self,name:str,fn:Callable):
        self._steps.append((name,fn))
        return self
    def run(self,image:np.ndarray) -> np.ndarray:
        result = image.copy()
        for name,fn in self._steps:
            try:
                result = fn(result)
            except Exception as e:
                raise RuntimeError(f'Ошибка в шаге: {name}:{e}') from e
            return result
    def summary(self) -> List[str]:
            return [name for name,_ in self._steps]

if __name__ == "__main__":
    image = np.zeros((800, 600, 3), dtype=np.uint8)

    preprocessor = (
        ImagePreprocessor()
        .add_step("resize", lambda img: cv2.resize(img, (640, 480)))
        .add_step("to_gray", lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        .add_step("blur", lambda img: cv2.GaussianBlur(img, (5, 5), 0))
    )

    result = preprocessor.run(image)
    print("Pipeline steps:", preprocessor.summary())
    print("Result shape:", result.shape)