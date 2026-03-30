import unittest
import numpy as np
import cv2
from task2 import ImagePreprocessor


class TestImagePreprocessor(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.ones((100, 100, 3), dtype=np.uint8) * 255

    def test_pipeline_execution(self):
        preprocessor = (
            ImagePreprocessor()
            .add_step("invert", lambda img: 255 - img)
            .add_step("resize", lambda img: cv2.resize(img, (100, 100,3)))
        )
        preprocessor = ImagePreprocessor().add_step("invert", lambda img: 255 - img)
        result = preprocessor.run(self.image)
        self.assertEqual(result.shape, (100, 100, 3))

    def test_summary(self):
        preprocessor = (
            ImagePreprocessor()
            .add_step("step1", lambda img: img)
            .add_step("step2", lambda img: img)
        )
        self.assertEqual(preprocessor.summary(), ["step1", "step2"])

    def test_error_handling(self):
        def fail_fn(img):
            raise ValueError("fail")

        preprocessor = ImagePreprocessor().add_step("fail_step", fail_fn)
        with self.assertRaises(RuntimeError) as context:
            preprocessor.run(self.image)
        self.assertIn("fail_step", str(context.exception))
        self.assertIn("fail", str(context.exception))


if __name__ == "__main__":
    unittest.main()