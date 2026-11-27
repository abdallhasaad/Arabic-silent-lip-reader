import unittest

import numpy as np

from camera_preprocessor import NUM_FRAMES, preprocess_window
from camera_preview import SAMPLE_PERIOD_NS, TimedLandmarkWindow
from inference import GeometryPredictor
from visper_inference import resample_points


TEST_DURATION_NS = 28 * SAMPLE_PERIOD_NS


class TimedLandmarkWindowTest(unittest.TestCase):
    def collect_window(self, fps, dropped=(), undetected=()):
        window = TimedLandmarkWindow(TEST_DURATION_NS)
        offsets = [round(index * 1_000_000_000 / fps) for index in range(100)]
        frame = np.zeros((8, 8), dtype=np.uint8)
        for index, offset in enumerate(offsets):
            if index in dropped:
                continue
            lips = eyes = None
            if index not in undetected:
                lips = np.full((40, 2), offset / 1_000_000_000)
                eyes = np.zeros((2, 2))
            if window.add(10_000_000_000 + offset, lips, eyes, frame):
                return window, offsets[:index + 1]
        self.fail("Window did not close")

    def test_tracks_a_fixed_duration_at_common_capture_rates(self):
        for fps in (12, 15, 25, 29.97, 30, 60):
            with self.subTest(fps=fps):
                window, _ = self.collect_window(fps)
                self.assertTrue(window.has_camera_coverage())
                self.assertGreaterEqual(window.elapsed_ns, TEST_DURATION_NS)
                self.assertLess(window.elapsed_ns - TEST_DURATION_NS, 1_000_000_000 / fps)

    def test_resamples_slow_input_without_stretching_the_word_window(self):
        for fps, dropped in ((12, ()), (15, (8,))):
            with self.subTest(fps=fps):
                window, offsets = self.collect_window(fps, dropped=dropped)
                self.assertLess(len(window.samples), 22)
                self.assertTrue(window.has_camera_coverage())
                times_ns = [offset for index, offset in enumerate(offsets) if index not in dropped]
                lips = resample_points(
                    times_ns, np.stack([sample.lips for sample in window.valid_landmarks()]),
                    np.arange(NUM_FRAMES) * SAMPLE_PERIOD_NS,
                )
                self.assertEqual(lips.shape, (29, 40, 2))
                np.testing.assert_allclose(lips[:, 0, 0], np.arange(29) * 0.04, atol=1e-7)

    def test_rejects_very_slow_or_interrupted_capture(self):
        slow, _ = self.collect_window(10)
        interrupted, _ = self.collect_window(30, dropped=range(10, 15))
        self.assertFalse(slow.has_camera_coverage())
        self.assertFalse(interrupted.has_camera_coverage())

    def test_checks_face_coverage_on_real_frames(self):
        window, _ = self.collect_window(15, undetected=(6, 7, 8))
        self.assertTrue(window.has_camera_coverage())
        self.assertFalse(window.valid_landmarks())


class LandmarkInputTest(unittest.TestCase):
    def test_geometry_only_matches_preview_and_counts_actual_detections(self):
        angles = np.arange(40) * 2 * np.pi / 40
        lips = np.column_stack([150 + 27 * np.cos(angles), 140 + 9 * np.sin(angles)])
        eyes = np.array([[100, 80], [200, 80]], dtype=np.float32)
        lips_sequence = [lips] * NUM_FRAMES
        eyes_sequence = [eyes] * NUM_FRAMES
        lips_sequence[9] = np.full((40, 2), np.nan)

        geometry = preprocess_window(None, lips_sequence, eyes_sequence)
        preview = preprocess_window(
            [np.zeros((240, 320, 3), dtype=np.uint8)] * NUM_FRAMES,
            lips_sequence, eyes_sequence,
        )
        self.assertIsNone(geometry.mouth_frames)
        self.assertEqual(geometry.detection_rate, 28 / 29)
        np.testing.assert_array_equal(geometry.landmarks, preview.landmarks)

        lips_sequence[:7] = [None] * 7
        self.assertIsNone(preprocess_window(None, lips_sequence, eyes_sequence))

    def test_predictor_rejects_pixel_coordinates(self):
        # Rejection happens during input validation, before a checkpoint is used.
        predictor = GeometryPredictor.__new__(GeometryPredictor)
        predictor.config = {"num_frames": NUM_FRAMES, "num_points": 40}
        with self.assertRaisesRegex(ValueError, "normalized"):
            predictor.predict(np.full((29, 40, 2), 50, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
