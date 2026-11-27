import unittest

import numpy as np

from camera_evaluation import CameraEvaluation, correction_summary, word_edits, word_tokens
from visper_inference import model_input, mouth_images, spoken_frame_span


class SpeechWindowTest(unittest.TestCase):
    def tracked_face(self, opening):
        steps = len(opening)
        angles = np.arange(40) * 2 * np.pi / 40
        lips = np.empty((steps, 40, 2), dtype=np.float32)
        lips[:, :, 0] = 150 + 27 * np.cos(angles)
        lips[:, :, 1] = 145 + (8 + opening[:, None]) * np.sin(angles)
        eyes = np.broadcast_to(np.array([[100, 80], [200, 80]], dtype=np.float32), (steps, 2, 2))

        # Frame-dependent rigid motion should not count as deformation of the mouth.
        rotation = np.linspace(-0.2, 0.2, steps)
        transforms = np.stack(
            [np.cos(rotation), -np.sin(rotation), np.sin(rotation), np.cos(rotation)], axis=1,
        ).reshape(-1, 2, 2)
        scale = np.linspace(0.9, 1.1, steps)[:, None, None]
        translation = np.linspace(0, 50, steps)[:, None, None]
        return lips @ transforms * scale + translation, eyes @ transforms * scale + translation

    def test_rejects_rigid_head_movement_without_articulation(self):
        lips, eyes = self.tracked_face(np.zeros(130))
        self.assertIsNone(spoken_frame_span(lips, eyes, cue_index=12))

    def test_keeps_context_around_a_short_word(self):
        opening = np.zeros(130)
        opening[25:39] = 12 * np.sin(np.linspace(0, np.pi, 14))
        lips, eyes = self.tracked_face(opening)
        span = spoken_frame_span(lips, eyes, cue_index=12)

        self.assertLess(span.start, 25)
        self.assertGreater(span.stop, 39)
        self.assertEqual(span.stop - span.start, 50)

    def test_keeps_words_after_the_old_three_second_window(self):
        opening = np.zeros(138)
        for start in (23, 74, 108):
            opening[start:start + 12] = 12 * np.sin(np.linspace(0, np.pi, 12))
        lips, eyes = self.tracked_face(opening)
        span = spoken_frame_span(lips, eyes, cue_index=12)

        self.assertLess(span.start, 23)
        self.assertGreater(span.stop, 120)


class CameraEvaluationTest(unittest.TestCase):
    def test_scores_spelling_variants_and_empty_output(self):
        self.assertEqual(word_edits(word_tokens("أُريدُ الماء"), word_tokens("اريد الماء.")), 0)
        evaluation = CameraEvaluation()
        evaluation.record("كان", "انا")
        evaluation.record("أريد", "")
        self.assertEqual((evaluation.completed, evaluation.reference_words), (2, 2))
        self.assertEqual((evaluation.previous_errors, evaluation.current_errors), (1, 1))

    def test_personal_comparison_reports_direction_and_ties(self):
        evaluation = CameraEvaluation(personal=True)
        evaluation.record("", evaluation.prompt)
        self.assertIn("انخفض", evaluation.summary)
        evaluation.record(evaluation.prompt, "")
        self.assertIn("متساوٍ", evaluation.summary)
        evaluation.record(evaluation.prompt, "")
        self.assertIn("زاد", evaluation.summary)
        self.assertIn("1 إلى 0", correction_summary("أريد الماء", "أريد", "اريد الماء"))
        self.assertIn("بقي", correction_summary("أريد الماء", "أريد", "أريد"))

    def test_feedback_revises_the_spoken_reference_without_replacing_original_predictions(self):
        evaluation = CameraEvaluation(personal=True)
        evaluation.record("وصلت", "الآن")
        self.assertEqual((evaluation.previous_errors, evaluation.current_errors), (1, 0))
        evaluation.revise_reference(0, "وصلت")
        self.assertEqual((evaluation.previous_errors, evaluation.current_errors), (0, 1))
        self.assertEqual(evaluation.completed, 1)
        self.assertEqual(evaluation.observations[0].current, "الآن")


class VisperInputTest(unittest.TestCase):
    def test_uses_source_pixels_with_interpolated_alignment(self):
        anchors = np.array([[70, 80], [180, 80], [125, 125], [99, 157], [151, 157]], dtype=np.float32)
        frames = [np.full((256, 256), intensity, dtype=np.uint8) for intensity in (20, 105, 173)]
        times = np.array([0, 66_000_000, 132_000_000])
        targets = np.array([0, 40_000_000, 80_000_000, 120_000_000])

        crops = mouth_images(frames, times, [anchors, None, anchors], targets, anchors)

        self.assertEqual(crops.shape, (4, 96, 96))
        np.testing.assert_array_equal(crops[:, 48, 48], [20, 105, 105, 173])

    def test_rejects_a_clip_without_alignment(self):
        frame = np.zeros((256, 256), dtype=np.uint8)
        self.assertIsNone(mouth_images([frame], np.zeros(1), [None], np.zeros(1), None))

    def test_center_crops_and_normalizes_pixels_for_visper(self):
        crops = np.zeros((2, 96, 96), dtype=np.uint8)
        crops[:, 4:-4, 4:-4] = 255

        inputs = model_input(crops)

        self.assertEqual(tuple(inputs.shape), (2, 1, 88, 88))
        np.testing.assert_allclose(inputs.numpy(), (1.0 - 0.421) / 0.165, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
