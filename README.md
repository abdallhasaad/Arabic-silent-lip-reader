# Arabic Silent Lip Reader for Windows

The camera app uses the visual-only ViSpeR model for Arabic words and
phrases. It captures, aligns and decodes mouth images in memory.

## Install

Use 64-bit Python 3.13 with Tcl/Tk installed.
Create a fresh virtual environment on each computer. If a copied `.venv`
points to the previous computer, follow the
[Windows setup guide](ArabicLipGeometryNet_Windows_Setup_Commands.md)
to save it under a backup name and create a local environment.

    py -3.13 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Download these two local model files after cloning:

- Save the MediaPipe Face Landmarker to `assets/face_landmarker.task`. [Official model download](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)
- Save the visual-only ViSpeR checkpoint to `visper_vsr_base.pth` in the project directory. [Official checkpoint and checksum](https://huggingface.co/tiiuae/visper/blob/main/visper_vsr_base.pth)

`SHA256SUMS.json` lists the expected hashes of these files and the original
local geometry bundle. Personal feedback, recordings, and model checkpoints
are local files.

The Windows runtime uses the bundled visual encoder, tokenizer and beam
search from ViSpeR by Sanath Narayan, Yasser Abdelaziz Dahou Djilali,
Ankit Singh, Eustache Le Bihan and Hakim Hacid, with minor Python 3.13
compatibility changes. The upstream license is included in
`assets/visper/LICENSE`. The visual checkpoint is published for
non-commercial research. [ViSpeR source and citation](https://github.com/YasserdahouML/visper)

## Use the live camera

    .\.venv\Scripts\python.exe camera_preview.py

Press **ابدأ المحاولة** or **Space** and relax your lips until the green
**انطق الآن** cue appears. The countdown shows how long you have to speak.
Stop at the red **توقّف عن النطق** cue. Click **انتهيت من النطق** or press
**Space** to end early. Press **Esc** or **Q** to exit.

The app samples the camera using timestamps. It uses the short rest
period to estimate landmark noise, finds sustained lip movement after the
cue, and keeps surrounding frames for context. If it finds no clear
articulation, it asks for another attempt.

Use `--camera 1` for another camera or `--width 480 --height 360` if the
camera drops frames. Use `--visper-repo` and `--visper-checkpoint` for
different local model file paths.

## Personalize recognition from the live camera

After a result, click **تأكيد النتيجة** to confirm it or **تصحيح النتيجة** to enter what
you actually said. A captured utterance with an empty prediction also
accepts a correction. The app saves the example and adapts the visual
decoder and CTC head in the background, using some older examples alongside
the new one. Replay includes a prior recording of the same text when available
and rotates older examples according to when they were last used.
The status under the buttons confirms the save and reports whether this
specific example updated the weights. The session counters reset when the app
opens, and the saved totals reflect the local feedback history.
The next attempt uses the latest accepted update automatically. After a
correction, the app shows word errors before and after training on that same
camera attempt.

The app creates `camera_feedback.sqlite3` next to the application after
the first saved attempt. It stores the transcript, visual features from
the aligned mouth crops, and the personal weights. It restores them at
startup and resumes any saved example with pending training. Use
`--feedback-db` with a different path for another personal history.
Use `--no-personalization` to run the base model (including an explicitly
selected offline adapter, if supplied).

## Compare recognition accuracy

Click **اختبر أثر التدريب** and follow the eight Arabic word and phrase
prompts. Let each timed attempt end on its own. The app decodes the same
visual input with and without the personalized weights and reports each
version's word error rate on those new attempts. Confirming or correcting a
test result updates that attempt's reference while preserving both original
predictions for scoring. That feedback can train the model for later
attempts. Test scores stay in memory, and feedback that you submit is saved
as visual features and text.

With `--no-personalization`, **اختبر دقة الالتقاط** compares the original
three-second capture window and the motion-based window.

## Verify package

    .\.venv\Scripts\python.exe camera_preview.py --check-visper
    .\.venv\Scripts\python.exe -m unittest -v test_camera_preview test_visper_inference test_visper_training

The checkpoint check runs on synthetic frames and keeps the camera closed.
Recognition accuracy on actual speech requires a labeled camera session.

## Adapt ViSpeR on Arabic camera recordings

Install the training-only tokenizer, which uses the SentencePiece model from
the ViSpeR source checkout.

    .\.venv\Scripts\python.exe -m pip install -r requirements-training.txt

Create a UTF-8 JSONL manifest with one recording per line. Paths are relative
to the manifest unless absolute. Use speaker codes consistently across sessions
and assign each speaker to exactly one of `train`, `validation` or `test`.
The three splits need distinct speakers. Give the test and validation clips
manually checked transcripts and include everyday phrases and unfamiliar words.

```jsonl
{"clip":"mouth/001.npy","text":"وصلت إلى البيت الآن","speaker":"s01","split":"train"}
{"clip":"mouth/002.npy","text":"الكيس عند الباب","speaker":"s02","split":"validation"}
{"clip":"mouth/003.npy","text":"أين أضع الطرد","speaker":"s03","split":"test"}
```

A clip can be a `.npy` file with aligned 25 fps grayscale mouth crops as
`uint8[T,96,96]` or a short, trimmed video with a single visible speaker.
A video is used only for offline training and scoring. The live app keeps
using the camera. Obtain permission for using the recordings for training.

To collect clips with the live camera, make a UTF-8 text file containing one short
utterance per line. With the participant's consent, run the following.

    .\.venv\Scripts\python.exe camera_preview.py --collect dataset --prompts prompts.txt --speaker s01 --split train

The collection window displays the reference prompt. Each completed capture
writes an aligned mouth crop sequence and its text to `dataset`. The microphone
stays closed. Reopening collection for the same speaker, split and prompt file
resumes the remaining prompts. For validation and test, use different speakers
and different utterances where possible. Give speakers recording and collection
instructions appropriate to the cameras, lighting and dialects they use.

    .\.venv\Scripts\python.exe train_visper.py dataset\manifest.jsonl runs\arabic-01

Training caches the visual encoder features locally during the run and adjusts
low-rank weights in the decoder cross-attention and CTC head. The report records
WER and CER for the base and adapted models on the same held-out speakers. It
includes separate measurements for individual words, phrases and utterances
with words absent from training. Validation chooses the epoch. A separate
adapter file is saved when validation and test WER improve and test CER stays
at or below the base error. The strength of the comparison depends on the
number and variety of held-out examples. Each new run uses a new directory.

To try a passing adapter with the camera, use the optional adapter path.

    .\.venv\Scripts\python.exe camera_preview.py --visper-adapter runs\arabic-01\adapter.pt

    .\.venv\Scripts\python.exe -m unittest -v test_visper_training

## Geometry research model

The original research model uses geometric lip landmarks and a BiGRU for
Arabic visual speech recognition.

If you have the original 100-word geometry checkpoint, place `weights.pt`
in the project directory to use the `GeometryPredictor` Python API. Run
`python verify_model.py` to check that local checkpoint.
Its input is a `float32` array with shape `(29, 40, 2)` containing x/y
coordinates normalized within the aligned 96×96 mouth ROI. See
`preprocessing_spec.json` for the point order and `export_metadata.json`
for research validation metrics. Review the LRW-AR rights-holder terms
before redistribution or commercial use.
