import numpy as np

from inference import GeometryPredictor


def main():

    predictor = GeometryPredictor()

    # Synthetic input for a technical smoke test only.
    # This is NOT a real video and NOT an accuracy evaluation.
    rng = np.random.default_rng(42)

    synthetic_landmarks = rng.uniform(
        0.15,
        0.85,
        size=(29, 40, 2)
    ).astype(np.float32)

    result = predictor.predict(
        synthetic_landmarks,
        top_k=5
    )

    print("Model loaded successfully.")
    print("Device:", predictor.device)
    print("\nSynthetic-input smoke test:")

    for item in result["top_k"]:

        print(
            item["word"],
            f"{item['probability']:.4f}"
        )

    print(
        "\nNOTE: Synthetic output does not measure "
        "real lip-reading accuracy."
    )


if __name__ == "__main__":
    main()
