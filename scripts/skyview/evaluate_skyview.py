import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from app.inference.skyview_inference import predict_skyview

TEST_DIR = Path("data/processed/SkyView/test")

y_true = []
y_pred = []

for class_dir in sorted(TEST_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    actual_class = class_dir.name

    for image_path in class_dir.glob("*.jpg"):
        try:
            image = Image.open(image_path).convert("RGB")
            result = predict_skyview(image)

            y_true.append(actual_class)
            y_pred.append(result["predicted_class"])

        except Exception as e:
            print(f"Skipped {image_path}: {e}")

print("\nTotal evaluated:", len(y_true))

print("\nTEST ACCURACY:")
print(f"{accuracy_score(y_true, y_pred) * 100:.2f}%")

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred, zero_division=0))

print("\nCONFUSION MATRIX:")
classes = sorted(set(y_true) | set(y_pred))
print("Classes:", classes)
print(confusion_matrix(y_true, y_pred, labels=classes))
