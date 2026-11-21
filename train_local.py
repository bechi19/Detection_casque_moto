"""Fine-tuning local de YOLO11 pour la detection du port du casque (INFO-CV401)."""
import os
from roboflow import Roboflow
from ultralytics import YOLO

# A VERIFIER sur Roboflow : Versions > Download Dataset > Show download code
WORKSPACE = "guedaims-workspace"
PROJECT = "bike-helmet-detection-2vdjo-agpe6-1ijvt"
VERSION = 1

EPOCHS = 50
IMGSZ = 640
BATCH = 16      # baisser a 8 en cas d'erreur de memoire GPU
SEED = 0


def f1(p, r):
    return 2 * p * r / (p + r + 1e-9)


def report(model, data_yaml, split):
    """Affiche precision, rappel, F1 et mAP globaux et par classe."""
    m = model.val(data=data_yaml, split=split, device=0, workers=0)
    print(f"\n=== {split} : P={m.box.mp:.3f} R={m.box.mr:.3f} "
          f"mAP50={m.box.map50:.3f} mAP50-95={m.box.map:.3f}")
    for i, p, r, ap50 in zip(m.box.ap_class_index, m.box.p, m.box.r, m.box.ap50):
        print(f"  {model.names[int(i)]:<20} P={p:.3f} R={r:.3f} "
              f"F1={f1(p, r):.3f} mAP50={ap50:.3f}")


def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Definissez ROBOFLOW_API_KEY (set ROBOFLOW_API_KEY=...)")

    rf = Roboflow(api_key=api_key)
    ds = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download("yolov11")
    data_yaml = f"{ds.location}/data.yaml"
    print("Dataset :", ds.location)

    model = YOLO("yolo11n.pt")
    res = model.train(data=data_yaml, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                      device=0, workers=0, seed=SEED,
                      project="runs_helmet", name="yolo11n_ft")

    best = YOLO(f"{res.save_dir}/weights/best.pt")
    print("Meilleurs poids :", f"{res.save_dir}/weights/best.pt")
    report(best, data_yaml, "val")
    try:
        report(best, data_yaml, "test")
    except Exception as e:  # pas de split test
        print("Pas de jeu de test :", e)


if __name__ == "__main__":  # obligatoire sous Windows
    main()