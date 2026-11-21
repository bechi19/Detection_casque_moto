"""Montre la detection casque / sans casque sur quelques images d'une video.

Usage (cmd, venv actif) :
    python demo_frames.py MODELE.pt VIDEO.mp4 SORTIE.jpg [CONF]
Exemple :
    python demo_frames.py helmet_yolo11n_best.pt video_crowd.mp4 demo_crowd.jpg 0.25
Vert = avec casque, rouge = sans casque.
"""
import sys
import cv2
import numpy as np
from ultralytics import YOLO

N_FRAMES = 6      # nombre d'images extraites (grille 2 colonnes)
TILE_W = 640      # largeur de chaque vignette
IMGSZ = 1280


def is_no_helmet(name):
    n = name.lower().replace(" ", "_").replace("-", "_")
    return any(k in n for k in ["without", "no_helmet", "not_wear", "nohelmet"])


def annotate(model, frame, conf):
    """Dessine les boites (vert/rouge) et renvoie l'image + les comptes."""
    res = model.predict(frame, conf=conf, imgsz=IMGSZ, device=0, verbose=False)[0]
    with_h = without_h = 0
    for b in res.boxes:
        name = model.names[int(b.cls)]
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
        bad = is_no_helmet(name)
        color = (0, 0, 255) if bad else (0, 200, 0)   # BGR
        with_h += 0 if bad else 1
        without_h += 1 if bad else 0
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} {float(b.conf):.2f}", (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return frame, with_h, without_h


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    model_path, video_path, out_path = sys.argv[1:4]
    conf = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise SystemExit(f"Video illisible : {video_path}")
    idxs = np.linspace(0, total - 1, N_FRAMES + 2)[1:-1].astype(int)

    tiles = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        frame, w, wo = annotate(model, frame, conf)
        h = int(frame.shape[0] * TILE_W / frame.shape[1])
        tile = cv2.resize(frame, (TILE_W, h))
        cv2.rectangle(tile, (0, 0), (TILE_W, 26), (61, 33, 20), -1)
        cv2.putText(tile, f"frame {int(i)}  casque: {w}  sans casque: {wo}", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
        print(f"frame {int(i)}: {w} avec casque, {wo} sans casque")
    cap.release()

    if not tiles:
        raise SystemExit("Aucune image extraite.")
    if len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)]
    cv2.imwrite(out_path, np.vstack(rows))
    print("Image enregistree :", out_path)


if __name__ == "__main__":
    main()
