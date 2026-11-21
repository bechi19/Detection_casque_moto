"""Detection casque / sans casque en direct avec suivi et compteur.

Usage (cmd, venv actif) :
    python live_counter.py --source video_crowd.mp4
    python live_counter.py --source video_topdown.mp4 --conf 0.1
    python live_counter.py --source 0                  (webcam)
Options : --model, --conf, --imgsz, --min-frames, --save SORTIE.mp4
Touches : q = quitter, p = pause.

Regle de comptage : chaque motard (identifiant de suivi) est compte seulement
apres --min-frames images, et classe selon la majorite de ses etiquettes
(avec casque / sans casque). Cela limite les alertes dues aux clignotements.
"""
import argparse
import time
import cv2

GREEN, RED, GREY = (0, 200, 0), (0, 0, 255), (160, 160, 160)


def is_no_helmet(name):
    n = name.lower().replace(" ", "_").replace("-", "_")
    return any(k in n for k in ["without", "no_helmet", "not_wear", "nohelmet"])


class TrackStats:
    """Vote par identifiant de suivi : [images avec casque, images sans casque]."""

    def __init__(self, min_frames=5):
        self.min_frames = min_frames
        self.votes = {}

    def update(self, tid, no_helmet):
        v = self.votes.setdefault(tid, [0, 0])
        v[1 if no_helmet else 0] += 1

    def status(self, tid):
        """'confirmed_no', 'confirmed_yes' ou 'pending' (pas assez d'images)."""
        h, n = self.votes.get(tid, (0, 0))
        if h + n < self.min_frames:
            return "pending"
        return "confirmed_no" if n > h else "confirmed_yes"

    def totals(self):
        s = [self.status(t) for t in self.votes]
        return (s.count("confirmed_yes") + s.count("confirmed_no"),
                s.count("confirmed_yes"), s.count("confirmed_no"))


def draw_counter(frame, riders, with_h, without_h, fps):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (61, 33, 20), -1)
    txt = (f"Motards: {riders}   Avec casque: {with_h}   "
           f"Sans casque (suspects): {without_h}   FPS: {fps:.0f}")
    cv2.putText(frame, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="helmet_yolo11n_best.pt")
    ap.add_argument("--source", required=True, help="fichier video ou 0 pour la webcam")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--min-frames", type=int, default=5)
    ap.add_argument("--save", default=None, help="enregistrer la video annotee")
    args = ap.parse_args()

    from ultralytics import YOLO          # import ici : classes testables sans YOLO
    model = YOLO(args.model)
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Impossible d'ouvrir la source : {args.source}")

    writer = None
    if args.save:
        w, h = int(cap.get(3)), int(cap.get(4))
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                 cap.get(cv2.CAP_PROP_FPS) or 25, (w, h))

    stats, fps, paused = TrackStats(args.min_frames), 0.0, False
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.time()
            r = model.track(frame, persist=True, tracker="bytetrack.yaml",
                            conf=args.conf, imgsz=args.imgsz, device=0, verbose=False)[0]
            ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
            for box, tid in zip(r.boxes, ids):
                bad = is_no_helmet(model.names[int(box.cls)])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                if tid is None:
                    color, label = GREY, "?"
                else:
                    stats.update(tid, bad)
                    st = stats.status(tid)
                    color = GREY if st == "pending" else (RED if st == "confirmed_no" else GREEN)
                    label = f"#{tid} {'SANS casque' if st == 'confirmed_no' else 'casque' if st == 'confirmed_yes' else '...'}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(y1 - 4, 46)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            riders, with_h, without_h = stats.totals()
            fps = 0.9 * fps + 0.1 / max(time.time() - t0, 1e-6) if fps else 1 / max(time.time() - t0, 1e-6)
            draw_counter(frame, riders, with_h, without_h, fps)
            if writer:
                writer.write(frame)
        scale = min(1.0, 1280 / frame.shape[1])
        cv2.imshow("Detection casque (q: quitter, p: pause)",
                   cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            paused = not paused

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    riders, with_h, without_h = stats.totals()
    print(f"Bilan : {riders} motards, {with_h} avec casque, {without_h} sans casque (suspects)")


if __name__ == "__main__":
    main()
