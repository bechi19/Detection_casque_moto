# Detection du port du casque chez les motocyclistes (INFO-CV401)

Projet de Computer Vision : detecter les motocyclistes, verifier s'ils portent un
casque et afficher un bilan (motards, avec casque, sans casque) sur une video.

## Installation
Python 3.12 recommande (les versions tres recentes manquent de paquets precompiles).

```
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```
Avec une carte NVIDIA, installer PyTorch avec CUDA depuis https://pytorch.org/get-started/locally/

## Lancer la demonstration (1 commande)
```
python live_counter.py --source VOTRE_VIDEO.mp4
```
`--source 0` utilise la webcam. Touches : `q` quitter, `p` pause.
Options : `--model`, `--conf`, `--imgsz`, `--min-frames`, `--save sortie.mp4`.

Les videos de test ne sont pas incluses (droits d'auteur). Ajoutez la votre.

## Entrainement
```
set ROBOFLOW_API_KEY=votre_cle
python train_local.py
```
Fine-tuning de YOLO11 sur le dataset Bike Helmet Detection (Roboflow Universe).
Verifier le nom du projet et la version dans `train_local.py`.
**Ne jamais publier la cle API.**

## Contenu
- `live_counter.py` : video en direct, suivi ByteTrack, compteur
- `train_local.py` : telechargement du dataset, entrainement, metriques
- `demo_frames.py` : images annotees (vert = casque, rouge = sans casque)
- `helmet_yolo11n_best.pt` : poids du modele YOLO11n entraine

## Resultats (jeu de test, 64 images, YOLO11n, 50 epoques)
| Classe | Precision | Rappel | F1 | mAP50 |
|---|---|---|---|---|
| With Helmet | 0.836 | 0.901 | 0.867 | 0.934 |
| Without Helmet | 0.841 | 0.587 | 0.692 | 0.727 |
| Global | 0.838 | 0.744 | - | 0.830 |

## Limites connues
- Jeu de test tres petit : les metriques varient beaucoup.
- Le rappel sur « Without Helmet » est faible (0.587).
- Le compteur compte des identifiants de suivi, pas des personnes : il sur-compte
  quand le suivi perd un motard. Les valeurs de `--conf` et `--min-frames` changent
  fortement le resultat.
- Le modele rate les petits motards et les scenes denses ; faux « casque » observe
  sur une webcam en interieur (ecart de domaine).

## Licences et sources
- Dataset : Bike Helmet Detection (Roboflow Universe) ; **verifier la licence**.
- Poids et bibliotheque Ultralytics : AGPL-3.0.

## Equipe
Noms et contributions : a completer.
