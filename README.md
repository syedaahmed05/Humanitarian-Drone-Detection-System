# Laila
Hack Dearborn 5 Project where an AI system is developed to find people in crises such as earthquakes, floods, etc. 

AI-assisted search and rescue support for crises such as earthquakes and floods.

## Local setup (Git Bash on Windows)

```bash
python -m venv venv
source venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Add service credentials to `.env` as needed. Never commit `.env`; it is ignored by Git. After installing or changing dependencies, refresh the lock-style snapshot with `python -m pip freeze > requirements.txt`.

The `data/` and `runs/` directories are intentionally ignored because they hold local datasets and generated model artifacts.

## Run inference

Place the trained weights at `models/best.pt`, then run the detector on a local image or video:

```bash
python -m src.inference path/to/image-or-video.jpg
```

Annotated output is saved under `runs/inference/`. Adjust the confidence threshold or paths with `--confidence`, `--model`, and `--output`. Model weights are kept local and are ignored by Git.
