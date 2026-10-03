# FindMe
<<<<<<< HEAD
Hack Dearborn 5 Project where an AI system is developed to find people in crises such as earthquakes, floods, etc. 
=======

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
>>>>>>> 7c909c0 (configured environment)
