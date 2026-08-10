# Fine Tuning — Qwen Claude Code / Qwen Pi

Este repo contiene el código, datasets y resultados de dos proyectos de fine-tuning
sobre modelos Qwen: **Qwen Claude Code** y **Qwen Pi**.

## Qué SÍ está en el repo

- Scripts de entrenamiento y benchmark (`entrenar.py`, `benchmark/*.py`, `dataset/generate.py`)
- Datasets de entrenamiento (`data/*.jsonl`, `dataset/output/*.jsonl`)
- Resultados y logs de entrenamientos previos (`resultados/`)
- Adaptadores LoRA entrenados (`adaptadores/*.safetensors`) vía **Git LFS**
- `requirements.txt` por proyecto (generados con `pip freeze` de los venvs originales)

## Qué NO está en el repo (y por qué)

- **`modelo-base/`** (~4 GB por proyecto): pesos del modelo base de Hugging Face.
  Se descargan de nuevo, no se versionan.
- **`modelo-fusionado/`** (~35 GB): modelo base + adaptador fusionado. Se regenera
  localmente después de entrenar (ver más abajo).
- **`venv/`, `venv-gguf/`**: entornos virtuales de Python. Se recrean con los
  `requirements.txt` incluidos.
- **`llama.cpp-tools/`**: clon de terceros (llama.cpp), no es código propio.
  Clónalo aparte si necesitas conversión a GGUF: `git clone https://github.com/ggml-org/llama.cpp`

## Cómo poner en marcha una instancia nueva

```bash
git clone https://github.com/JoseLuis0022/fine-tuning-qwen.git
cd fine-tuning-qwen

# elige el proyecto con el que quieras trabajar
cd "Qwen Claude Code"   # o "Qwen Pi"

# 1. crear entorno virtual e instalar dependencias
python3 -m venv venv
venv/bin/python3 -m pip install -r requirements.txt

# (Qwen Claude Code también tiene un segundo entorno para conversión GGUF)
python3 -m venv venv-gguf
venv-gguf/bin/python3 -m pip install -r requirements-gguf.txt

# 2. descargar el modelo base (no incluido en el repo)
#    revisa entrenar.py para ver qué modelo base se usa exactamente
#    y descárgalo con huggingface-cli o snapshot_download()

# 3. entrenar / usar el adaptador LoRA ya incluido en adaptadores/
venv/bin/python3 entrenar.py
```

## Notas

- Los adaptadores LoRA se versionan con **Git LFS**. Instala `git-lfs` antes de
  clonar (`brew install git-lfs && git lfs install`) para que se descarguen
  correctamente; si ya clonaste sin LFS, corre `git lfs pull` después.
- `.claude/` (config local de Claude Code) no se versiona a propósito.
