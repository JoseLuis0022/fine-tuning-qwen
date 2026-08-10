#!/usr/bin/env python3
"""
Generador de dataset de tool-calling para Pi (earendil-works/pi).
Target  : Qwen3-8B | Mac 16GB | MLX-LM LoRA
Protocolo: las 6 herramientas REALES de Pi (read, bash, edit, write, web_search, web_fetch)
           capturadas en vivo (ver ../ESPEC_PI.md y ../pi_tools.json).
Formato : se renderiza con el chat template NATIVO de Qwen3 vía apply_chat_template(tools=...)
          -> idéntico a lo que Ollama produce en inferencia. Se eliminan los <think> vacíos
          para que el target empiece directo tras "assistant\\n" (= prompt de generación real).
Voz     : español mexicano casual del usuario ("porfa/pls/a ver/checa/mira/oye/sig/órale/al chile").
Salida  : output/train.jsonl (~800) + output/valid.jsonl (~100), formato {"text": ...}
"""
import json, os, random, hashlib, re
from typing import List, Dict
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOK = AutoTokenizer.from_pretrained(os.path.join(ROOT, "modelo-base"), trust_remote_code=True)
TOOLS = json.load(open(os.path.join(ROOT, "pi_tools.json")))

# ── System prompt: núcleo ESTABLE real de Pi (sin skills/paths/fecha volátiles) ──
PI_SYS = """You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.

Available tools:
- read: Read file contents
- bash: Execute bash commands (ls, grep, find, etc.)
- edit: Make precise file edits with exact text replacement, including multiple disjoint edits in one call
- write: Create or overwrite files

In addition to the tools above, you may have access to other custom tools depending on the project.

Guidelines:
- Use bash for file operations like ls, rg, find
- Use read to examine files instead of cat or sed.
- Use edit for precise changes (edits[].oldText must match exactly)
- When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls
- Use write only for new files or complete rewrites.
- Be concise in your responses
- Show file paths clearly when working with files"""

_THINK_RE = re.compile(r"<think>\s*</think>\s*", re.DOTALL)

# ── Helpers de mensajes ──
def call(name, **args):
    return {"type": "function", "function": {"name": name, "arguments": args}}

def render(*turns) -> Dict:
    """turns: ('user', txt) | ('assistant', txt) | ('calls', [call,...]) | ('tool', content)"""
    msgs = [{"role": "system", "content": PI_SYS}]
    for kind, payload in turns:
        if kind == "user":
            msgs.append({"role": "user", "content": payload})
        elif kind == "assistant":
            msgs.append({"role": "assistant", "content": payload})
        elif kind == "calls":
            msgs.append({"role": "assistant", "content": "", "tool_calls": payload})
        elif kind == "tool":
            msgs.append({"role": "tool", "content": payload})
    text = TOK.apply_chat_template(msgs, tools=TOOLS, tokenize=False, add_generation_prompt=False)
    text = _THINK_RE.sub("", text)  # quitar <think></think> vacíos -> target directo
    return {"text": text.rstrip()}

R = random.choice
RS = random.sample

# ── Pools ──
PY = ["main.py","utils.py","config.py","models.py","routes.py","app.py","server.py","auth.py",
      "api.py","parser.py","schemas.py","services.py","database.py","cache.py","helpers.py",
      "src/app.py","src/models.py","src/utils.py","src/auth.py","src/routes.py","src/config.py",
      "tests/test_app.py","tests/test_models.py","tests/conftest.py"]
TS = ["index.ts","app.ts","utils.ts","types.ts","api.ts","hooks.ts","store.ts","router.ts",
      "src/index.ts","src/app.ts","components/App.tsx","components/Button.tsx","components/Modal.tsx",
      "pages/index.tsx","hooks/useAuth.ts","lib/client.ts"]
CFG = ["package.json","tsconfig.json","config.json",".env","README.md","Dockerfile",
       "docker-compose.yml","requirements.txt","Makefile",".gitignore"]
ALLF = PY + TS + CFG
DIRS = ["src","lib","tests","components","pages","api","utils","scripts","docs","config",
        "models","services","hooks","controllers","migrations"]
PY_PKGS = ["numpy","pandas","requests","fastapi","flask","sqlalchemy","pydantic","pytest",
           "httpx","celery","redis","boto3","openai","langchain","transformers","click","rich"]
JS_PKGS = ["react","next","express","axios","lodash","zod","prisma","tailwindcss","jest",
           "eslint","typescript","vite","vitest","zustand","dayjs","stripe"]
PY_BODIES = [
 'from fastapi import FastAPI\nfrom .routes import router\n\napp = FastAPI(title="MyAPI", version="1.0.0")\napp.include_router(router, prefix="/api/v1")',
 'from sqlalchemy import Column, Integer, String, Boolean\nfrom .database import Base\n\nclass User(Base):\n    __tablename__ = "users"\n    id = Column(Integer, primary_key=True)\n    email = Column(String, unique=True)\n    is_active = Column(Boolean, default=True)',
 'from pydantic import BaseModel, EmailStr\n\nclass UserCreate(BaseModel):\n    email: EmailStr\n    name: str\n    password: str',
 'import pytest\nfrom httpx import AsyncClient\nfrom .main import app\n\n@pytest.mark.asyncio\nasync def test_health(client):\n    resp = await client.get("/health")\n    assert resp.status_code == 200',
 'def average(items):\n    return sum(items) / len(items)\n\ndef process(data):\n    return average(data)',
]
TS_BODIES = [
 'import { useState } from "react";\n\nexport function useAuth() {\n  const [user, setUser] = useState(null);\n  return { user, setUser };\n}',
 'import { z } from "zod";\n\nexport const UserSchema = z.object({\n  id: z.string().uuid(),\n  email: z.string().email(),\n  name: z.string().min(1),\n});',
 'export interface ButtonProps {\n  label: string;\n  onClick: () => void;\n  variant?: "primary" | "secondary";\n}',
]

# ── Voz del usuario (mexicano casual): plantillas de petición ──
def vz(*opts):  # elige una variante de voz
    return R(opts)

# ════════════════════════════════════════════════════════════════
# GENERADORES (cada uno devuelve lista de ejemplos renderizados)
# ════════════════════════════════════════════════════════════════

def g_bash_ls(n=60):
    out = []
    for _ in range(n):
        d = R(DIRS)
        files = RS(ALLF, min(5, len(ALLF)))
        fnames = [f.split("/")[-1] for f in files]
        u = vz(f"oye porfa, ¿qué hay en {d}/?", f"chécame qué archivos tiene {d}/",
               f"a ver, lístame {d}/ pls", f"mira, ¿qué hay dentro de {d}/?",
               f"ándale, muéstrame {d}/", f"qué tiene la carpeta {d}/ porfa")
        out.append(render(
            ("user", u),
            ("calls", [call("bash", command=f"ls -la {d}/")]),
            ("tool", "\n".join(["total 24"] + [f"-rw-r--r--  1 dev staff  {random.randint(200,9000)} Jun {random.randint(1,28)} 10:0{random.randint(0,9)} {f}" for f in fnames])),
            ("assistant", f"En `{d}/` hay {len(fnames)}: {', '.join(fnames[:4])}."),
        ))
    return out

def g_bash_find(n=55):
    out = []
    exts = [("py","Python"),("ts","TypeScript"),("tsx","componentes React"),("json","JSON"),
            ("md","Markdown"),("env","archivos .env"),("yml","YAML")]
    for _ in range(n):
        ext, desc = R(exts)
        d = R(DIRS + ["."])
        hits = [f"./{d}/file{i}.{ext}" for i in range(1, random.randint(2,5))]
        u = vz(f"a ver, ¿cuántos archivos .{ext} hay?", f"oye porfa búscame todos los .{ext}",
               f"chécame los {desc} del proyecto", f"mira, encuéntrame los archivos .{ext} pls",
               f"¿dónde están los .{ext}? porfa")
        cmd = R([f'find {d} -name "*.{ext}"', f'find {d} -type f -name "*.{ext}"'])
        out.append(render(
            ("user", u),
            ("calls", [call("bash", command=cmd)]),
            ("tool", "\n".join(hits)),
            ("assistant", f"Hay {len(hits)} archivo(s) `.{ext}`: {', '.join(h.split('/')[-1] for h in hits)}."),
        ))
    return out

def g_bash_count(n=30):
    out = []
    for _ in range(n):
        ext = R(["py","ts","js","tsx"])
        total = random.randint(8, 90)
        u = vz(f"al chile, ¿cuántos archivos {ext} hay en total?", f"cuéntame los .{ext} porfa",
               f"oye ¿cuántas líneas de {ext} tiene el proyecto?", f"a ver cuántos {ext} hay pls")
        if "líneas" in u:
            out.append(render(
                ("user", u),
                ("calls", [call("bash", command=f'find . -name "*.{ext}" | xargs wc -l | tail -1')]),
                ("tool", f"  {random.randint(500,9000)} total"),
                ("assistant", f"En total hay varias miles de líneas `.{ext}`."),
            ))
        else:
            out.append(render(
                ("user", u),
                ("calls", [call("bash", command=f'find . -name "*.{ext}" | wc -l')]),
                ("tool", f"      {total}"),
                ("assistant", f"Hay **{total}** archivos `.{ext}`."),
            ))
    return out

def g_bash_grep(n=55):
    out = []
    pats = [("TODO","# TODO: validar input"),("FIXME","# FIXME: bug de unicode"),
            ("async def","async def get_users():"),("console.log","console.log('debug');"),
            ("import React","import React from 'react';"),("raise ValueError","raise ValueError('bad')")]
    for _ in range(n):
        pat, line = R(pats)
        f1, f2 = RS(PY, 2)
        u = vz(f"oye, ¿dónde aparece '{pat}' en el código?", f"búscame '{pat}' porfa",
               f"chécame dónde se usa '{pat}'", f"a ver, ¿en qué archivos está '{pat}'? pls",
               f"mira, encuéntrame '{pat}' en el proyecto")
        cmd = R([f'rg "{pat}"', f'grep -rn "{pat}" .'])
        out.append(render(
            ("user", u),
            ("calls", [call("bash", command=cmd)]),
            ("tool", f"{f1}:34:{line}\n{f2}:78:{line}"),
            ("assistant", f"`{pat}` aparece en `{f1}` y `{f2}`."),
        ))
    # ¿dónde se importa X?
    for _ in range(n // 3):
        pkg = R(PY_PKGS)
        out.append(render(
            ("user", vz(f"¿dónde se importa {pkg}? porfa", f"oye chécame dónde uso {pkg}",
                        f"a ver dónde está {pkg} en el código pls")),
            ("calls", [call("bash", command=f'rg "import {pkg}|from {pkg}"')]),
            ("tool", f"src/models.py:1:import {pkg}\nsrc/utils.py:3:from {pkg} import core"),
            ("assistant", f"`{pkg}` se importa en `src/models.py` y `src/utils.py`."),
        ))
    return out

def g_bash_run(n=55):
    out = []
    # pip / npm install
    for _ in range(n // 3):
        if random.random() < 0.6:
            pkg = R(PY_PKGS)
            out.append(render(
                ("user", vz(f"instálame {pkg} porfa", f"oye agrega {pkg} al proyecto",
                            f"a ver, pip install {pkg} pls", f"ándale instala {pkg}")),
                ("calls", [call("bash", command=f"pip install {pkg}")]),
                ("tool", f"Collecting {pkg}\nSuccessfully installed {pkg}-2.1.0"),
                ("assistant", f"Listo, `{pkg}` instalado."),
            ))
        else:
            pkg = R(JS_PKGS)
            out.append(render(
                ("user", vz(f"instala {pkg} porfa", f"oye mete {pkg} con npm pls",
                            f"agrega la dependencia {pkg}")),
                ("calls", [call("bash", command=f"npm install {pkg}")]),
                ("tool", "added 1 package in 2s\nfound 0 vulnerabilities"),
                ("assistant", f"Listo, `{pkg}` instalado vía npm."),
            ))
    # pytest
    for _ in range(n // 4):
        p = random.randint(12, 40)
        out.append(render(
            ("user", vz("córreme los tests porfa", "oye checa si pasan los tests",
                        "a ver, ejecuta pytest pls", "ándale corre las pruebas")),
            ("calls", [call("bash", command="python -m pytest -q")]),
            ("tool", f"collected {p} items\n.....\n{p} passed in {random.uniform(0.5,2.5):.2f}s"),
            ("assistant", f"Todo bien: **{p}/{p}** tests pasaron."),
        ))
    # git status / log
    for _ in range(n // 4):
        if random.random() < 0.5:
            out.append(render(
                ("user", vz("¿cómo va el repo? porfa", "oye chécame el git status",
                            "a ver qué cambios hay pls")),
                ("calls", [call("bash", command="git status")]),
                ("tool", "On branch main\nChanges not staged:\n  modified: src/app.py\nUntracked: src/new.py"),
                ("assistant", "Rama `main`: `src/app.py` modificado y `src/new.py` sin rastrear."),
            ))
        else:
            out.append(render(
                ("user", vz("muéstrame los últimos commits porfa", "oye el git log pls",
                            "a ver el historial de commits")),
                ("calls", [call("bash", command="git log --oneline -5")]),
                ("tool", "a3f9b2c feat: add auth\nb8d1e45 fix: null pointer\nc92a3f1 refactor"),
                ("assistant", "Últimos commits: feat auth, fix null pointer, refactor."),
            ))
    return out

def g_bash_fileops(n=45):
    """Organizar/mover archivos -> la petición tipo 'acomódame la carpeta'."""
    out = []
    for _ in range(n):
        kind = R(["mkdir","mv_ext","organize_downloads","rm","cp","rename"])
        if kind == "mkdir":
            d = R(DIRS)
            out.append(render(
                ("user", vz(f"créame la carpeta {d} porfa", f"oye haz un directorio {d} pls",
                            f"a ver, mkdir {d}")),
                ("calls", [call("bash", command=f"mkdir -p {d}")]),
                ("tool", ""),
                ("assistant", f"Listo, carpeta `{d}/` creada."),
            ))
        elif kind == "mv_ext":
            ext = R(["pdf","png","jpg","csv","zip"])
            out.append(render(
                ("user", vz(f"acomódame los {ext} en una carpeta aparte porfa",
                            f"oye mueve todos los .{ext} a una subcarpeta pls",
                            f"a ver, mete los {ext} en su propia carpeta")),
                ("calls", [call("bash", command=f"mkdir -p {ext}s && mv *.{ext} {ext}s/")]),
                ("tool", ""),
                ("assistant", f"Listo, moví los `.{ext}` a `{ext}s/`."),
            ))
        elif kind == "organize_downloads":
            out.append(render(
                ("user", vz("acomódame la carpeta de descargas, sepárame por tipo porfa",
                            "oye ordéname Downloads por extensión pls",
                            "a ver, organízame las descargas en carpetas")),
                ("calls", [call("bash", command='cd ~/Downloads && for ext in pdf png jpg zip csv; do mkdir -p "$ext" && mv *."$ext" "$ext"/ 2>/dev/null; done')]),
                ("tool", ""),
                ("assistant", "Listo, organicé `~/Downloads` en subcarpetas por tipo (pdf, png, jpg, zip, csv)."),
            ))
        elif kind == "rm":
            out.append(render(
                ("user", vz("bórrame los __pycache__ porfa", "oye limpia los .pyc pls",
                            "a ver elimina los archivos temporales")),
                ("calls", [call("bash", command='find . -name "__pycache__" -type d -exec rm -rf {} +')]),
                ("tool", ""),
                ("assistant", "Listo, limpié los `__pycache__`."),
            ))
        elif kind == "cp":
            f = R(CFG)
            out.append(render(
                ("user", vz(f"hazme un respaldo de {f} porfa", f"oye copia {f} como backup pls")),
                ("calls", [call("bash", command=f"cp {f} {f}.bak")]),
                ("tool", ""),
                ("assistant", f"Listo, respaldo en `{f}.bak`."),
            ))
        else:  # rename
            f = R(PY)
            out.append(render(
                ("user", vz(f"renómbrame {f} a old_{f.split('/')[-1]} porfa")),
                ("calls", [call("bash", command=f"mv {f} {os.path.dirname(f) + '/' if '/' in f else ''}old_{f.split('/')[-1]}")]),
                ("tool", ""),
                ("assistant", f"Listo, renombrado."),
            ))
    return out

def g_read(n=85):
    out = []
    for _ in range(n):
        if random.random() < 0.6:
            f = R(PY); body = R(PY_BODIES)
        else:
            f = R(TS); body = R(TS_BODIES)
        u = vz(f"léeme {f} porfa", f"oye muéstrame {f} pls", f"a ver qué hay en {f}",
               f"chécame el archivo {f}", f"mira, abre {f}", f"ábreme {f} porfa")
        out.append(render(
            ("user", u),
            ("calls", [call("read", path=f)]),
            ("tool", body),
            ("assistant", f"`{f}`: {body.splitlines()[0][:70]}..."),
        ))
    specials = [
        ("package.json", '{"name":"my-app","version":"2.1.0","dependencies":{"react":"^18","next":"^14"}}',
         "app `my-app` v2.1.0 con Next 14 y React 18."),
        (".env", "APP_ENV=production\nDB_URL=postgres://localhost:5432/mydb\nJWT_SECRET=secret",
         "variables: entorno producción, Postgres y JWT."),
        ("requirements.txt", "fastapi==0.110.2\nuvicorn==0.29.0\nsqlalchemy==2.0.29\npytest==8.1.1",
         "deps: FastAPI, uvicorn, SQLAlchemy y pytest."),
        ("config.json", '{"database":{"host":"localhost","port":5432},"cache":{"ttl":3600}}',
         "config: BD localhost:5432 y caché 1h."),
    ]
    for f, body, resp in specials:
        for _ in range(4):
            out.append(render(
                ("user", vz(f"léeme {f} porfa", f"oye chécame {f} pls", f"a ver qué tiene {f}")),
                ("calls", [call("read", path=f)]),
                ("tool", body),
                ("assistant", f"`{f}`: {resp}"),
            ))
    return out

def g_write(n=70):
    out = []
    templates = [
        ("hello.py", 'print("Hola Mundo")\n', "créame un hello.py que imprima Hola Mundo"),
        (".gitignore", "__pycache__/\n*.pyc\n.env\nnode_modules/\ndist/\n.DS_Store\n", "hazme un .gitignore para Python y Node"),
        ("Dockerfile", "FROM python:3.11-slim\nWORKDIR /app\nCOPY . .\nRUN pip install -r requirements.txt\nCMD [\"python\",\"main.py\"]\n", "arma un Dockerfile para una app de Python"),
        ("utils.py", 'import json\n\ndef load(p):\n    with open(p) as f:\n        return json.load(f)\n', "créame utils.py con una función para leer JSON"),
        ("README.md", "# Mi Proyecto\n\n## Instalación\n```bash\npip install -r requirements.txt\n```\n", "hazme un README básico"),
        ("config.json", '{\n  "app": {"name": "MyApp", "debug": false},\n  "port": 8000\n}\n', "genérame un config.json base"),
    ]
    for path, content, desc in templates:
        for _ in range(max(1, n // len(templates) // 2)):
            u = vz(f"{desc} porfa", f"oye, {desc} pls", f"a ver, {desc}", f"ándale {desc}")
            out.append(render(
                ("user", u),
                ("calls", [call("write", path=path, content=content)]),
                ("tool", ""),
                ("assistant", f"Listo, creé `{path}`."),
            ))
    # dinámico
    for _ in range(n):
        if random.random() < 0.6:
            f = R(PY); body = R(PY_BODIES)
        else:
            f = R(TS); body = R(TS_BODIES)
        out.append(render(
            ("user", vz(f"créame {f} con código base porfa", f"oye escríbeme {f} pls",
                        f"a ver, hazme {f} desde cero")),
            ("calls", [call("write", path=f, content=body + "\n")]),
            ("tool", ""),
            ("assistant", f"Listo, creé `{f}`."),
        ))
    return out

def g_edit(n=75):
    out = []
    edits = [
        ("config.json", '"debug": false', '"debug": true', "actívame el modo debug"),
        ("config.json", '"debug": true', '"debug": false', "desactívame el debug"),
        ("package.json", '"version": "1.0.0"', '"version": "2.0.0"', "súbeme la versión a 2.0.0"),
        ("Dockerfile", "FROM python:3.11-slim", "FROM python:3.12-slim", "actualiza Python a 3.12"),
        ("config.json", '"ttl": 3600', '"ttl": 1800', "baja el cache TTL a 30 min"),
        ("src/app.py", 'title="My API"', 'title="Production API"', "cámbiame el título de la API"),
    ]
    for f, old, new, desc in edits:
        for _ in range(max(1, n // len(edits))):
            u = vz(f"{desc} en {f} porfa", f"oye {desc} en {f} pls", f"a ver, {desc} en `{f}`")
            out.append(render(
                ("user", u),
                ("calls", [call("edit", path=f, edits=[{"oldText": old, "newText": new}])]),
                ("tool", "Edit applied successfully"),
                ("assistant", f"Listo, cambié `{f}`."),
            ))
    # edits múltiples en una llamada
    for _ in range(n // 3):
        f = "config.json"
        out.append(render(
            ("user", vz(f"oye porfa pásame {f} a prod: host db.prod.com y puerto 5433",
                        f"a ver, en {f} cámbiame host a db.prod.com y puerto a 5433 pls")),
            ("calls", [call("edit", path=f, edits=[
                {"oldText": '"host": "localhost"', "newText": '"host": "db.prod.com"'},
                {"oldText": '"port": 5432', "newText": '"port": 5433'},
            ])]),
            ("tool", "Edit applied successfully (2 edits)"),
            ("assistant", f"Listo, `{f}`: host → db.prod.com y puerto → 5433."),
        ))
    # dinámico: puertos y versiones en archivos random (alta entropía)
    ports = [3000, 4000, 5000, 8000, 8080, 9000, 5432, 6379]
    for _ in range(n):
        f = R(PY)
        if random.random() < 0.5:
            p1, p2 = RS(ports, 2)
            out.append(render(
                ("user", vz(f"oye cámbiame el puerto de {p1} a {p2} en {f} porfa",
                            f"a ver, en {f} pon el puerto {p2} en vez de {p1} pls")),
                ("calls", [call("edit", path=f, edits=[{"oldText": f"PORT = {p1}", "newText": f"PORT = {p2}"}])]),
                ("tool", "Edit applied successfully"),
                ("assistant", f"Listo, puerto en `{f}` → {p2}."),
            ))
        else:
            v1 = f"{random.randint(0,3)}.{random.randint(0,9)}.{random.randint(0,9)}"
            v2 = f"{random.randint(1,4)}.{random.randint(0,9)}.{random.randint(0,9)}"
            out.append(render(
                ("user", vz(f"súbeme la versión de {v1} a {v2} en {f} porfa",
                            f"oye en {f} cambia la versión a {v2} pls")),
                ("calls", [call("edit", path=f, edits=[{"oldText": f'__version__ = "{v1}"', "newText": f'__version__ = "{v2}"'}])]),
                ("tool", "Edit applied successfully"),
                ("assistant", f"Listo, versión en `{f}` → {v2}."),
            ))
    return out

def g_web_search(n=70):
    out = []
    topics = [
        ("Python asyncio best practices 2026", "mejores prácticas de asyncio en Python",
         "Usa TaskGroup (3.11+), evita llamadas bloqueantes, prefiere httpx sobre requests."),
        ("FastAPI vs Flask performance", "si FastAPI es más rápido que Flask",
         "FastAPI ~65k req/s vs Flask ~25k req/s gracias al soporte async."),
        ("React Server Components 2026", "qué son los React Server Components",
         "Componentes que renderizan en el servidor, reducen el JS del cliente."),
        ("mejores laptops para machine learning 2026", "buenas laptops para ML",
         "Apple Silicon (M-series) y portátiles con RTX 40/50 lideran por memoria unificada y CUDA."),
        ("cómo arreglar CORS en FastAPI", "cómo arreglar el error de CORS en FastAPI",
         "Agrega CORSMiddleware con allow_origins, allow_methods y allow_headers."),
        ("precio del dólar hoy México", "el precio del dólar hoy en México",
         "El tipo de cambio ronda los 18-19 MXN por USD según el día."),
        ("qué es fine-tuning de LLMs", "qué es el fine-tuning de modelos",
         "Es reentrenar un modelo base con datos propios para especializarlo en una tarea."),
        ("mejores extensiones VS Code 2026", "buenas extensiones para VS Code",
         "Populares: linters, GitLens, formatters y asistentes de IA integrados."),
        ("cómo hacer deploy de FastAPI en producción", "cómo desplegar FastAPI en prod",
         "Usa uvicorn/gunicorn detrás de nginx, en contenedor Docker, con variables de entorno."),
        ("diferencia entre Docker y máquina virtual", "la diferencia entre Docker y una VM",
         "Docker comparte el kernel del host (ligero); una VM virtualiza todo el SO (pesado)."),
        ("qué es RAG en inteligencia artificial", "qué es RAG en IA",
         "Retrieval-Augmented Generation: el modelo consulta documentos externos antes de responder."),
        ("cómo optimizar consultas SQL lentas", "cómo acelerar consultas SQL lentas",
         "Agrega índices, evita SELECT *, revisa el plan de ejecución con EXPLAIN."),
        ("clima en Ciudad de México hoy", "el clima de hoy en CDMX",
         "Suele estar templado, 12-24°C, con posibilidad de lluvia por la tarde."),
        ("mejores prácticas de seguridad en APIs", "buenas prácticas de seguridad en APIs",
         "Usa HTTPS, autenticación con tokens, rate limiting y valida toda entrada."),
    ]
    topics = topics + [(f"{p} tutorial 2026", f"un tutorial de {p}",
                        f"Hay guías oficiales y ejemplos prácticos para empezar con {p}.")
                       for p in RS(PY_PKGS + JS_PKGS, 10)]
    for query, what, ans in topics:
        for _ in range(max(1, n // len(topics))):
            u = vz(f"oye porfa búscame en internet {what}", f"a ver, búscame {what} en la web pls",
                   f"chécame en internet {what}", f"al chile googléame {what} porfa",
                   f"mira, investiga {what} en internet", f"oye {what}, búscalo en la web porfa")
            out.append(render(
                ("user", u),
                ("calls", [call("web_search", query=query)]),
                ("tool", f"1. {ans}\n2. Más detalles en la documentación oficial y blogs especializados."),
                ("assistant", ans),
            ))
    return out

def g_web_fetch(n=50):
    out = []
    pages = [
        ("https://pypi.org/pypi/fastapi/json", "la versión más reciente de FastAPI",
         '{"info":{"name":"fastapi","version":"0.115.0"}}', "FastAPI más reciente: **0.115.0**."),
        ("https://news.ycombinator.com/", "qué hay en Hacker News hoy",
         "Top: nuevos modelos de IA local, debate sobre Rust vs Go, lanzamiento de un editor de código.",
         "Lo top en HN: IA local, Rust vs Go y un nuevo editor de código."),
        ("https://imperioon.com/", "de qué trata esta página",
         "Imperioon: software a medida y POS inteligente para negocios, inventario y automatización.",
         "Imperioon ofrece software a medida y POS inteligente: inventario y automatización para negocios."),
        ("https://docs.ollama.com/", "cómo correr un modelo con Ollama",
         "Instala Ollama, luego 'ollama run <modelo>'. Soporta API local y modelos abiertos.",
         "Con Ollama instalado, corres `ollama run <modelo>`; tiene API local y modelos abiertos."),
    ]
    for url, what, content, resp in pages:
        for _ in range(max(1, n // len(pages))):
            u = vz(f"oye porfa chécame esta página y resúmeme {what}: {url}",
                   f"a ver, lée esta url y dime {what} pls: {url}",
                   f"mira {url}, resúmeme {what} porfa",
                   f"al chile entra a {url} y dime {what}")
            out.append(render(
                ("user", u),
                ("calls", [call("web_fetch", url=url)]),
                ("tool", f"Content:\n{content}"),
                ("assistant", resp),
            ))
    # dinámico: versiones de paquetes en PyPI / npm (alta entropía)
    for _ in range(n):
        if random.random() < 0.5:
            pkg = R(PY_PKGS); ver = f"{random.randint(0,3)}.{random.randint(0,30)}.{random.randint(0,9)}"
            url = f"https://pypi.org/pypi/{pkg}/json"
            content = json.dumps({"info": {"name": pkg, "version": ver}})
            resp = f"`{pkg}` más reciente en PyPI: **{ver}**."
        else:
            pkg = R(JS_PKGS); ver = f"{random.randint(0,9)}.{random.randint(0,30)}.{random.randint(0,9)}"
            url = f"https://registry.npmjs.org/{pkg}/latest"
            content = json.dumps({"name": pkg, "version": ver})
            resp = f"`{pkg}` más reciente en npm: **{ver}**."
        out.append(render(
            ("user", vz(f"oye chécame la última versión de {pkg} porfa: {url}",
                        f"a ver, entra a {url} y dime qué versión es pls",
                        f"mira {url}, ¿qué versión sale? porfa")),
            ("calls", [call("web_fetch", url=url)]),
            ("tool", f"Content:\n{content}"),
            ("assistant", resp),
        ))
    return out

def g_chains(n=70):
    out = []
    # read -> edit
    for _ in range(n // 6):
        f = R(PY)
        out.append(render(
            ("user", vz(f"oye léeme {f} y cámbiame VERSION a 2.0 porfa",
                        f"a ver, revisa {f} y sube la versión a 2.0 pls")),
            ("calls", [call("read", path=f)]),
            ("tool", R(PY_BODIES) + '\nVERSION = "1.0"'),
            ("calls", [call("edit", path=f, edits=[{"oldText": 'VERSION = "1.0"', "newText": 'VERSION = "2.0"'}])]),
            ("tool", "Edit applied successfully"),
            ("assistant", f"Listo, leí `{f}` y subí la versión a 2.0."),
        ))
    # web_search -> write
    for _ in range(n // 6):
        out.append(render(
            ("user", vz("búscame en internet las mejores prácticas de FastAPI y guárdamelas en notas.md porfa",
                        "oye investiga buenas prácticas de FastAPI y mételo en notas.md pls")),
            ("calls", [call("web_search", query="FastAPI best practices 2026")]),
            ("tool", "1. Usa dependencias para inyección. 2. Valida con Pydantic. 3. Async en I/O."),
            ("calls", [call("write", path="notas.md", content="# FastAPI - buenas prácticas\n- Inyección de dependencias\n- Validación con Pydantic\n- Async en I/O\n")]),
            ("tool", ""),
            ("assistant", "Listo, busqué las prácticas y las guardé en `notas.md`."),
        ))
    # web_fetch -> write (resumen a archivo)
    for _ in range(n // 6):
        url = "https://docs.ollama.com/"
        out.append(render(
            ("user", vz(f"oye entra a {url}, resúmelo y guárdalo en resumen.md porfa",
                        f"a ver, lée {url} y hazme un resumen en resumen.md pls")),
            ("calls", [call("web_fetch", url=url)]),
            ("tool", "Content:\nOllama corre modelos abiertos localmente con una API simple."),
            ("calls", [call("write", path="resumen.md", content="# Ollama\nCorre modelos abiertos localmente con una API simple.\n")]),
            ("tool", ""),
            ("assistant", "Listo, resumí la página y la guardé en `resumen.md`."),
        ))
    # bash(find) -> read
    for _ in range(n // 6):
        out.append(render(
            ("user", vz("oye busca el archivo principal y léemelo porfa",
                        "a ver, encuentra main.py y ábremelo pls")),
            ("calls", [call("bash", command='find . -name "main.py"')]),
            ("tool", "./src/main.py"),
            ("calls", [call("read", path="src/main.py")]),
            ("tool", R(PY_BODIES)),
            ("assistant", "Encontré `src/main.py` y aquí está: es el punto de entrada de la app."),
        ))
    # write -> bash(test)
    for _ in range(n // 6):
        out.append(render(
            ("user", vz("créame un test para utils.py y córrelo porfa",
                        "oye hazme un test rápido y ejecútalo pls")),
            ("calls", [call("write", path="tests/test_utils.py", content="def test_ok():\n    assert 1 + 1 == 2\n")]),
            ("tool", ""),
            ("calls", [call("bash", command="python -m pytest tests/test_utils.py -q")]),
            ("tool", "1 passed in 0.05s"),
            ("assistant", "Listo, creé el test y pasó: 1/1."),
        ))
    # debug fix: read -> edit (ZeroDivision)
    for _ in range(n // 6):
        f = R(PY)
        out.append(render(
            ("user", vz(f"oye {f} truena con ZeroDivisionError, arréglalo porfa",
                        f"a ver, hay un bug de división entre cero en {f}, córrelo pls")),
            ("calls", [call("read", path=f)]),
            ("tool", "def average(items):\n    return sum(items) / len(items)"),
            ("calls", [call("edit", path=f, edits=[{"oldText": "    return sum(items) / len(items)", "newText": "    if not items:\n        return 0.0\n    return sum(items) / len(items)"}])]),
            ("tool", "Edit applied successfully"),
            ("assistant", f"Arreglado: `average()` ahora regresa 0.0 si la lista está vacía."),
        ))
    return out

def g_no_tool(n=55):
    out = []
    qa = [
        ("¿qué es un closure?", "Un closure es una función que recuerda las variables de su ámbito externo aunque se ejecute fuera de él."),
        ("¿diferencia entre == y === en JS?", "`==` compara con coerción de tipos; `===` compara valor y tipo. Siempre usa `===`."),
        ("¿qué hace useEffect en React?", "Ejecuta efectos secundarios tras el render. Con `[]` corre solo al montar; con `[x]` cuando `x` cambia."),
        ("¿qué es Docker en pocas palabras?", "Empaqueta tu app con sus dependencias en un contenedor aislado, así corre igual en cualquier máquina."),
        ("¿qué es una API REST?", "Un estilo de API con endpoints por recurso sobre HTTP, usando GET/POST/PUT/DELETE."),
        ("¿qué es el GIL en Python?", "El Global Interpreter Lock impide ejecutar hilos en paralelo en tareas CPU-bound; no afecta I/O con asyncio."),
        ("¿async def vs def?", "`async def` define una corutina que usa `await` y necesita un event loop; `def` es síncrona y bloqueante."),
        ("¿qué es un índice en una base de datos?", "Una estructura que acelera las búsquedas a costa de más espacio y escrituras más lentas."),
        ("¿qué es Big O?", "Describe cómo crece el costo de un algoritmo: O(1), O(log n), O(n), O(n²)..."),
        ("¿qué es JWT?", "Un token con 3 partes (header, payload, firma) que el servidor valida sin guardar sesión."),
        ("¿qué es un decorador en Python?", "Una función que envuelve a otra para extender su comportamiento; se aplica con `@nombre`."),
        ("¿REST vs GraphQL?", "REST tiene endpoints fijos por recurso; GraphQL usa un solo endpoint y el cliente pide solo los campos que necesita."),
        ("¿qué es CI/CD?", "Integración y entrega continuas: automatizan pruebas y despliegue cada vez que cambias el código."),
        ("¿qué es memoización?", "Cachear el resultado de funciones costosas para no recalcularlas. `lru_cache` en Python, `useMemo` en React."),
        ("¿stack vs queue?", "Stack es LIFO (último en entrar, primero en salir); queue es FIFO (primero en entrar, primero en salir)."),
        ("¿qué es CORS?", "Política del navegador que bloquea peticiones a otro dominio; se habilita con headers en el servidor."),
        ("¿SQL vs NoSQL?", "SQL es relacional con esquema fijo y ACID; NoSQL es flexible y escala horizontal (documentos, clave-valor)."),
        ("¿qué es un virtualenv?", "Un entorno de Python aislado con sus propias dependencias, para no mezclar versiones entre proyectos."),
        ("¿qué es idempotencia?", "Una operación que da el mismo resultado sin importar cuántas veces se ejecute. GET, PUT y DELETE lo son; POST no."),
        ("¿qué es un webhook?", "Una URL tuya que un servicio externo llama automáticamente cuando ocurre un evento."),
        ("¿qué hace git rebase?", "Reaplica tus commits sobre otra base, dejando un historial lineal (a diferencia de merge)."),
    ]
    greetings = [
        ("hola", "¡Hola! ¿En qué te ayudo? Puedo leer y editar archivos, correr comandos o buscar en la web."),
        ("oye, ¿en qué me puedes ayudar?", "Puedo leer/editar tu código, correr comandos en bash, buscar en internet y crear archivos. ¿Qué necesitas?"),
        ("buenas, ¿qué onda?", "¡Qué tal! Listo para ayudarte con tu proyecto: archivos, comandos o búsquedas web. ¿Por dónde empezamos?"),
    ]
    wraps = ["{q}", "oye, {q}", "a ver, {q} porfa", "tengo una duda: {q}", "explícame {q} pls"]
    for q, a in qa:
        for _ in range(max(1, n // (len(qa) + len(greetings)))):
            out.append(render(("user", R(wraps).format(q=q)), ("assistant", a)))
    for g, a in greetings:
        for _ in range(3):
            out.append(render(("user", g), ("assistant", a)))
    return out

# ════════════════════════════════════════════════════════════════
def main():
    print("Generando dataset Pi tool-calling (voz mexicana)...")
    pool = []
    pool += g_bash_ls(70)
    pool += g_bash_find(65)
    pool += g_bash_count(40)
    pool += g_bash_grep(65)
    pool += g_bash_run(70)
    pool += g_bash_fileops(55)
    pool += g_read(100)
    pool += g_write(90)
    pool += g_edit(90)
    pool += g_web_search(80)
    pool += g_web_fetch(70)
    pool += g_chains(96)
    pool += g_no_tool(75)
    print(f"  Generados: {len(pool)}")

    seen, uniq = set(), []
    for it in pool:
        k = hashlib.md5(it["text"].encode()).hexdigest()
        if k not in seen:
            seen.add(k); uniq.append(it)
    print(f"  Únicos:    {len(uniq)}")
    random.shuffle(uniq)

    T, V = 800, 100
    if len(uniq) >= T + V:
        train, valid = uniq[:T], uniq[T:T+V]
    else:
        s = int(len(uniq) * 0.9); train, valid = uniq[:s], uniq[s:]

    os.makedirs(os.path.join(HERE, "output"), exist_ok=True)
    for name, data in (("train", train), ("valid", valid)):
        with open(os.path.join(HERE, "output", f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for it in data:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    avg = sum(len(e["text"]) for e in train) / max(len(train), 1)
    maxtok = max(len(TOK(e["text"])["input_ids"]) for e in train)
    print(f"\nDataset listo en ./output/")
    print(f"  train.jsonl: {len(train)} ej.  | valid.jsonl: {len(valid)} ej.")
    print(f"  Avg length : {avg:.0f} chars (~{avg/4:.0f} tokens) | Max tokens: {maxtok}")
    print("\nDistribución de LLAMADAS reales por herramienta en train:")
    # contar solo tool_calls reales (assistant), no el bloque de esquemas ni la instrucción
    for t in ["read","bash","edit","write","web_search","web_fetch"]:
        pat = f'<tool_call>\n{{"name": "{t}"'
        c = sum(e["text"].count(pat) for e in train)
        ej = sum(1 for e in train if pat in e["text"])
        print(f"  {t:<12} {c:4d} llamadas en {ej:4d} ej.  {'#' * (ej // 4)}")
    notool = sum(1 for e in train if '<tool_call>\n{"name": "' not in e["text"])
    print(f"  {'(sin tool)':<12} {notool:4d} ej.  {'#' * (notool // 4)}")

if __name__ == "__main__":
    main()
