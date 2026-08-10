#!/usr/bin/env python3
"""
Claude Code Tool-Calling Dataset Generator
Target : Qwen2.5-7B | Mac M2 Pro 16 GB | MLX-LM LoRA
Format : ChatML  {"text": "<|im_start|>system..."}
Output : train.jsonl  (~900 ej.)  +  valid.jsonl (~100 ej.)
"""
import json, random, hashlib
from typing import List, Dict
# ────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ────────────────────────────────────────────────────────────
SYS = """Eres un agente de programación con acceso a herramientas del sistema.
Herramientas disponibles:
- Bash(command, timeout?)       → ejecuta comandos shell
- Read(file_path)               → lee un archivo
- Write(file_path, content)     → crea o sobreescribe un archivo
- Edit(file_path, old_string, new_string) → reemplaza texto en archivo
- MultiEdit(file_path, edits[]) → múltiples ediciones en un archivo
- Glob(pattern, path?)          → busca archivos por patrón glob
- Grep(pattern, path?, include?)→ busca texto dentro de archivos
- LS(path)                      → lista un directorio
- WebFetch(url, prompt)         → descarga y procesa una URL
- WebSearch(query)              → búsqueda web
- Task(description, prompt)     → delega subtarea a subagente
- TodoRead()                    → lee tareas pendientes
- TodoWrite(todos[])            → actualiza tareas pendientes
- NotebookRead(notebook_path)   → lee un Jupyter notebook
- NotebookEdit(notebook_path, cell_id, new_source) → edita celda
Para llamar una herramienta usa EXACTAMENTE:
<tool_call>
{"name": "NombreHerramienta", "arguments": {...}}
</tool_call>
Reglas:
1. Emite el bloque tool_call sin texto previo cuando necesites una herramienta.
2. Para llamadas en paralelo, emite múltiples bloques seguidos.
3. Encadena herramientas cuando sea necesario.
4. Si no necesitas herramienta, responde directamente."""
# ────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────
def tc(name: str, **kwargs) -> str:
    return f'<tool_call>\n{json.dumps({"name": name, "arguments": kwargs}, ensure_ascii=False)}\n</tool_call>'
def tr(name: str, content: str) -> str:
    return f'<tool_response>\n{json.dumps({"name": name, "content": content}, ensure_ascii=False)}\n</tool_response>'
def build(*turns) -> Dict:
    text = f"<|im_start|>system\n{SYS}<|im_end|>\n"
    for role, content in turns:
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return {"text": text.rstrip()}
R  = random.choice
RS = random.sample
# ────────────────────────────────────────────────────────────
# POOLS DE VARIABLES
# ────────────────────────────────────────────────────────────
PY = [
    "main.py","utils.py","config.py","models.py","routes.py","app.py",
    "server.py","client.py","helpers.py","constants.py","database.py",
    "auth.py","api.py","core.py","parser.py","validators.py","schemas.py",
    "middleware.py","exceptions.py","decorators.py","services.py","tasks.py",
    "cache.py","permissions.py","repository.py","serializers.py","signals.py",
    "src/app.py","src/models.py","src/utils.py","src/auth.py","src/routes.py",
    "src/schemas.py","src/services.py","src/database.py","src/config.py",
    "tests/test_app.py","tests/test_models.py","tests/test_auth.py",
    "tests/test_utils.py","tests/conftest.py","tests/test_routes.py",
]
TS = [
    "index.ts","app.ts","utils.ts","types.ts","api.ts","hooks.ts","store.ts",
    "router.ts","middleware.ts","validators.ts","constants.ts","helpers.ts",
    "src/index.ts","src/app.ts","src/utils.ts","src/types.ts",
    "components/App.tsx","components/Button.tsx","components/Modal.tsx",
    "components/Header.tsx","components/UserCard.tsx","components/Form.tsx",
    "pages/index.tsx","pages/login.tsx","pages/dashboard.tsx",
    "hooks/useAuth.ts","hooks/useFetch.ts","hooks/useForm.ts",
    "lib/client.ts","lib/validators.ts","lib/helpers.ts",
]
ALL_F = PY + TS + [
    "package.json","tsconfig.json","config.json",".eslintrc.json",
    "README.md","Makefile","Dockerfile","docker-compose.yml",
    ".env",".gitignore","requirements.txt","pyproject.toml",
    ".github/workflows/ci.yml",".github/workflows/deploy.yml",
]
DIRS = [
    "src","lib","tests","components","pages","api","utils","scripts",
    "docs","config","data","dist","public","core","services",
    "hooks","models","controllers","middleware","migrations","assets",
]
PY_PKGS = [
    "numpy","pandas","requests","fastapi","flask","sqlalchemy","pydantic",
    "pytest","aiohttp","celery","redis","boto3","httpx","alembic","passlib",
    "python-jose","pillow","openai","langchain","transformers","torch",
    "scikit-learn","matplotlib","seaborn","click","typer","rich","loguru",
]
JS_PKGS = [
    "react","next","express","axios","lodash","zod","prisma","tailwindcss",
    "jest","eslint","prettier","typescript","@tanstack/react-query","socket.io",
    "mongoose","drizzle-orm","hono","vite","vitest","playwright","zustand",
    "jotai","framer-motion","clsx","dayjs","sharp","stripe","@auth/core",
]
BRANCHES = [
    "main","develop","feature/auth","feature/api-v2","feature/payments",
    "hotfix/null-pointer","hotfix/login-bug","release/1.2.0","fix/cors-error",
    "chore/update-deps","refactor/models","feat/dashboard","fix/typo",
]
VERSIONS_PAIRS = [
    ("1.0.0","2.0.0"),("1.2.3","1.3.0"),("0.9.0","1.0.0"),
    ("2.1.0","2.2.0"),("3.0.0","3.1.0"),("1.0.1","1.1.0"),
]
PORTS = [3000, 4000, 5000, 8000, 8080, 8888, 9000, 9090]
DB_HOSTS = ["localhost","db.staging.com","db.prod.com","postgres","127.0.0.1"]
LOG_LEVELS = ["DEBUG","INFO","WARNING","ERROR","CRITICAL"]
# Contenidos realistas para archivos Python
PY_BODIES = [
    'from fastapi import FastAPI\nfrom .routes import router\nfrom .database import init_db\n\napp = FastAPI(title="MyAPI", version="1.0.0")\napp.include_router(router, prefix="/api/v1")\n\n@app.on_event("startup")\nasync def startup():\n    await init_db()',
    'from sqlalchemy import Column, Integer, String, DateTime, Boolean\nfrom .database import Base\nfrom datetime import datetime\n\nclass User(Base):\n    __tablename__ = "users"\n    id = Column(Integer, primary_key=True)\n    email = Column(String, unique=True, nullable=False)\n    name = Column(String, nullable=False)\n    is_active = Column(Boolean, default=True)\n    created_at = Column(DateTime, default=datetime.utcnow)',
    'from fastapi import APIRouter, Depends, HTTPException\nfrom .schemas import UserCreate, UserResponse\nfrom .services import UserService\n\nrouter = APIRouter(prefix="/users", tags=["users"])\n\n@router.get("/", response_model=list[UserResponse])\nasync def list_users(service: UserService = Depends()):\n    return await service.get_all()\n\n@router.post("/", status_code=201)\nasync def create_user(data: UserCreate, service: UserService = Depends()):\n    return await service.create(data)',
    'from jose import jwt\nfrom passlib.context import CryptContext\nfrom datetime import datetime, timedelta\nfrom .config import settings\n\npwd_ctx = CryptContext(schemes=["bcrypt"])\n\ndef hash_password(plain: str) -> str:\n    return pwd_ctx.hash(plain)\n\ndef verify_password(plain: str, hashed: str) -> bool:\n    return pwd_ctx.verify(plain, hashed)\n\ndef create_token(data: dict) -> str:\n    payload = {**data, "exp": datetime.utcnow() + timedelta(minutes=30)}\n    return jwt.encode(payload, settings.SECRET_KEY)',
    'from pydantic import BaseModel, EmailStr, validator\nfrom typing import Optional\nfrom datetime import datetime\n\nclass UserBase(BaseModel):\n    email: EmailStr\n    name: str\n\nclass UserCreate(UserBase):\n    password: str\n\n    @validator("password")\n    def password_length(cls, v):\n        if len(v) < 8:\n            raise ValueError("Min 8 chars")\n        return v\n\nclass UserResponse(UserBase):\n    id: int\n    is_active: bool\n    created_at: datetime\n\n    class Config:\n        from_attributes = True',
    'import pytest\nfrom httpx import AsyncClient\nfrom .main import app\n\n@pytest.fixture\nasync def client():\n    async with AsyncClient(app=app, base_url="http://test") as ac:\n        yield ac\n\n@pytest.mark.asyncio\nasync def test_create_user(client):\n    resp = await client.post("/api/v1/users/", json={\n        "email": "t@test.com", "name": "Test", "password": "secret123"\n    })\n    assert resp.status_code == 201\n\n@pytest.mark.asyncio\nasync def test_duplicate_email(client):\n    payload = {"email": "d@test.com", "name": "D", "password": "pass1234"}\n    await client.post("/api/v1/users/", json=payload)\n    resp = await client.post("/api/v1/users/", json=payload)\n    assert resp.status_code == 400',
    'from pydantic_settings import BaseSettings\nfrom functools import lru_cache\n\nclass Settings(BaseSettings):\n    APP_NAME: str = "MyApp"\n    VERSION: str = "1.0.0"\n    DEBUG: bool = False\n    SECRET_KEY: str\n    DATABASE_URL: str\n    REDIS_URL: str = "redis://localhost:6379"\n    ACCESS_TOKEN_EXPIRE: int = 30\n    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000"]\n\n    class Config:\n        env_file = ".env"\n\n@lru_cache()\ndef get_settings() -> Settings:\n    return Settings()',
    'import aioredis\nimport json\nfrom typing import Optional, Any\n\nclass Cache:\n    def __init__(self, url: str, ttl: int = 3600):\n        self.url = url\n        self.ttl = ttl\n        self._r = None\n\n    async def connect(self):\n        self._r = await aioredis.from_url(self.url)\n\n    async def get(self, key: str) -> Optional[Any]:\n        v = await self._r.get(key)\n        return json.loads(v) if v else None\n\n    async def set(self, key: str, value: Any) -> None:\n        await self._r.setex(key, self.ttl, json.dumps(value))\n\n    async def delete(self, key: str) -> None:\n        await self._r.delete(key)',
]
TS_BODIES = [
    'import { useState, useCallback } from "react";\nimport type { User } from "../types";\n\nexport function useAuth() {\n  const [user, setUser] = useState<User | null>(null);\n  const [loading, setLoading] = useState(false);\n\n  const login = useCallback(async (email: string, password: string) => {\n    setLoading(true);\n    try {\n      const res = await fetch("/api/auth/login", {\n        method: "POST",\n        headers: { "Content-Type": "application/json" },\n        body: JSON.stringify({ email, password }),\n      });\n      const data = await res.json();\n      setUser(data.user);\n    } finally {\n      setLoading(false);\n    }\n  }, []);\n\n  const logout = () => setUser(null);\n  return { user, loading, login, logout };\n}',
    'import { z } from "zod";\n\nexport const UserSchema = z.object({\n  id: z.string().uuid(),\n  email: z.string().email(),\n  name: z.string().min(1).max(100),\n  role: z.enum(["admin", "user", "viewer"]).default("user"),\n  createdAt: z.string().datetime(),\n  isActive: z.boolean().default(true),\n});\n\nexport const CreateUserSchema = UserSchema.omit({ id: true, createdAt: true });\nexport type User = z.infer<typeof UserSchema>;\nexport type CreateUser = z.infer<typeof CreateUserSchema>;',
    'import React, { forwardRef } from "react";\nimport { clsx } from "clsx";\n\nexport interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {\n  variant?: "primary" | "secondary" | "danger" | "ghost";\n  size?: "sm" | "md" | "lg";\n  isLoading?: boolean;\n}\n\nexport const Button = forwardRef<HTMLButtonElement, ButtonProps>(\n  ({ variant="primary", size="md", isLoading, className, children, disabled, ...props }, ref) => (\n    <button\n      ref={ref}\n      disabled={disabled || isLoading}\n      className={clsx("btn", `btn-${variant}`, `btn-${size}`, { "opacity-50": disabled || isLoading }, className)}\n      {...props}\n    >\n      {isLoading ? <span className="spinner" /> : null}\n      {children}\n    </button>\n  )\n);\nButton.displayName = "Button";',
    'import { create } from "zustand";\nimport type { User } from "../types";\n\ninterface AuthState {\n  user: User | null;\n  token: string | null;\n  setUser: (user: User) => void;\n  setToken: (token: string) => void;\n  logout: () => void;\n}\n\nexport const useAuthStore = create<AuthState>()((set) => ({\n  user: null,\n  token: typeof window !== "undefined" ? localStorage.getItem("token") : null,\n  setUser: (user) => set({ user }),\n  setToken: (token) => {\n    if (typeof window !== "undefined") localStorage.setItem("token", token);\n    set({ token });\n  },\n  logout: () => {\n    if (typeof window !== "undefined") localStorage.removeItem("token");\n    set({ user: null, token: null });\n  },\n}));',
]
# ────────────────────────────────────────────────────────────
# GENERADORES
# ────────────────────────────────────────────────────────────
def gen_bash_ls(n=80) -> List[Dict]:
    results = []
    for _ in range(n):
        d = R(DIRS)
        files = RS(ALL_F, min(5, len(ALL_F)))
        fnames = [f.split("/")[-1] for f in files]
        listing = "\n".join(
            [f"drwxr-xr-x  1 dev staff   64 Jun {random.randint(1,28)} 10:00 ."] +
            [f"-rw-r--r--  1 dev staff {random.randint(200,9999):5d} Jun {random.randint(1,28)} {random.randint(9,18)}:{random.randint(10,59)} {f}" for f in fnames]
        )
        uv = R([f"Lista el directorio {d}", f"¿Qué hay en {d}?",
                f"Muéstrame {d}/", f"Show files in {d}", f"Contenido de {d}"])
        results.append(build(
            ("user", uv),
            ("assistant", tc("Bash", command=f"ls -la {d}/")),
            ("tool", tr("Bash", listing)),
            ("assistant", f"`{d}/` contiene: {', '.join(fnames[:3])}{'...' if len(fnames)>3 else ''}."),
        ))
    return results
def gen_bash_git(n=70) -> List[Dict]:
    results = []
    templates = [
        ("git status",
         ["¿Estado del repo?","¿Qué cambios hay?","git status","Estado del repositorio"],
         lambda b,f: f"On branch {b}\nChanges not staged:\n  modified: {f}\nUntracked: src/new_feature.py",
         lambda b,f: f"Rama `{b}`: `{f}` modificado y 1 nuevo sin rastrear."),
        ("git log --oneline -10",
         ["Historial de commits","Últimos commits","Log del repo","¿Qué commits hay?"],
         lambda b,f: f"a3f9b2c (HEAD -> {b}) feat: update {f}\nb8d1e45 fix: null pointer\nc92a3f1 refactor: cleanup",
         lambda b,f: f"Últimos commits: feat `{f}`, fix null pointer, refactor cleanup."),
        ("git diff HEAD",
         ["¿Qué cambié?","Ver el diff","Cambios sin commitear","¿Qué modifiqué?"],
         lambda b,f: f"diff --git a/{f} b/{f}\n--- a/{f}\n+++ b/{f}\n@@ -5,0 +6 @@\n+    logger.info('updated')",
         lambda b,f: f"Cambio en `{f}`: se agregó una línea de logging."),
        ("git branch -a",
         ["Lista las ramas","¿Qué branches hay?","Ramas del repo","Muéstrame las ramas"],
         lambda b,f: f"* {b}\n  main\n  feature/payments\n  remotes/origin/main\n  remotes/origin/{b}",
         lambda b,f: f"5 ramas. Activa: `{b}`."),
        ("git stash",
         ["Guarda los cambios","Stashea todo","Guarda el WIP"],
         lambda b,f: f"Saved working directory state WIP on {b}: a3f9b2c",
         lambda b,f: f"Cambios guardados en stash. Working dir limpio."),
        ("git pull origin main",
         ["Actualiza el repo","Pull de main","Trae cambios de origin","Sincroniza con main"],
         lambda b,f: "From github.com:org/repo\nAlready up to date.",
         lambda b,f: "Repositorio ya actualizado con `origin/main`."),
        ("git commit -am 'fix: patch issues'",
         ["Commitea todo","Haz commit de los cambios","Commit rápido de todo"],
         lambda b,f: f"[{b} a1b2c3d] fix: patch issues\n 2 files changed, 8 insertions(+), 3 deletions(-)",
         lambda b,f: f"Commit creado en `{b}`."),
    ]
    per = max(1, n // len(templates))
    for cmd, uvs, out_fn, resp_fn in templates:
        for _ in range(per):
            b = R(BRANCHES); f = R(PY)
            results.append(build(
                ("user", R(uvs)),
                ("assistant", tc("Bash", command=cmd)),
                ("tool", tr("Bash", out_fn(b, f))),
                ("assistant", resp_fn(b, f)),
            ))
    return results
def gen_bash_run(n=80) -> List[Dict]:
    results = []
    # pip install
    for _ in range(n // 4):
        pkg = R(PY_PKGS)
        results.append(build(
            ("user", R([f"Instala {pkg}", f"pip install {pkg}", f"Agrega {pkg} al proyecto"])),
            ("assistant", tc("Bash", command=f"pip install {pkg}")),
            ("tool", tr("Bash", f"Collecting {pkg}\nSuccessfully installed {pkg}-latest")),
            ("assistant", f"✅ `{pkg}` instalado."),
        ))
    # npm install
    for _ in range(n // 6):
        pkg = R(JS_PKGS)
        results.append(build(
            ("user", R([f"Instala {pkg}", f"npm install {pkg}", f"Agrega dependencia {pkg}"])),
            ("assistant", tc("Bash", command=f"npm install {pkg}")),
            ("tool", tr("Bash", "added 1 package in 2s\nfound 0 vulnerabilities")),
            ("assistant", f"✅ `{pkg}` instalado via npm."),
        ))
    # pytest pass
    for _ in range(5):
        passed = random.randint(10, 40)
        results.append(build(
            ("user", R(["Corre los tests","Ejecuta pytest","¿Pasan los tests?","python -m pytest"])),
            ("assistant", tc("Bash", command="python -m pytest tests/ -v")),
            ("tool", tr("Bash", f"collected {passed} items\n\n" +
                "tests/test_app.py::test_create PASSED\ntests/test_app.py::test_read PASSED\n" +
                f"\n{passed} passed in {random.uniform(0.5,3.0):.2f}s")),
            ("assistant", f"✅ **{passed}/{passed} tests** pasaron."),
        ))
    # pytest fail
    results.append(build(
        ("user", "Corre los tests"),
        ("assistant", tc("Bash", command="python -m pytest tests/ -v")),
        ("tool", tr("Bash", "FAILED tests/test_auth.py::test_jwt_expired - AssertionError: expected 401, got 200\n\n1 failed, 17 passed in 1.45s")),
        ("assistant", "⚠️ **1 test falló**: `test_jwt_expired` retorna 200 en vez de 401."),
    ))
    # coverage
    results.append(build(
        ("user", "Tests con cobertura"),
        ("assistant", tc("Bash", command="python -m pytest --cov=src tests/ -q")),
        ("tool", tr("Bash", "Name          Stmts  Miss  Cover\n-------------------------------\nsrc/app.py      45     3    93%\nsrc/utils.py    67     8    88%\nTOTAL          146    11    92%")),
        ("assistant", "Cobertura: **92%** total. `utils.py` es el módulo más bajo (88%)."),
    ))
    # python version, node version, disk space
    results.append(build(("user","¿Versión de Python?"),("assistant",tc("Bash",command="python3 --version")),("tool",tr("Bash","Python 3.11.6")),("assistant","Python 3.11.6.")))
    results.append(build(("user","¿Espacio en disco?"),("assistant",tc("Bash",command="df -h .")),("tool",tr("Bash","460G  190G  270G  41%")),("assistant","270 GB disponibles (41% usado).")))
    results.append(build(("user","Cuenta las líneas de Python"),("assistant",tc("Bash",command="find . -name '*.py' | xargs wc -l | tail -1")),("tool",tr("Bash","  5812 total")),("assistant","**5,812 líneas** Python en total.")))
    return results
def gen_read(n=90) -> List[Dict]:
    results = []
    uvs = ["Léeme {f}","Muéstrame {f}","¿Qué hay en {f}?","Abre {f}","Lee el archivo {f}","Muéstrame el contenido de {f}"]
    for _ in range(n):
        if random.random() < 0.6:
            f = R(PY); body = R(PY_BODIES)
            resp = body.split("\n")[0][:80]
        else:
            f = R(TS); body = R(TS_BODIES)
            resp = body.split("\n")[0][:80]
        results.append(build(
            ("user", R(uvs).format(f=f)),
            ("assistant", tc("Read", file_path=f)),
            ("tool", tr("Read", body)),
            ("assistant", f"`{f}`: {resp}..."),
        ))
    # Special files
    specials = [
        ("package.json", '{"name":"my-app","version":"2.1.0","scripts":{"dev":"next dev","build":"next build","test":"jest"},"dependencies":{"react":"^18","next":"^14"}}',
         "app `my-app` v2.1.0 con Next.js 14 y React 18."),
        (".env", "APP_ENV=production\nDB_URL=postgresql://user:pass@localhost:5432/mydb\nJWT_SECRET=ultra_secret\nREDIS_URL=redis://localhost:6379",
         "Variables configuradas: BD PostgreSQL, JWT y Redis."),
        ("README.md", "# My Project\n\n## Setup\n```bash\npip install -r requirements.txt\npython main.py\n```\n\n## API\nDocs en `/docs`.",
         "README con instrucciones de instalación y referencia a docs en `/docs`."),
        ("requirements.txt", "fastapi==0.110.2\nuvicorn[standard]==0.29.0\nsqlalchemy==2.0.29\npydantic==2.6.0\npytest==8.1.1",
         "Dependencias principales: FastAPI, SQLAlchemy, Pydantic, pytest."),
        ("config.json", '{"database":{"host":"localhost","port":5432},"cache":{"ttl":3600},"logging":{"level":"INFO"}}',
         "Config: BD localhost:5432, caché TTL 1h, logging INFO."),
    ]
    for f, body, resp in specials:
        results.append(build(
            ("user", R(["Lee {}","Muéstrame {}","¿Qué hay en {}?"]).format(f)),
            ("assistant", tc("Read", file_path=f)),
            ("tool", tr("Read", body)),
            ("assistant", f"`{f}`: {resp}"),
        ))
    return results
def gen_write(n=80) -> List[Dict]:
    results = []
    templates = [
        ("Crea utils.py con funciones para JSON","utils.py",
         '"""Utility functions."""\nimport json\nfrom typing import Any, Dict\n\ndef load_json(path: str) -> Dict:\n    with open(path) as f:\n        return json.load(f)\n\ndef save_json(data: Any, path: str) -> None:\n    with open(path, "w") as f:\n        json.dump(data, f, indent=2)\n',
         "Creé `utils.py` con `load_json()` y `save_json()`."),
        ("Crea .gitignore para Python y Node",".gitignore",
         "__pycache__/\n*.pyc\n.env\n.venv/\nvenv/\nnode_modules/\n.next/\ndist/\nbuild/\n*.log\n.DS_Store\n.coverage\nhtmlcov/\n.pytest_cache/\n",
         "`.gitignore` creado para Python y Node."),
        ("Crea un Dockerfile para FastAPI","Dockerfile",
         "FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nEXPOSE 8000\nCMD [\"uvicorn\",\"main:app\",\"--host\",\"0.0.0.0\",\"--port\",\"8000\"]\n",
         "Dockerfile creado: Python 3.11-slim, expone puerto 8000."),
        ("Crea config.json con configuración base","config.json",
         '{\n  "app":{"name":"MyApp","version":"1.0.0","debug":false},\n  "database":{"host":"localhost","port":5432},\n  "cache":{"ttl":3600}\n}\n',
         "`config.json` creado con secciones app, database y cache."),
        ("Crea el workflow de CI para GitHub Actions",".github/workflows/ci.yml",
         'name: CI\non:\n  push:\n    branches: [main, develop]\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n      - uses: actions/setup-python@v5\n        with:\n          python-version: "3.11"\n      - run: pip install -r requirements.txt\n      - run: pytest tests/ -v\n',
         "CI creado: corre en push a main/develop, Python 3.11."),
        ("Crea el componente Button en TypeScript","components/Button.tsx",
         'import React from "react";\n\ninterface ButtonProps {\n  label: string;\n  onClick: () => void;\n  disabled?: boolean;\n  variant?: "primary"|"secondary";\n}\n\nexport const Button: React.FC<ButtonProps> = ({label,onClick,disabled=false,variant="primary"}) => (\n  <button onClick={onClick} disabled={disabled} className={`btn btn-${variant}`}>\n    {label}\n  </button>\n);\n',
         "`Button.tsx` creado con props `label`, `onClick`, `disabled`, `variant`."),
        ("Crea requirements.txt para FastAPI","requirements.txt",
         "fastapi==0.110.2\nuvicorn[standard]==0.29.0\nsqlalchemy==2.0.29\npydantic==2.6.0\npython-jose[cryptography]==3.3.0\npasslib[bcrypt]==1.7.4\nhttpx==0.27.0\nalembic==1.13.1\npytest==8.1.1\n",
         "`requirements.txt` creado con el stack completo de FastAPI."),
        ("Crea docker-compose.yml con app y postgres","docker-compose.yml",
         "version: '3.8'\nservices:\n  app:\n    build: .\n    ports: ['8000:8000']\n    env_file: .env\n    depends_on: [db]\n  db:\n    image: postgres:15\n    environment:\n      POSTGRES_DB: mydb\n      POSTGRES_USER: user\n      POSTGRES_PASSWORD: password\n    ports: ['5432:5432']\n",
         "`docker-compose.yml` con servicios `app` y `db` (Postgres 15)."),
        ("Crea un Makefile con targets básicos","Makefile",
         ".PHONY: dev test lint\ndev:\n\tuvicorn main:app --reload\ntest:\n\tpytest tests/ -v --cov=src\nlint:\n\tflake8 src/ && black --check src/\nformat:\n\tblack src/ tests/\nclean:\n\trm -rf __pycache__ .pytest_cache htmlcov\n",
         "`Makefile` con targets: dev, test, lint, format, clean."),
        ("Crea tests básicos para utils.py","tests/test_utils.py",
         'import pytest\nfrom utils import load_json, save_json\n\ndef test_round_trip(tmp_path):\n    data = {"x": 1, "y": "hello"}\n    p = str(tmp_path / "t.json")\n    save_json(data, p)\n    assert load_json(p) == data\n\ndef test_load_missing():\n    with pytest.raises(FileNotFoundError):\n        load_json("no_existe.json")\n',
         "Tests creados en `tests/test_utils.py`: round-trip y file-not-found."),
    ]
    per = max(1, n // len(templates))
    for user_msg, path, content, response in templates:
        for _ in range(per):
            results.append(build(
                ("user", user_msg),
                ("assistant", tc("Write", file_path=path, content=content)),
                ("tool", tr("Write", "File written successfully")),
                ("assistant", response),
            ))
    # Dynamic creation across the file/body pools (much higher entropy)
    write_uvs = ["Crea {f}", "Necesito el archivo {f}", "Genera {f} con código base",
                 "Escribe {f}", "Hazme {f} desde cero"]
    for _ in range(max(1, n)):
        if random.random() < 0.6:
            f = R(PY); body = R(PY_BODIES)
        else:
            f = R(TS); body = R(TS_BODIES)
        first_line = body.split("\n")[0][:60]
        results.append(build(
            ("user", R(write_uvs).format(f=f)),
            ("assistant", tc("Write", file_path=f, content=body)),
            ("tool", tr("Write", "File written successfully")),
            ("assistant", f"Creé `{f}`: {first_line}..."),
        ))
    return results
def gen_edit(n=80) -> List[Dict]:
    results = []
    v_pairs = VERSIONS_PAIRS
    edits = [
        ("config.json", 'Cambia el nivel de log a INFO', '"level": "DEBUG"', '"level": "INFO"', "Log level → INFO."),
        ("src/app.py", "Cambia el título de la API", 'title="My API"', 'title="Production API"', "Título → `Production API`."),
        ("package.json", "Actualiza la versión", '"version": "1.0.0"', '"version": "1.1.0"', "Versión → 1.1.0."),
        ("Dockerfile", "Actualiza Python a 3.12", "FROM python:3.11-slim", "FROM python:3.12-slim", "Imagen → Python 3.12-slim."),
        ("Dockerfile", "Actualiza Python a 3.11", "FROM python:3.10-slim", "FROM python:3.11-slim", "Imagen → Python 3.11-slim."),
        (".github/workflows/ci.yml", "Actualiza Python", 'python-version: "3.10"', 'python-version: "3.12"', "CI → Python 3.12."),
        ("config.json", "Activa modo debug", '"debug": false', '"debug": true', "Modo debug activado."),
        ("config.json", "Desactiva debug", '"debug": true', '"debug": false', "Modo debug desactivado."),
        ("config.json", "Cambia el cache TTL a 30m", '"ttl": 3600', '"ttl": 1800', "Cache TTL → 30 minutos."),
    ]
    for f, user_msg, old, new, response in edits:
        for _ in range(max(1, n // len(edits))):
            results.append(build(
                ("user", f"{user_msg} en `{f}`"),
                ("assistant", tc("Edit", file_path=f, old_string=old, new_string=new)),
                ("tool", tr("Edit", "Edit applied successfully")),
                ("assistant", response),
            ))
    # Dynamic port/version edits
    for _ in range(20):
        f = R(PY); p1, p2 = R(PORTS), R(PORTS)
        results.append(build(
            ("user", f"Cambia el puerto de {p1} a {p2} en {f}"),
            ("assistant", tc("Edit", file_path=f, old_string=f"port={p1}", new_string=f"port={p2}")),
            ("tool", tr("Edit", "Edit applied successfully")),
            ("assistant", f"Puerto cambiado de {p1} a {p2} en `{f}`."),
        ))
    for _ in range(15):
        f = R(PY); v1, v2 = R(v_pairs)
        results.append(build(
            ("user", f"Actualiza la versión de {v1} a {v2} en {f}"),
            ("assistant", tc("Edit", file_path=f, old_string=f'VERSION = "{v1}"', new_string=f'VERSION = "{v2}"')),
            ("tool", tr("Edit", "Edit applied successfully")),
            ("assistant", f"Versión en `{f}` → {v2}."),
        ))
    return results
def gen_multiedit(n=50) -> List[Dict]:
    results = []
    cases = [
        (lambda: R(PY), "Agrega import sys, VERSION → 2.0, desactiva DEBUG",
         [{"old_string":"import os\n","new_string":"import os\nimport sys\n"},
          {"old_string":'VERSION = "1.0"',"new_string":'VERSION = "2.0"'},
          {"old_string":"DEBUG = True","new_string":"DEBUG = False"}],
         "3 cambios aplicados."),
        ("config.json", "Cambia a prod: host db.prod.com, port 5433, ssl true",
         [{"old_string":'"host": "localhost"',"new_string":'"host": "db.prod.com"'},
          {"old_string":'"port": 5432',"new_string":'"port": 5433'},
          {"old_string":'"ssl": false',"new_string":'"ssl": true'}],
         "Config BD → db.prod.com:5433 con SSL activado."),
        ("Dockerfile", "Python 3.12, puerto 9000",
         [{"old_string":"FROM python:3.11-slim","new_string":"FROM python:3.12-slim"},
          {"old_string":"EXPOSE 8000","new_string":"EXPOSE 9000"},
          {"old_string":'"--port", "8000"',"new_string":'"--port", "9000"'}],
         "Dockerfile: Python→3.12, puerto→9000."),
    ]
    for item in cases:
        f_or_fn, user_msg, edits_list, response = item
        for _ in range(max(1, n // len(cases) // 5)):
            f = f_or_fn() if callable(f_or_fn) else f_or_fn
            results.append(build(
                ("user", f"{user_msg} en `{f}`"),
                ("assistant", tc("MultiEdit", file_path=f, edits=edits_list)),
                ("tool", tr("MultiEdit", f"{len(edits_list)} edits applied")),
                ("assistant", response),
            ))
    # Dynamic port + version + debug combo across files (high entropy)
    for _ in range(n):
        f = R(PY); p1, p2 = R(PORTS), R(PORTS); v1, v2 = R(VERSIONS_PAIRS)
        edits_list = [
            {"old_string": f"port={p1}", "new_string": f"port={p2}"},
            {"old_string": f'VERSION = "{v1}"', "new_string": f'VERSION = "{v2}"'},
        ]
        results.append(build(
            ("user", f"Cambia el puerto de {p1} a {p2} y la versión de {v1} a {v2} en {f}"),
            ("assistant", tc("MultiEdit", file_path=f, edits=edits_list)),
            ("tool", tr("MultiEdit", "2 edits applied")),
            ("assistant", f"`{f}`: puerto → {p2}, versión → {v2}."),
        ))
    return results
def gen_glob(n=70) -> List[Dict]:
    results = []
    specs = [
        ("**/*.py", "Todos los archivos Python", "src/main.py\nsrc/utils.py\ntests/test_app.py\nmain.py", "4 archivos Python."),
        ("**/*.ts", "Archivos TypeScript", "src/index.ts\nsrc/utils.ts\nhooks/useAuth.ts", "3 archivos TypeScript."),
        ("**/*.tsx", "Componentes React", "components/App.tsx\ncomponents/Button.tsx\npages/index.tsx", "3 componentes React."),
        ("**/*.json", "Archivos JSON", "package.json\ntsconfig.json\nconfig/settings.json", "3 archivos JSON."),
        ("tests/**", "Contenido de tests/", "tests/conftest.py\ntests/test_app.py\ntests/test_models.py", "3 archivos en tests/."),
        ("*.md", "Markdown en raíz", "README.md\nCHANGELOG.md\nCONTRIBUTING.md", "3 Markdowns."),
        ("**/*.env*", "Archivos de entorno", ".env\n.env.example\n.env.local", "3 archivos de entorno."),
        ("**/*.test.ts", "Tests TypeScript", "src/__tests__/api.test.ts\nhooks/__tests__/useAuth.test.ts", "2 test files TypeScript."),
        ("**/*.yaml", "Archivos YAML", ".github/workflows/ci.yml\ndocker-compose.yml", "2 archivos YAML."),
        ("src/**/*.py", "Python en src/", "src/app.py\nsrc/models.py\nsrc/routes.py\nsrc/utils.py", "4 archivos Python en src/."),
    ]
    uvs = ["Encuentra todos los {ext}", "Lista los {ext}", "¿Cuántos {ext} hay?", "Busca {ext}"]
    per = max(1, n // len(specs))
    for pat, desc, result, resp in specs:
        ext = pat.replace("**/","").replace("src/","").replace("*","")
        for _ in range(per):
            results.append(build(
                ("user", R(uvs).format(ext=ext if ext else desc.lower())),
                ("assistant", tc("Glob", pattern=pat)),
                ("tool", tr("Glob", result)),
                ("assistant", resp),
            ))
    for d in RS(DIRS, 6):
        results.append(build(
            ("user", f"¿Qué Python hay en {d}/?"),
            ("assistant", tc("Glob", pattern="**/*.py", path=d)),
            ("tool", tr("Glob", f"{d}/main.py\n{d}/utils.py")),
            ("assistant", f"2 archivos Python en `{d}/`."),
        ))
    # Dynamic extension+directory combos (high entropy)
    for ext in ["py","ts","tsx","json","md","yml","css","sh"]:
        for _ in range(max(1, n // 8)):
            d = R(DIRS)
            results.append(build(
                ("user", f"Busca archivos .{ext} en {d}/"),
                ("assistant", tc("Glob", pattern=f"{d}/**/*.{ext}")),
                ("tool", tr("Glob", f"{d}/file1.{ext}\n{d}/file2.{ext}\n{d}/file3.{ext}")),
                ("assistant", f"3 archivos `.{ext}` en `{d}/`."),
            ))
    return results
def gen_grep(n=70) -> List[Dict]:
    results = []
    searches = [
        ("TODO", "src/app.py:34:# TODO: add validation\nsrc/utils.py:67:# TODO: optimize", "2 TODOs pendientes."),
        ("FIXME", "src/parser.py:78:# FIXME: unicode bug\nsrc/db.py:12:# FIXME: pooling", "2 FIXMEs."),
        ("async def", "src/app.py:15:async def startup():\nsrc/routes.py:23:async def get_users():", "2 funciones async."),
        ("console.log", "src/index.ts:34:console.log('start');\nlib/api.ts:89:console.log('err');", "2 console.log — considera usar logger."),
        ("import React", "components/App.tsx:1:import React from 'react';\ncomponents/Modal.tsx:1:import React from 'react';", "React en 2 componentes."),
        ("raise ValueError", "src/validators.py:23:raise ValueError('bad email')\nsrc/models.py:56:raise ValueError('short')", "2 ValueError."),
        ("useEffect", "components/App.tsx:12:useEffect(() => {\ncomponents/List.tsx:8:useEffect(() => {", "useEffect en 2 componentes."),
        ("SELECT.*\\+|execute.*SELECT", "src/legacy.py:34:db.execute('SELECT * FROM users WHERE id='+id)", "⚠️ SQL injection potencial en legacy.py:34."),
    ]
    uvs = ["Busca '{}' en el proyecto","¿Dónde aparece '{}'?","Grep de '{}'","Encuentra '{}'"]
    per = max(1, n // len(searches))
    for pattern, result, resp in searches:
        for _ in range(per):
            results.append(build(
                ("user", R(uvs).format(pattern)),
                ("assistant", tc("Grep", pattern=pattern, path=".")),
                ("tool", tr("Grep", result)),
                ("assistant", resp),
            ))
    for pkg in RS(PY_PKGS, len(PY_PKGS)):
        results.append(build(
            ("user", f"¿Dónde se importa {pkg}?"),
            ("assistant", tc("Grep", pattern=f"import {pkg}|from {pkg}", path=".", include="*.py")),
            ("tool", tr("Grep", f"src/models.py:1:import {pkg}\nsrc/utils.py:3:from {pkg} import core")),
            ("assistant", f"`{pkg}` importado en `models.py` y `utils.py`."),
        ))
    for pkg in RS(JS_PKGS, len(JS_PKGS)):
        results.append(build(
            ("user", f"¿Dónde se usa {pkg}?"),
            ("assistant", tc("Grep", pattern=f"from ['\"]{pkg}['\"]", path=".", include="*.ts")),
            ("tool", tr("Grep", f"src/index.ts:1:import x from '{pkg}'\nlib/api.ts:4:import y from '{pkg}'")),
            ("assistant", f"`{pkg}` se usa en `index.ts` y `api.ts`."),
        ))
    return results
def gen_ls(n=50) -> List[Dict]:
    results = []
    contents = {
        "src":       "app.py\nmodels.py\nroutes.py\nutils.py\ndatabase.py\n__init__.py",
        "tests":     "conftest.py\ntest_app.py\ntest_models.py\ntest_routes.py\n__init__.py",
        "components":"App.tsx\nButton.tsx\nHeader.tsx\nFooter.tsx\nUserList.tsx\nModal.tsx",
        "api":       "users.py\nauth.py\nposts.py\ncomments.py\n__init__.py",
        ".":         "src/\ntests/\nREADME.md\nDockerfile\nrequirements.txt\npackage.json\nmain.py",
        "scripts":   "deploy.sh\nsetup.sh\nbackup.sh\nmigrate.sh\nseed.sh",
        "docs":      "API.md\nSETUP.md\nCHANGELOG.md\nARCHITECTURE.md",
        "config":    "base.json\ndevelopment.json\nproduction.json\ntest.json",
    }
    uvs = ["Lista {}","¿Qué hay en {}?","Muéstrame {}","Contenido de {}","Show {}"]
    per = max(1, n // len(contents))
    for d, content in contents.items():
        n_files = len(content.split("\n"))
        for _ in range(per):
            results.append(build(
                ("user", R(uvs).format(d)),
                ("assistant", tc("LS", path=d)),
                ("tool", tr("LS", content)),
                ("assistant", f"`{d}` tiene {n_files} elementos: {', '.join(content.split(chr(10))[:3])}..."),
            ))
    # Dynamic directory listings across the full DIRS pool
    for _ in range(n):
        d = R(DIRS)
        files = RS(ALL_F, min(4, len(ALL_F)))
        fnames = [f.split("/")[-1] for f in files]
        listing = "\n".join(fnames)
        results.append(build(
            ("user", R(uvs).format(f"{d}/")),
            ("assistant", tc("LS", path=d)),
            ("tool", tr("LS", listing)),
            ("assistant", f"`{d}/` tiene {len(fnames)} archivos: {', '.join(fnames[:3])}..."),
        ))
    return results
def gen_webfetch(n=40) -> List[Dict]:
    results = []
    cases = [
        ("https://api.github.com/repos/anthropics/anthropic-sdk-python/releases/latest",
         "¿Última versión del SDK Python de Anthropic?","versión más reciente",
         '{"tag_name":"v0.34.2","published_at":"2024-06-10"}',
         "SDK Python Anthropic: **v0.34.2** (10 jun 2024)."),
        ("https://pypi.org/pypi/fastapi/json",
         "¿Versión más reciente de FastAPI?","versión disponible",
         '{"info":{"name":"fastapi","version":"0.110.2"}}',
         "FastAPI en PyPI: **0.110.2**."),
        ("https://registry.npmjs.org/react/latest",
         "¿Última versión de React?","versión más reciente",
         '{"name":"react","version":"18.3.0"}',
         "React más reciente: **18.3.0**."),
        ("https://jsonplaceholder.typicode.com/users/1",
         "Obtén el usuario 1 de la API de prueba","nombre y email",
         '{"id":1,"name":"Leanne Graham","email":"sincere@april.biz"}',
         "Usuario 1: **Leanne Graham** (sincere@april.biz)."),
        ("https://docs.anthropic.com/claude/reference/messages-create",
         "¿Cómo usar stream en la API de Claude?","parámetro stream",
         "The stream parameter (boolean): returns Server-Sent Events when true.",
         "Con `stream: true` la API devuelve SSE (Server-Sent Events)."),
    ]
    per = max(1, n // len(cases))
    for url, user_msg, prompt, content, resp in cases:
        for _ in range(per):
            results.append(build(
                ("user", user_msg),
                ("assistant", tc("WebFetch", url=url, prompt=prompt)),
                ("tool", tr("WebFetch", content)),
                ("assistant", resp),
            ))
    # Dynamic PyPI/npm version lookups (high entropy via random package + version)
    for _ in range(n):
        pkg = R(PY_PKGS)
        ver = f"{random.randint(0,3)}.{random.randint(0,20)}.{random.randint(0,9)}"
        results.append(build(
            ("user", f"¿Cuál es la última versión de {pkg} en PyPI?"),
            ("assistant", tc("WebFetch", url=f"https://pypi.org/pypi/{pkg}/json", prompt="versión más reciente")),
            ("tool", tr("WebFetch", json.dumps({"info": {"name": pkg, "version": ver}}))),
            ("assistant", f"`{pkg}` en PyPI: **{ver}**."),
        ))
    for _ in range(n):
        pkg = R(JS_PKGS)
        ver = f"{random.randint(0,9)}.{random.randint(0,20)}.{random.randint(0,9)}"
        results.append(build(
            ("user", f"¿Versión más reciente de {pkg} en npm?"),
            ("assistant", tc("WebFetch", url=f"https://registry.npmjs.org/{pkg}/latest", prompt="versión más reciente")),
            ("tool", tr("WebFetch", json.dumps({"name": pkg, "version": ver}))),
            ("assistant", f"`{pkg}` en npm: **{ver}**."),
        ))
    return results
def gen_websearch(n=40) -> List[Dict]:
    results = []
    cases = [
        ("Python asyncio best practices 2024","¿Mejores prácticas asyncio Python?",
         "Use TaskGroup (3.11+), avoid blocking calls, use async context managers, httpx over requests.",
         "Principales: `TaskGroup` (3.11+), evitar código bloqueante, `httpx` para HTTP async."),
        ("FastAPI vs Flask performance 2024","¿FastAPI es más rápido que Flask?",
         "FastAPI ~65k req/s vs Flask ~25k req/s. Async support explains the gap.",
         "FastAPI (~65k req/s) es ~2.5x más rápido que Flask (~25k req/s)."),
        ("fix CORS error FastAPI","¿Cómo arreglo CORS en FastAPI?",
         "Add CORSMiddleware: allow_origins=['*'], allow_methods=['*'], allow_headers=['*'].",
         "Agrega `CORSMiddleware` con `allow_origins`, `allow_methods`, `allow_headers`."),
        ("Qwen2.5 fine-tuning MLX Mac Apple Silicon",
         "¿Cómo hacer fine-tuning de Qwen2.5 con MLX en Mac?",
         "python -m mlx_lm.lora --model Qwen/Qwen2.5-7B-Instruct --train --data ./data",
         "Comando: `python -m mlx_lm.lora --model Qwen/Qwen2.5-7B-Instruct --train --data ./data`."),
        ("pydantic v2 migration breaking changes",
         "¿Cambios de Pydantic v1 a v2?",
         "@validator → @field_validator, .dict() → .model_dump(), Config → model_config.",
         "Cambios clave: `@validator`→`@field_validator`, `.dict()`→`.model_dump()`."),
        ("SQLAlchemy 2.0 async session tutorial",
         "¿Cómo usar async con SQLAlchemy 2.0?",
         "Use AsyncSession: async with AsyncSession(engine) as session: await session.execute(select(User))",
         "SQLAlchemy 2.0 async: `AsyncSession` con `async with` y `await session.execute()`."),
    ]
    per = max(1, n // len(cases))
    for query, user_msg, result, resp in cases:
        for _ in range(per):
            results.append(build(
                ("user", user_msg),
                ("assistant", tc("WebSearch", query=query)),
                ("tool", tr("WebSearch", result)),
                ("assistant", resp),
            ))
    # Dynamic searches across the package pools + topics (high entropy)
    topics = [
        "best practices", "performance tips", "common pitfalls", "migration guide",
        "security vulnerabilities", "testing strategies", "deployment checklist",
    ]
    for _ in range(n):
        pkg = R(PY_PKGS + JS_PKGS)
        topic = R(topics)
        results.append(build(
            ("user", f"Busca {topic} para {pkg}"),
            ("assistant", tc("WebSearch", query=f"{pkg} {topic}")),
            ("tool", tr("WebSearch", f"Found relevant guidance on {topic} for {pkg}: official docs recommend reviewing the changelog and following community style guides.")),
            ("assistant", f"Encontré recomendaciones sobre {topic} para `{pkg}` en la documentación oficial."),
        ))
    return results
def gen_todo(n=40) -> List[Dict]:
    results = []
    todo_items = [
        {"id":"1","content":"Implementar autenticación JWT","status":"in_progress","priority":"high"},
        {"id":"2","content":"Escribir tests para API","status":"pending","priority":"medium"},
        {"id":"3","content":"Actualizar documentación","status":"pending","priority":"low"},
    ]
    results.append(build(("user","¿Qué tareas tengo?"),("assistant",tc("TodoRead")),("tool",tr("TodoRead","[]")),("assistant","No hay tareas pendientes.")))
    results.append(build(
        ("user","Muéstrame mis tareas"),
        ("assistant",tc("TodoRead")),
        ("tool",tr("TodoRead",json.dumps(todo_items))),
        ("assistant","3 tareas: 🔄 JWT auth (alta), ⏳ tests (media), ⏳ docs (baja)."),
    ))
    new_todos = [
        {"id":"1","content":"Revisar PRs","status":"pending","priority":"high"},
        {"id":"2","content":"Refactorizar auth","status":"pending","priority":"medium"},
        {"id":"3","content":"Agregar logs","status":"pending","priority":"low"},
    ]
    results.append(build(
        ("user","Agrega: revisar PRs (alta), refactorizar auth (media), agregar logs (baja)"),
        ("assistant",tc("TodoWrite", todos=new_todos)),
        ("tool",tr("TodoWrite","Todos updated")),
        ("assistant","✅ 3 tareas creadas."),
    ))
    updated = [{"id":"1","content":"Revisar PRs","status":"completed","priority":"high"},
               {"id":"2","content":"Refactorizar auth","status":"in_progress","priority":"medium"}]
    results.append(build(
        ("user","Marca 'Revisar PRs' como completado e inicia 'Refactorizar auth'"),
        ("assistant",tc("TodoRead")),
        ("tool",tr("TodoRead",json.dumps(new_todos))),
        ("assistant",tc("TodoWrite", todos=updated)),
        ("tool",tr("TodoWrite","Todos updated")),
        ("assistant","✅ PRs completado, auth en progreso."),
    ))
    for _ in range(max(1, n - 4)):
        tasks = [{"id":str(i+1),"content":R(["Deploy a staging","Fix linter","Code review","Update deps","Write changelog","Security audit"]),"status":"pending","priority":R(["high","medium","low"])} for i in range(random.randint(2,4))]
        results.append(build(
            ("user","Crea un plan de tareas para hoy"),
            ("assistant",tc("TodoWrite", todos=tasks)),
            ("tool",tr("TodoWrite","Todos updated")),
            ("assistant",f"✅ {len(tasks)} tareas creadas."),
        ))
    return results
def gen_notebook(n=30) -> List[Dict]:
    results = []
    nb = json.dumps({"cells":[
        {"id":"c1","cell_type":"markdown","source":"# Analysis\nData exploration."},
        {"id":"c2","cell_type":"code","source":"import pandas as pd\nimport numpy as np\ndf = pd.read_csv('data.csv')\ndf.head()"},
        {"id":"c3","cell_type":"code","source":"df.describe()"},
        {"id":"c4","cell_type":"code","source":"df.groupby('region')['sales'].sum().plot(kind='bar')"},
    ]})
    for _ in range(n // 3):
        nb_name = R(["analysis.ipynb","eda.ipynb","model.ipynb","exploration.ipynb"])
        results.append(build(
            ("user", R([f"Muéstrame {nb_name}", f"Lee el notebook {nb_name}", f"¿Qué hay en {nb_name}?"])),
            ("assistant", tc("NotebookRead", notebook_path=nb_name)),
            ("tool", tr("NotebookRead", nb)),
            ("assistant", f"`{nb_name}`: 4 celdas — markdown intro, carga de datos, estadísticas y gráfico."),
        ))
    results.append(build(
        ("user","Añade matplotlib a los imports del notebook analysis.ipynb"),
        ("assistant",tc("NotebookRead", notebook_path="analysis.ipynb")),
        ("tool",tr("NotebookRead", nb)),
        ("assistant",tc("NotebookEdit", notebook_path="analysis.ipynb", cell_id="c2",
                        new_source="import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\ndf = pd.read_csv('data.csv')\ndf.head()")),
        ("tool",tr("NotebookEdit","Cell updated")),
        ("assistant","Celda `c2` actualizada: `import matplotlib.pyplot as plt` añadido."),
    ))
    # Dynamic notebooks across name + analysis-op pools (higher entropy)
    nb_names = ["analysis.ipynb","eda.ipynb","model.ipynb","exploration.ipynb",
                "report.ipynb","sales_analysis.ipynb","churn_model.ipynb","etl.ipynb"]
    cell_ops = [
        "df.groupby('region')['sales'].sum().plot(kind='bar')",
        "df['total'] = df['price'] * df['qty']",
        "df.dropna(inplace=True)",
        "df.corr()",
        "sns.heatmap(df.corr(), annot=True)",
        "df['date'] = pd.to_datetime(df['date'])",
        "df.pivot_table(index='category', values='sales', aggfunc='sum')",
        "model.fit(X_train, y_train)",
    ]
    for _ in range(n):
        name = R(nb_names); op = R(cell_ops)
        nb_dyn = json.dumps({"cells": [
            {"id": "c1", "cell_type": "markdown", "source": "# Analysis"},
            {"id": "c2", "cell_type": "code", "source": "import pandas as pd\ndf = pd.read_csv('data.csv')"},
            {"id": "c3", "cell_type": "code", "source": op},
        ]})
        results.append(build(
            ("user", R([f"Muéstrame {name}", f"Lee el notebook {name}", f"¿Qué hay en {name}?", f"Resume {name}"])),
            ("assistant", tc("NotebookRead", notebook_path=name)),
            ("tool", tr("NotebookRead", nb_dyn)),
            ("assistant", f"`{name}`: 3 celdas — markdown, carga de datos y una operación de análisis."),
        ))
    return results
def gen_chains(n=100) -> List[Dict]:
    results = []
    # Read → Edit
    for _ in range(n // 10):
        f = R(PY); body = R(PY_BODIES)
        v1, v2 = R(VERSIONS_PAIRS)
        results.append(build(
            ("user", f"Actualiza la versión de {v1} a {v2} en {f} (léelo primero)"),
            ("assistant", tc("Read", file_path=f)),
            ("tool", tr("Read", body + f'\n__version__ = "{v1}"')),
            ("assistant", tc("Edit", file_path=f, old_string=f'__version__ = "{v1}"', new_string=f'__version__ = "{v2}"')),
            ("tool", tr("Edit","Edit applied")),
            ("assistant", f"Versión en `{f}` → {v2}."),
        ))
    # Glob → Grep
    for _ in range(n // 10):
        pat = R(["TODO","FIXME","console.log","raise ValueError"])
        results.append(build(
            ("user", f"Busca todos los '{pat}' en el código Python"),
            ("assistant", tc("Glob", pattern="**/*.py")),
            ("tool", tr("Glob", "\n".join(RS(PY[:10], 4)))),
            ("assistant", tc("Grep", pattern=pat, path=".", include="*.py")),
            ("tool", tr("Grep", f"src/utils.py:45:# {pat}: fix this\nsrc/models.py:78:# {pat}: revisit")),
            ("assistant", f"Encontré 2 ocurrencias de `{pat}` en el proyecto."),
        ))
    # Write → Bash (test)
    for _ in range(n // 10):
        results.append(build(
            ("user", "Crea tests para utils.py y córrelos"),
            ("assistant", tc("Write", file_path="tests/test_utils.py", content='import pytest\nfrom utils import load_json\ndef test_load(tmp_path):\n    import json\n    p = tmp_path/"t.json"\n    p.write_text(json.dumps({"x":1}))\n    assert load_json(str(p)) == {"x":1}\n')),
            ("tool", tr("Write","File written")),
            ("assistant", tc("Bash", command="python -m pytest tests/test_utils.py -v")),
            ("tool", tr("Bash", "collected 1 item\ntest_utils.py::test_load PASSED\n1 passed in 0.08s")),
            ("assistant", "Tests creados y ejecutados. ✅ 1/1 passed."),
        ))
    # LS → Read
    for _ in range(n // 10):
        results.append(build(
            ("user", "Explora src/ y léeme el archivo principal"),
            ("assistant", tc("LS", path="src")),
            ("tool", tr("LS", "app.py\nmodels.py\nroutes.py\nutils.py\n__init__.py")),
            ("assistant", tc("Read", file_path="src/app.py")),
            ("tool", tr("Read", R(PY_BODIES))),
            ("assistant", "`src/` tiene 5 archivos. `app.py` es el punto de entrada principal."),
        ))
    # Bash → TodoWrite
    for _ in range(n // 10):
        results.append(build(
            ("user", "Analiza el estado del proyecto y crea un plan de tareas"),
            ("assistant", tc("Bash", command="python -m pytest tests/ -q 2>&1 | tail -2")),
            ("tool", tr("Bash", f"{random.randint(1,3)} failed, {random.randint(15,25)} passed in {random.uniform(0.8,2.5):.2f}s")),
            ("assistant", tc("Bash", command="find . -name '*.py' | xargs grep -l 'TODO' 2>/dev/null | head -5")),
            ("tool", tr("Bash", f"./src/{R(PY[:10]).split('/')[-1]}\n./tests/{R(['test_app.py','test_utils.py'])}")),
            ("assistant", tc("TodoWrite", todos=[
                {"id":"1","content":"Fix failing tests","status":"pending","priority":"high"},
                {"id":"2","content":"Resolve TODO items","status":"pending","priority":"medium"},
            ])),
            ("tool", tr("TodoWrite","Todos updated")),
            ("assistant", "Plan creado: arreglar tests (alta) y resolver TODOs (media)."),
        ))
    # Parallel reads
    for _ in range(n // 10):
        f1, f2 = RS(PY, 2)
        results.append(build(
            ("user", f"Compara {f1} y {f2}"),
            ("assistant", tc("Read", file_path=f1) + "\n" + tc("Read", file_path=f2)),
            ("tool", tr("Read", R(PY_BODIES))),
            ("tool", tr("Read", "def parse(text: str) -> str:\n    return text.strip().lower()\n")),
            ("assistant", f"`{f1}` es más complejo (clases/funciones async). `{f2}` es una utilidad simple."),
        ))
    # Security grep chain
    for _ in range(n // 10):
        results.append(build(
            ("user", "Haz un audit de seguridad básico del código"),
            ("assistant", tc("Grep", pattern="eval\\(|exec\\(|os\\.system", path=".", include="*.py")),
            ("tool", tr("Grep", "")),
            ("assistant", tc("Grep", pattern="SELECT.*\\+|execute.*'.*\\+", path=".", include="*.py")),
            ("tool", tr("Grep", f"src/legacy.py:34:db.execute('SELECT * FROM users WHERE id='+str(id))")),
            ("assistant", tc("Grep", pattern="SECRET.*=.*['\"][^'\"]{5}", path=".", include="*.py")),
            ("tool", tr("Grep", "config.py:12:JWT_SECRET = 'hardcoded_key_123'")),
            ("assistant", "Audit:\n- ✅ Sin `eval()`/`exec()` peligrosos\n- ⚠️ SQL injection en `legacy.py:34`\n- ⚠️ Secret hardcodeado en `config.py:12`"),
        ))
    # Debug fix chain
    for _ in range(n // 10):
        f = R(PY)
        results.append(build(
            ("user", f"Hay un bug de ZeroDivisionError en {f}, corrígelo"),
            ("assistant", tc("Read", file_path=f)),
            ("tool", tr("Read", f"def average(items):\n    return sum(items) / len(items)  # bug: empty list\n\ndef process(data):\n    return average(data)")),
            ("assistant", tc("Edit", file_path=f, old_string="    return sum(items) / len(items)  # bug: empty list", new_string="    if not items:\n        return 0.0\n    return sum(items) / len(items)")),
            ("tool", tr("Edit","Edit applied")),
            ("assistant", f"Bug corregido en `{f}`: `average()` retorna 0.0 para listas vacías."),
        ))
    # Overview chain
    for _ in range(n // 10):
        results.append(build(
            ("user", "Dame un overview del proyecto"),
            ("assistant", tc("Bash", command="find . -name '*.py' | wc -l")),
            ("tool", tr("Bash", f"  {random.randint(15,50)}")),
            ("assistant", tc("Bash", command="python -m pytest tests/ --tb=no -q 2>&1 | tail -1")),
            ("tool", tr("Bash", f"{random.randint(15,30)} passed in {random.uniform(0.5,3.0):.2f}s")),
            ("assistant", tc("TodoRead")),
            ("tool", tr("TodoRead", json.dumps([{"id":"1","content":"Deploy staging","status":"pending","priority":"high"}]))),
            ("assistant", f"Overview: ~{random.randint(20,50)} archivos Python, tests pasando, 1 tarea pendiente."),
        ))
    return results
def gen_no_tool(n=60) -> List[Dict]:
    results = []
    qa = [
        ("¿Qué es un decorador en Python?","Un decorador es una función que envuelve a otra para extender su comportamiento sin modificarla. Se aplica con `@decorator` antes de la definición."),
        ("¿Diferencia entre async def y def?","`async def` define una coroutine que puede usar `await`. `def` define una función síncrona bloqueante. Las coroutines requieren un event loop."),
        ("¿Qué hace useEffect en React?","`useEffect` ejecuta efectos secundarios tras el render. Array de dependencias vacío `[]` = solo al montar; `[x]` = cuando `x` cambia."),
        ("¿REST vs GraphQL?","**REST**: endpoints fijos por recurso. **GraphQL**: un endpoint, el cliente define qué campos necesita — evita over/under-fetching."),
        ("¿Cómo funciona JWT?","JWT tiene 3 partes base64: header (algoritmo), payload (datos), signature (verificación). El servidor valida sin almacenar sesión."),
        ("¿Qué es Docker?","Docker empaqueta apps con sus dependencias en contenedores aislados. Garantiza consistencia entre dev, staging y prod."),
        ("¿Cuándo usar useReducer vs useState?","`useState` para estado simple. `useReducer` cuando el estado es complejo (objetos con múltiples campos) o la lógica de actualización es elaborada."),
        ("¿Diferencia entre == y === en JavaScript?","`==` compara con coerción de tipos (`'5' == 5` → true). `===` compara valor Y tipo. Siempre prefiere `===`."),
        ("¿Qué es Big O?","Describe cómo crece el tiempo/espacio de un algoritmo. O(1) constante, O(log n) logarítmico, O(n) lineal, O(n²) cuadrático."),
        ("¿Qué es el GIL en Python?","El Global Interpreter Lock impide ejecución paralela de threads en CPU-bound. No afecta I/O-bound con asyncio."),
        ("¿Qué es SOLID?","5 principios de diseño: Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion."),
        ("¿Qué es CORS?","Cross-Origin Resource Sharing: política de seguridad del browser que bloquea peticiones a dominios distintos. Se habilita en el servidor con headers `Access-Control-Allow-*`."),
        ("¿SQL vs NoSQL?","**SQL**: datos relacionales, ACID, esquema fijo. **NoSQL**: escalabilidad horizontal, esquema flexible, para documentos/grafos/clave-valor."),
        ("¿Qué es WebSockets?","Comunicación bidireccional en tiempo real sobre conexión persistente. El servidor puede enviar datos sin que el cliente pregunte (a diferencia de HTTP)."),
        ("¿Qué es un índice en BD?","Estructura que acelera consultas al costo de más espacio y lentitud en escrituras. Ideal para columnas de búsqueda frecuente."),
        ("Hola, ¿en qué puedes ayudarme?","Puedo ayudarte con tu código: leer y editar archivos, ejecutar comandos, buscar bugs, instalar dependencias y más. ¿En qué proyecto trabajas?"),
        ("¿Qué es el patrón Repository?","Separa la lógica de acceso a datos del dominio. El repositorio provee una interfaz de colección para acceder a entidades sin exponer detalles de la BD."),
        ("¿Qué es idempotencia en APIs?","Una operación idempotente produce el mismo resultado sin importar cuántas veces se llame. GET, PUT y DELETE son idempotentes; POST no."),
        ("¿Diferencia entre stack y queue?","**Stack** (LIFO): el último en entrar es el primero en salir — ej: call stack. **Queue** (FIFO): el primero en entrar es el primero en salir — ej: colas de mensajes."),
        ("¿Qué es memoización?","Técnica de optimización que cachea los resultados de funciones costosas para evitar recalcularlos. `functools.lru_cache` en Python, `useMemo` en React."),
    ]
    wraps = ["{q}", "Explícame: {q}", "Tengo una duda — {q}", "Pregunta rápida: {q}"]
    per = max(1, n // len(qa))
    for user_msg, resp in qa:
        for _ in range(per):
            results.append(build(("user", R(wraps).format(q=user_msg)), ("assistant", resp)))
    return results
# ────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────
def main():
    print("🔧 Generando dataset Claude Code tool-calling...")
    pool = []
    pool += gen_bash_ls(90)
    pool += gen_bash_git(90)
    pool += gen_bash_run(80)
    pool += gen_read(110)
    pool += gen_write(70)
    pool += gen_edit(90)
    pool += gen_multiedit(60)
    pool += gen_glob(80)
    pool += gen_grep(80)
    pool += gen_ls(60)
    pool += gen_webfetch(50)
    pool += gen_websearch(50)
    pool += gen_todo(60)
    pool += gen_notebook(50)
    pool += gen_chains(120)
    pool += gen_no_tool(80)
    print(f"   Total generados : {len(pool)}")
    # Deduplicar por hash de primeros 400 chars
    seen, unique = set(), []
    for item in pool:
        key = hashlib.md5(item["text"].encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    print(f"   Únicos          : {len(unique)}")
    random.shuffle(unique)
    TARGET_TRAIN, TARGET_VALID = 900, 100
    if len(unique) >= TARGET_TRAIN + TARGET_VALID:
        train = unique[:TARGET_TRAIN]
        valid = unique[TARGET_TRAIN:TARGET_TRAIN + TARGET_VALID]
    else:
        split = int(len(unique) * 0.9)
        train = unique[:split]
        valid = unique[split:]
    import os
    os.makedirs("output", exist_ok=True)
    with open("output/train.jsonl", "w", encoding="utf-8") as f:
        for item in train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open("output/valid.jsonl", "w", encoding="utf-8") as f:
        for item in valid:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    ts = os.path.getsize("output/train.jsonl")
    vs = os.path.getsize("output/valid.jsonl")
    avg = sum(len(e["text"]) for e in train) / max(len(train), 1)
    print(f"\n✅ Dataset listo en ./output/")
    print(f"   train.jsonl : {len(train):4d} ejemplos  ({ts/1024:.0f} KB)")
    print(f"   valid.jsonl : {len(valid):4d} ejemplos  ({vs/1024:.0f} KB)")
    print(f"   Avg length  : {avg:.0f} chars  (~{avg/4:.0f} tokens)")
    tools = ["Bash","Read","Write","Edit","MultiEdit","Glob","Grep","LS",
             "WebFetch","WebSearch","TodoRead","TodoWrite","NotebookRead","NotebookEdit"]
    print("\n📊 Distribución en train:")
    for t in tools:
        c = sum(1 for e in train if f'"name": "{t}"' in e["text"])
        bar = "█" * (c // 5)
        print(f"   {t:<15} {c:4d}  {bar}")
    no_tool = sum(1 for e in train if "<|im_start|>assistant\n<tool_call>" not in e["text"])
    print(f"   {'No-tool':<15} {no_tool:4d}  {'█' * (no_tool // 5)}")
if __name__ == "__main__":
    main()
