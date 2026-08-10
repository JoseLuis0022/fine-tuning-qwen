"""
Benchmark de tool-calling para el protocolo de Pi (6 herramientas reales).
Usa el MISMO render que el dataset: apply_chat_template(messages, tools=TOOLS) con la
plantilla nativa de Qwen3, de modo que mide exactamente lo que verá Pi en producción.
Carga el modelo con MLX (base o base+adapter) y evalúa los 16 casos en jerga mexicana.
"""
import json, os, re, sys, time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS = json.load(open(os.path.join(ROOT, "pi_tools.json")))

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

# expected_tool = herramienta esperada; expected_args_keys = claves requeridas
TEST_CASES = [
    {"id": "read_1",  "prompt": "léeme src/main.py porfa",
     "expected_tool": "read", "expected_args_keys": ["path"]},
    {"id": "read_2",  "prompt": "oye chécame qué hay en config.json pls",
     "expected_tool": "read", "expected_args_keys": ["path"]},
    {"id": "bash_ls", "prompt": "a ver, ¿qué archivos hay en la carpeta src/?",
     "expected_tool": "bash", "expected_args_keys": ["command"]},
    {"id": "bash_find", "prompt": "oye porfa, ¿cuántos archivos .py hay en el proyecto?",
     "expected_tool": "bash", "expected_args_keys": ["command"]},
    {"id": "bash_grep", "prompt": "¿dónde se importa axios en el código? porfa",
     "expected_tool": "bash", "expected_args_keys": ["command"]},
    {"id": "bash_run", "prompt": "córreme los tests porfa",
     "expected_tool": "bash", "expected_args_keys": ["command"]},
    {"id": "bash_fileops", "prompt": "créame la carpeta logs porfa",
     "expected_tool": "bash", "expected_args_keys": ["command"]},
    {"id": "bash_git", "prompt": "oye chécame cómo va el repo (git status) pls",
     "expected_tool": "bash", "expected_args_keys": ["command"]},
    {"id": "write_1", "prompt": "créame un archivo hello.py con un print de Hola Mundo porfa",
     "expected_tool": "write", "expected_args_keys": ["path", "content"]},
    {"id": "write_2", "prompt": "hazme un .gitignore para Python pls",
     "expected_tool": "write", "expected_args_keys": ["path", "content"]},
    {"id": "edit_1", "prompt": "actívame el modo debug en config.json porfa",
     "expected_tool": "edit", "expected_args_keys": ["path", "edits"]},
    {"id": "edit_2", "prompt": "súbeme la versión a 2.0.0 en package.json pls",
     "expected_tool": "edit", "expected_args_keys": ["path", "edits"]},
    {"id": "websearch_1", "prompt": "oye porfa búscame en internet las novedades de Python 2026",
     "expected_tool": "web_search", "expected_args_keys": ["query"]},
    {"id": "webfetch_1", "prompt": "a ver, entra a https://docs.ollama.com y resúmemelo porfa",
     "expected_tool": "web_fetch", "expected_args_keys": ["url"]},
    {"id": "direct_1", "prompt": "¿qué es un closure? porfa",
     "no_tool_needed": True, "expected_tool": None},
    {"id": "direct_2", "prompt": "oye, ¿cuál es la diferencia entre == y === en JavaScript?",
     "no_tool_needed": True, "expected_tool": None},
]

_THINK_RE = re.compile(r"<think>\s*</think>\s*", re.DOTALL)


def parsear_tool_call(texto):
    m = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', texto, re.DOTALL)
    if not m:
        return None, None, None
    raw = m[0].strip()
    try:
        d = json.loads(raw)
        return raw, d.get("name", ""), d.get("arguments", {})
    except json.JSONDecodeError:
        return raw, None, None


def evaluar_caso(output, caso):
    r = {"id": caso["id"], "prompt": caso["prompt"],
         "output_raw": output[:300] + "..." if len(output) > 300 else output,
         "tool_llamada": False, "tool_correcta": False, "json_valido": False,
         "args_correctos": False, "comportamiento_correcto": False,
         "tool_detectada": None, "args_detectados": None, "error": None}
    raw, name, args = parsear_tool_call(output)
    r["tool_llamada"] = raw is not None
    r["tool_detectada"] = name
    r["args_detectados"] = list(args.keys()) if args else []
    if raw:
        try:
            json.loads(raw); r["json_valido"] = True
        except Exception:
            r["error"] = "JSON inválido"
    if caso.get("no_tool_needed"):
        r["comportamiento_correcto"] = not r["tool_llamada"]
        r["tool_correcta"] = not r["tool_llamada"]
        r["args_correctos"] = not r["tool_llamada"]
        return r
    if not r["tool_llamada"]:
        r["error"] = f"No emitió tool_call (esperado: {caso['expected_tool']})"
        return r
    if name:
        r["tool_correcta"] = (name == caso["expected_tool"])
        if not r["tool_correcta"]:
            r["error"] = f"Herramienta incorrecta: usó '{name}', esperaba '{caso['expected_tool']}'"
    if args and "expected_args_keys" in caso:
        exp = set(caso["expected_args_keys"])
        r["args_correctos"] = exp.issubset(set(args.keys()))
        if not r["args_correctos"]:
            r["error"] = f"Args faltantes: {exp - set(args.keys())}"
    r["comportamiento_correcto"] = (r["tool_llamada"] and r["json_valido"]
                                    and r["tool_correcta"] and r["args_correctos"])
    return r


def calcular_metricas(resultados):
    con = [r for r in resultados if not any(c["id"] == r["id"] and c.get("no_tool_needed") for c in TEST_CASES)]
    sin = [r for r in resultados if any(c["id"] == r["id"] and c.get("no_tool_needed") for c in TEST_CASES)]
    return {
        "total_casos": len(resultados),
        "comportamiento_correcto": sum(1 for r in resultados if r["comportamiento_correcto"]),
        "tool_call_emitida": sum(1 for r in con if r["tool_llamada"]),
        "total_con_tool": len(con),
        "tool_nombre_correcto": sum(1 for r in con if r["tool_correcta"]),
        "json_valido": sum(1 for r in con if r["json_valido"] and r["tool_llamada"]),
        "args_correctos": sum(1 for r in con if r["args_correctos"]),
        "no_hallucination": sum(1 for r in sin if not r["tool_llamada"]),
        "total_sin_tool": len(sin),
    }


def imprimir_reporte(m, resultados, nombre, tiempo):
    t = m["total_con_tool"]
    tcr = m["tool_call_emitida"] / t * 100 if t else 0
    tna = m["tool_nombre_correcto"] / t * 100 if t else 0
    jva = m["json_valido"] / t * 100 if t else 0
    aca = m["args_correctos"] / t * 100 if t else 0
    nhr = m["no_hallucination"] / m["total_sin_tool"] * 100 if m["total_sin_tool"] else 100
    score = m["comportamiento_correcto"] / m["total_casos"] * 100
    sep = "=" * 60
    print(f"\n{sep}\n  REPORTE BENCHMARK PI — {nombre}\n  {datetime.now():%Y-%m-%d %H:%M:%S}\n{sep}")
    print(f"\n  TCR  Tool Call Rate      {tcr:6.1f}%   ({m['tool_call_emitida']}/{t})")
    print(f"  TNA  Tool Name Accuracy  {tna:6.1f}%   ({m['tool_nombre_correcto']}/{t})")
    print(f"  JVA  JSON Valid Args     {jva:6.1f}%   ({m['json_valido']}/{t})")
    print(f"  ACA  Args Keys Accuracy  {aca:6.1f}%   ({m['args_correctos']}/{t})")
    print(f"  NHR  No Hallucination    {nhr:6.1f}%   ({m['no_hallucination']}/{m['total_sin_tool']})")
    print(f"  {'-'*56}\n  SCORE FINAL             {score:6.1f}%   ({m['comportamiento_correcto']}/{m['total_casos']})")
    print(f"\n  Tiempo: {tiempo:.1f}s ({tiempo/m['total_casos']:.1f}s/caso)\n\n  DETALLE:")
    for r in resultados:
        est = "OK  " if r["comportamiento_correcto"] else "FAIL"
        ti = f"-> {r['tool_detectada']}" if r["tool_detectada"] else "-> (sin tool)"
        print(f"  [{est}] [{r['id']:<12}] {ti}")
        if r.get("error"):
            print(f"          {r['error']}")
    print(sep)
    return {"modelo": nombre, "timestamp": datetime.now().isoformat(),
            "score_final": round(score, 2),
            "metricas": {"TCR": round(tcr, 2), "TNA": round(tna, 2), "JVA": round(jva, 2),
                         "ACA": round(aca, 2), "NHR": round(nhr, 2)},
            "detalle": resultados}


def construir_prompt(tokenizer, user_prompt):
    msgs = [{"role": "system", "content": PI_SYS}, {"role": "user", "content": user_prompt}]
    p = tokenizer.apply_chat_template(msgs, tools=TOOLS, tokenize=False, add_generation_prompt=True)
    return _THINK_RE.sub("", p)


def ejecutar(model_path, adapter_path=None, nombre="Modelo"):
    from mlx_lm import load, generate
    print(f"\nCargando {nombre} (adapter={adapter_path}) ...")
    model, tokenizer = load(model_path, adapter_path=adapter_path)
    print("Modelo cargado\n")
    resultados = []
    t0 = time.time()
    for i, caso in enumerate(TEST_CASES, 1):
        print(f"  [{i:02d}/{len(TEST_CASES)}] {caso['prompt'][:55]}...")
        prompt = construir_prompt(tokenizer, caso["prompt"])
        ti = time.time()
        out = generate(model, tokenizer, prompt=prompt, max_tokens=256, verbose=False)
        dt = time.time() - ti
        r = evaluar_caso(out, caso)
        r["tiempo_segundos"] = round(dt, 2)
        resultados.append(r)
        print(f"       [{'OK' if r['comportamiento_correcto'] else 'FAIL'}] {r.get('tool_detectada') or '(sin tool)'} [{dt:.1f}s]")
    return imprimir_reporte(calcular_metricas(resultados), resultados, nombre, time.time() - t0)


if __name__ == "__main__":
    mp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "modelo-base")
    ap = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
    nombre = sys.argv[3] if len(sys.argv) > 3 else "Pi-Modelo"
    rep = ejecutar(mp, ap, nombre)
    os.makedirs(os.path.join(ROOT, "resultados"), exist_ok=True)
    outf = os.path.join(ROOT, "resultados", f"benchmark_{nombre.replace(' ', '_').lower()}.json")
    json.dump(rep, open(outf, "w"), indent=2, ensure_ascii=False)
    print(f"\nGuardado: {outf}")
