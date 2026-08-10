"""
Benchmark de Tool Calling para Qwen3-8B
Mide la capacidad del modelo de usar herramientas correctamente.
"""
import json
import re
import sys
import time
from datetime import datetime

SYSTEM_PROMPT = """Eres un agente de programación con acceso a herramientas del sistema.
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

TEST_CASES = [
    {"id": "bash_1",    "prompt": "Ejecuta ls -la en el directorio actual",
     "expected_tool": "Bash", "expected_args_keys": ["command"]},
    {"id": "bash_2",    "prompt": "¿Cuántos archivos Python hay en el proyecto?",
     "expected_tool": "Bash", "expected_args_keys": ["command"]},
    {"id": "read_1",    "prompt": "Lee el archivo src/main.py",
     "expected_tool": "Read", "expected_args_keys": ["file_path"]},
    {"id": "read_2",    "prompt": "Muéstrame el contenido de config.json",
     "expected_tool": "Read", "expected_args_keys": ["file_path"]},
    {"id": "write_1",   "prompt": "Crea un archivo llamado hello.py con un print de Hola Mundo",
     "expected_tool": "Write", "expected_args_keys": ["file_path", "content"]},
    {"id": "edit_1",    "prompt": "Actualiza la versión de 1.0.0 a 2.0.0 en package.json (léelo primero)",
     "expected_tool": "Read", "expected_args_keys": ["file_path"]},
    {"id": "grep_1",    "prompt": "¿Dónde se importa axios en el proyecto?",
     "expected_tool": "Grep", "expected_args_keys": ["pattern"]},
    {"id": "grep_2",    "prompt": "Busca todos los TODO en archivos .py",
     "expected_tool": "Grep", "expected_args_keys": ["pattern"]},
    {"id": "glob_1",    "prompt": "Lista todos los archivos TypeScript del proyecto",
     "expected_tool": "Glob", "expected_args_keys": ["pattern"]},
    {"id": "glob_2",    "prompt": "Encuentra todos los archivos de configuración .env",
     "expected_tool": "Glob", "expected_args_keys": ["pattern"]},
    {"id": "ls_1",      "prompt": "Muéstrame qué hay en la carpeta components/",
     "expected_tool": "LS", "expected_args_keys": ["path"]},
    {"id": "ls_2",      "prompt": "¿Qué archivos tiene el directorio tests?",
     "expected_tool": "LS", "expected_args_keys": ["path"]},
    {"id": "web_1",     "prompt": "Busca ejemplos de uso de React hooks",
     "expected_tool": "WebSearch", "expected_args_keys": ["query"]},
    {"id": "nb_1",      "prompt": "¿Qué contiene analysis.ipynb?",
     "expected_tool": "NotebookRead", "expected_args_keys": ["notebook_path"]},
    {"id": "direct_1",  "prompt": "¿Qué es una función recursiva?",
     "no_tool_needed": True, "expected_tool": None},
    {"id": "direct_2",  "prompt": "¿Cuál es la diferencia entre == y === en JavaScript?",
     "no_tool_needed": True, "expected_tool": None},
]

def parsear_tool_call(texto):
    patron = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = re.findall(patron, texto, re.DOTALL)
    if not matches:
        return None, None, None
    raw_call = matches[0].strip()
    try:
        call_data = json.loads(raw_call)
        nombre = call_data.get("name", "")
        argumentos = call_data.get("arguments", {})
        return raw_call, nombre, argumentos
    except json.JSONDecodeError:
        return raw_call, None, None

def evaluar_caso(output, caso):
    resultado = {
        "id": caso["id"],
        "prompt": caso["prompt"],
        "output_raw": output[:300] + "..." if len(output) > 300 else output,
        "tool_llamada": False,
        "tool_correcta": False,
        "json_valido": False,
        "args_correctos": False,
        "comportamiento_correcto": False,
        "tool_detectada": None,
        "args_detectados": None,
        "error": None,
    }
    raw_call, tool_name, args = parsear_tool_call(output)
    resultado["tool_llamada"] = raw_call is not None
    resultado["tool_detectada"] = tool_name
    resultado["args_detectados"] = list(args.keys()) if args else []

    if raw_call:
        try:
            json.loads(raw_call)
            resultado["json_valido"] = True
        except Exception:
            resultado["json_valido"] = False
            resultado["error"] = "JSON inválido en tool_call"

    if caso.get("no_tool_needed"):
        resultado["comportamiento_correcto"] = not resultado["tool_llamada"]
        resultado["tool_correcta"] = not resultado["tool_llamada"]
        resultado["args_correctos"] = not resultado["tool_llamada"]
        return resultado

    if not resultado["tool_llamada"]:
        resultado["error"] = f"No emitió tool_call (esperado: {caso['expected_tool']})"
        return resultado

    if tool_name:
        resultado["tool_correcta"] = (tool_name == caso["expected_tool"])
        if not resultado["tool_correcta"]:
            resultado["error"] = f"Herramienta incorrecta: usó '{tool_name}', esperaba '{caso['expected_tool']}'"

    if args and "expected_args_keys" in caso:
        expected_keys = set(caso["expected_args_keys"])
        actual_keys = set(args.keys())
        resultado["args_correctos"] = expected_keys.issubset(actual_keys)
        if not resultado["args_correctos"]:
            faltantes = expected_keys - actual_keys
            resultado["error"] = f"Args faltantes: {faltantes}"

    resultado["comportamiento_correcto"] = (
        resultado["tool_llamada"] and
        resultado["json_valido"] and
        resultado["tool_correcta"] and
        resultado["args_correctos"]
    )
    return resultado

def calcular_metricas(resultados):
    total = len(resultados)
    casos_con_tool = [r for r in resultados if not any(
        c["id"] == r["id"] and c.get("no_tool_needed") for c in TEST_CASES
    )]
    casos_sin_tool = [r for r in resultados if any(
        c["id"] == r["id"] and c.get("no_tool_needed") for c in TEST_CASES
    )]
    return {
        "total_casos": total,
        "comportamiento_correcto": sum(1 for r in resultados if r["comportamiento_correcto"]),
        "tool_call_emitida": sum(1 for r in casos_con_tool if r["tool_llamada"]),
        "total_con_tool": len(casos_con_tool),
        "tool_nombre_correcto": sum(1 for r in casos_con_tool if r["tool_correcta"]),
        "json_valido": sum(1 for r in casos_con_tool if r["json_valido"] and r["tool_llamada"]),
        "args_correctos": sum(1 for r in casos_con_tool if r["args_correctos"]),
        "no_hallucination": sum(1 for r in casos_sin_tool if not r["tool_llamada"]),
        "total_sin_tool": len(casos_sin_tool),
    }

def imprimir_reporte(metricas, resultados, nombre_modelo, tiempo_total):
    t = metricas["total_con_tool"]
    tcr = metricas["tool_call_emitida"] / t * 100 if t > 0 else 0
    tna = metricas["tool_nombre_correcto"] / t * 100 if t > 0 else 0
    jva = metricas["json_valido"] / t * 100 if t > 0 else 0
    aca = metricas["args_correctos"] / t * 100 if t > 0 else 0
    nhr = metricas["no_hallucination"] / metricas["total_sin_tool"] * 100 if metricas["total_sin_tool"] > 0 else 100
    score = metricas["comportamiento_correcto"] / metricas["total_casos"] * 100

    separador = "=" * 60
    print(f"\n{separador}")
    print(f"  REPORTE DE BENCHMARK — {nombre_modelo}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(separador)
    print(f"\n  METRICAS PRINCIPALES")
    print(f"  {'-'*56}")
    print(f"  TCR  Tool Call Rate       {tcr:6.1f}%   ({metricas['tool_call_emitida']}/{t})")
    print(f"  TNA  Tool Name Accuracy   {tna:6.1f}%   ({metricas['tool_nombre_correcto']}/{t})")
    print(f"  JVA  JSON Valid Args      {jva:6.1f}%   ({metricas['json_valido']}/{t})")
    print(f"  ACA  Args Keys Accuracy   {aca:6.1f}%   ({metricas['args_correctos']}/{t})")
    print(f"  NHR  No Hallucination     {nhr:6.1f}%   ({metricas['no_hallucination']}/{metricas['total_sin_tool']})")
    print(f"  {'-'*56}")
    print(f"  SCORE FINAL              {score:6.1f}%   ({metricas['comportamiento_correcto']}/{metricas['total_casos']})")
    print(f"\n  Tiempo de evaluación: {tiempo_total:.1f}s  ({tiempo_total/metricas['total_casos']:.1f}s/caso)")

    print(f"\n  DETALLE POR CASO")
    print(f"  {'-'*56}")
    for r in resultados:
        estado = "OK" if r["comportamiento_correcto"] else "FAIL"
        tool_info = f"-> {r['tool_detectada']}" if r["tool_detectada"] else "-> (sin herramienta)"
        print(f"  [{estado:<4}] [{r['id']:<10}] {tool_info}")
        if r.get("error"):
            print(f"          {r['error']}")
    print(f"\n{separador}\n")

    return {
        "modelo": nombre_modelo,
        "timestamp": datetime.now().isoformat(),
        "score_final": round(score, 2),
        "metricas": {
            "TCR": round(tcr, 2),
            "TNA": round(tna, 2),
            "JVA": round(jva, 2),
            "ACA": round(aca, 2),
            "NHR": round(nhr, 2),
        },
        "detalle": resultados,
    }

def ejecutar_benchmark(model_path, adapter_path=None, nombre_modelo="Modelo"):
    from mlx_lm import load, generate

    print(f"\nCargando {nombre_modelo}...")
    modelo, tokenizer = load(model_path, adapter_path=adapter_path)
    print(f"Modelo cargado\n")

    resultados = []
    inicio_total = time.time()

    for i, caso in enumerate(TEST_CASES, 1):
        print(f"  [{i:02d}/{len(TEST_CASES)}] Evaluando: {caso['prompt'][:60]}...")
        prompt_formateado = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{caso['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inicio_caso = time.time()
        output = generate(
            modelo,
            tokenizer,
            prompt=prompt_formateado,
            max_tokens=256,
            verbose=False,
        )
        tiempo_caso = time.time() - inicio_caso

        resultado = evaluar_caso(output, caso)
        resultado["tiempo_segundos"] = round(tiempo_caso, 2)
        resultados.append(resultado)

        estado = "OK" if resultado["comportamiento_correcto"] else "FAIL"
        print(f"       [{estado}] {resultado.get('tool_detectada', '(sin herramienta)')} [{tiempo_caso:.1f}s]")

    tiempo_total = time.time() - inicio_total
    metricas = calcular_metricas(resultados)
    reporte = imprimir_reporte(metricas, resultados, nombre_modelo, tiempo_total)
    return reporte

if __name__ == "__main__":
    modelo_path = sys.argv[1] if len(sys.argv) > 1 else "./modelo-base"
    adapter = sys.argv[2] if len(sys.argv) > 2 else None
    adapter = adapter if adapter else None  # "" -> None
    nombre = sys.argv[3] if len(sys.argv) > 3 else "Modelo"

    reporte = ejecutar_benchmark(modelo_path, adapter, nombre)

    import os
    os.makedirs("resultados", exist_ok=True)
    output_file = f"resultados/benchmark_{nombre.replace(' ', '_').lower()}.json"
    with open(output_file, "w") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    print(f"Reporte guardado en: {output_file}")
