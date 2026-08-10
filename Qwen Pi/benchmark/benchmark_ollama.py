"""
Benchmark del modelo DESPLEGADO en Ollama (GGUF Q4_K_M + template).
Reutiliza los mismos 16 casos y la misma lógica de evaluación que benchmark.py,
pero en vez de cargar el modelo con MLX, llama a la API de Ollama tal como lo
haría Claude Code: solo envía el prompt del usuario y deja que el SYSTEM + el
TEMPLATE del Modelfile hagan su trabajo.
"""
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import (
    TEST_CASES,
    evaluar_caso,
    calcular_metricas,
    imprimir_reporte,
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3-toolcalling"
NOMBRE = sys.argv[2] if len(sys.argv) > 2 else "Qwen3-8B-Ollama-Q4"


def gen_ollama(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 256, "temperature": 0.1},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["response"]


def main():
    print(f"\nBenchmark via Ollama -> modelo '{MODEL}'")
    resultados = []
    inicio_total = time.time()

    for i, caso in enumerate(TEST_CASES, 1):
        print(f"  [{i:02d}/{len(TEST_CASES)}] {caso['prompt'][:55]}...")
        t0 = time.time()
        # Solo el prompt del usuario: el SYSTEM y el formato los pone el Modelfile
        output = gen_ollama(caso["prompt"])
        dt = time.time() - t0
        r = evaluar_caso(output, caso)
        r["tiempo_segundos"] = round(dt, 2)
        resultados.append(r)
        estado = "OK" if r["comportamiento_correcto"] else "FAIL"
        print(f"       [{estado}] {r.get('tool_detectada', '(sin herramienta)')} [{dt:.1f}s]")

    tiempo_total = time.time() - inicio_total
    metricas = calcular_metricas(resultados)
    reporte = imprimir_reporte(metricas, resultados, NOMBRE, tiempo_total)

    os.makedirs("resultados", exist_ok=True)
    out = f"resultados/benchmark_{NOMBRE.replace(' ', '_').lower()}.json"
    with open(out, "w") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    print(f"Reporte guardado en: {out}")


if __name__ == "__main__":
    main()
