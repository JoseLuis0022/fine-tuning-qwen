"""
Genera reporte comparativo de los benchmarks pre y post fine-tuning.
"""
import glob
import json
import sys


def cargar_ultimo_benchmark(patron):
    archivos = glob.glob(f"resultados/benchmark_{patron}*.json")
    if not archivos:
        return None
    return json.load(open(sorted(archivos)[-1]))


base = cargar_ultimo_benchmark("qwen3-8b-base")
ft = cargar_ultimo_benchmark("qwen3-8b-finetuned")

if not base or not ft:
    print("No se encontraron los archivos de benchmark.")
    print("Asegurate de haber ejecutado las fases 5 y 7.")
    sys.exit(1)

sep = "=" * 65
sep2 = "-" * 65
print(f"\n{sep}")
print("  COMPARATIVA: ANTES vs DESPUES DEL FINE-TUNING")
print(sep)
print(f"\n  {'Metrica':<45} {'BASE':>8} {'FT':>8} {'DELTA':>8}")
print(f"  {sep2}")

metricas_nombres = {
    "TCR": "Tool Call Rate",
    "TNA": "Tool Name Accuracy",
    "JVA": "JSON Valid Args",
    "ACA": "Args Keys Accuracy",
    "NHR": "No Hallucination",
}
for key, nombre in metricas_nombres.items():
    b = base["metricas"][key]
    f = ft["metricas"][key]
    delta = f - b
    signo = "+" if delta >= 0 else ""
    print(f"  {nombre:<45} {b:>7.1f}% {f:>7.1f}% {signo}{delta:>6.1f}%")

b_score = base["score_final"]
f_score = ft["score_final"]
delta_score = f_score - b_score
print(f"  {sep2}")
signo = "+" if delta_score >= 0 else ""
print(f"  {'SCORE FINAL':<45} {b_score:>7.1f}% {f_score:>7.1f}% {signo}{delta_score:>6.1f}%")
print(f"\n{sep}")

print("\n  VEREDICTO")
print(f"  {sep2}")
if delta_score >= 20:
    print("  EXCELENTE: El fine-tuning mejoro significativamente el tool calling.")
    print(f"  El modelo gano {delta_score:.1f} puntos porcentuales.")
elif delta_score >= 10:
    print("  BUENO: El fine-tuning mejoro el tool calling de forma notable.")
    print(f"  Mejora de {delta_score:.1f} puntos.")
elif delta_score >= 0:
    print("  LEVE: La mejora es pequena.")
else:
    print("  REGRESION: El modelo fine-tuneado empeoro.")

reporte = {
    "fecha": ft.get("timestamp", ""),
    "base": base,
    "finetuned": ft,
    "delta": {k: round(ft["metricas"][k] - base["metricas"][k], 2) for k in base["metricas"]},
    "delta_score": round(delta_score, 2),
}
with open("resultados/reporte_final.json", "w") as f_out:
    json.dump(reporte, f_out, indent=2, ensure_ascii=False)
print(f"\n  Reporte completo guardado en: resultados/reporte_final.json")
print(f"{sep}\n")
