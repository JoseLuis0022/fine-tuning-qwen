# ESPEC_PI.md — Especificación real de Pi (capturada en vivo)

Fuente: captura del payload real que **Pi v0.79.10** envía a Ollama vía proxy, + código
compilado en `/opt/homebrew/lib/node_modules/@earendil-works/pi-coding-agent/dist/`.
Esta es la fuente de verdad (B) del plan: lo que realmente recibe el modelo a través del launcher.

## 1. Cómo se conecta Pi a Ollama
- API: **`openai-completions`** (OpenAI-compatible) contra `http://127.0.0.1:11434/v1`.
- Usa **function calling estándar**: manda el array `tools` en el request; el modelo emite
  `<tool_call>{...}</tool_call>` en texto y **Ollama lo parsea** a `tool_calls` estructurados.
- El system prompt va con role **`developer`** (equivalente a system en OpenAI nuevo).
- Manda `reasoning_effort: "medium"` (Qwen3 es modelo de razonamiento). Para velocidad,
  entrenamos SIN bloques `<think>` (respuesta directa al tool_call).

## 2. Las 6 herramientas REALES (no 7) — esquemas exactos
Por defecto Pi expone **solo 4 nativas + 2 web** (paquete `@ollama/pi-web-search` ya instalado).
**NO** hay tools separadas `grep`/`find`/`ls`: listar/buscar/contar archivos se hace con **`bash`**
(ls, rg, find, wc). El system prompt lo dice: *"Use bash for file operations like ls, rg, find"*.

| Tool | Requeridos | Propiedades (tipo) |
|------|-----------|--------------------|
| `read` | `path` | `path`(string), `offset`(number), `limit`(number) |
| `bash` | `command` | `command`(string), `timeout`(number) |
| `edit` | `path`, `edits` | `path`(string), `edits`(array de `{oldText, newText}`) |
| `write` | `path`, `content` | `path`(string), `content`(string) |
| `web_search` | `query` | `query`(string), `max_results`(number, def 5) |
| `web_fetch` | `url` | `url`(string) |

Diferencias críticas vs Claude Code (el fine-tune anterior):
- `path` (no `file_path`); nombres en minúscula.
- `edit` usa `edits: [{"oldText":..., "newText":...}]` (no `old_string`/`new_string`, no `MultiEdit`).
- Sin `Glob`/`Grep`/`LS`/`Task`/`Todo*`/`Notebook*`. Búsqueda/listado → `bash`.

Esquemas JSON completos guardados en `pi_tools.json`. System prompt real en `pi_system_prompt.txt`.

## 3. System prompt de Pi (núcleo estable)
El developer prompt real incluye: rol ("expert coding assistant operating inside pi"),
lista de tools read/bash/edit/write, guidelines de uso, + secciones VOLÁTILES (rutas absolutas
de docs de pi, lista de skills instaladas, fecha y cwd actuales). Para el dataset usamos el
**núcleo estable** (rol + tools + guidelines) y omitimos lo volátil (skills/paths/fecha) para
no sobreajustar. El bloque `# Tools` con los esquemas lo añade la PLANTILLA, no el system.

## 4. Formato de wire EXACTO (plantilla Go de Ollama para qwen3)
Capturado con `ollama show qwen3:8b --template`. El dataset debe replicar esto literalmente.

**Bloque system (cuando hay System o Tools):**
```
<|im_start|>system
{SYSTEM}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {SCHEMA_1}}
{"type": "function", "function": {SCHEMA_2}}
...
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
```

**User:** `<|im_start|>user\n{CONTENT}<|im_end|>`

**Assistant con llamada a herramienta:**
```
<|im_start|>assistant
<tool_call>
{"name": "read", "arguments": {"path": "demo.py"}}
</tool_call><|im_end|>
```
(Múltiples llamadas en paralelo = varias líneas JSON dentro del mismo bloque `<tool_call>`.)

**Resultado de herramienta (role tool):** se renderiza como turno **user**:
```
<|im_start|>user
<tool_response>
{CONTENIDO}
</tool_response><|im_end|>
```

**Assistant respuesta directa (sin tool):** `<|im_start|>assistant\n{TEXTO}<|im_end|>`

## 5. Implicaciones para el dataset (Fase 2)
- Render con `tokenizer.apply_chat_template(messages, tools=TOOLS, ...)` del tokenizer base
  (Qwen3 canónico = lo que Ollama replica) — o réplica directa de la plantilla Go de arriba.
  Verificar que ambos coincidan antes de generar en masa.
- Solo 6 tools. Casos "lista/busca/cuenta archivos" → `bash` con ls/rg/find/wc.
- "búscame en internet" → `web_search`; "lee/resume esta página/URL" → `web_fetch`.
- `edit` con `edits:[{oldText,newText}]`.
- Prompts de usuario en la voz del usuario (mexicano casual: "porfa/pls/a ver/checa/mira/sig").
- Sin `<think>`: respuesta directa (tool_call o texto) para máxima velocidad.
- Mantener proporción de casos "sin herramienta" (preguntas conceptuales → texto directo).

## 6. Nota: por qué el modelo de Claude Code falla en Pi (evidencia)
En sesiones reales (`~/.pi/agent/sessions/`) con `qwen3-toolcalling` (Claude Code), Pi emitió
el `web_fetch` correctamente (el formato se parseó) pero el modelo **alucinó** la respuesta
("Imperioon es un agente de programación... v1.0.3 en PyPI") porque su system prompt y tools
entrenados son otros. Esto justifica el re-fine-tune específico para Pi.
