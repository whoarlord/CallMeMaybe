* **Este proyecto ha sido creado como parte del currículo de 42 por iarrien-.**

# Llamadas a funciones con decodificación restringida

## Descripción

Este proyecto transforma solicitudes en lenguaje natural en **llamadas a funciones estructuradas** utilizando un modelo de lenguaje pequeño (Qwen, ~0.6B parámetros). Dada una solicitud como *"¿Cuál es la suma de 40 y 2?"* y una lista de funciones disponibles, el programa produce un objeto JSON como:

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {"a": 40.0, "b": 2.0}
}
```

Los modelos pequeños no son fiables a la hora de generar JSON válido por sí solos. En lugar de confiar en que el modelo se comporte correctamente y volver a intentarlo cuando no lo haga, este proyecto utiliza **decodificación restringida**: en cada paso de generación, se enmascaran los tokens que romperían la estructura esperada, de modo que el modelo solo puede producir una llamada sintácticamente válida, con un nombre de función real y argumentos correctamente tipados.

### Resumen

* Entrada: un archivo con las definiciones de las funciones y un archivo con los prompts.
* Salida: un archivo JSON con una llamada a función por cada prompt.
* El modelo solo elige *entre continuaciones válidas*; la gramática garantiza el formato.

## Instrucciones

### Requisitos

* Python 3.10 o posterior
* El paquete `llm_sdk` proporcionado (envoltorio alrededor del LLM pequeño)
* Dependencias de Python: `numpy`, `pydantic`
* Herramientas de desarrollo: `flake8`, `mypy`

### Instalación

```bash
# [ajusta esto a tu configuración: uv / pip / Makefile]

make install

# o

pip install numpy pydantic flake8 mypy
```

### Ejecución

```bash
# [ajusta el nombre del módulo y las opciones a tu punto de entrada]

python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

Reglas de rutas aplicadas por la clase `Parser`:

* Las rutas deben ser relativas (sin `/` ni `..` al principio).
* Los archivos de entrada deben estar dentro de `data/input/` y existir como archivos legibles.
* El archivo de salida debe estar dentro de `data/output/` y debe poder escribirse en él.

### Formatos de entrada

`functions_definition.json`: una lista de funciones, cada una con `name`, `description`, `parameters` (un mapeo de cada nombre de parámetro a `{"type": ...}`) y `returns`.

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
    "returns": {"type": "number"}
  }
]
```

`function_calling_tests.json`: una lista de objetos con una clave `prompt`.

```json
[{"prompt": "What is the sum of 2 and 3?"}]
```

Tipos de parámetros compatibles: `string` (`str`), `number` (`num`), `integer` (`int`), `boolean` (`bool`).

### Linting y comprobación de tipos

```bash
flake8

mypy --strict .
```

## Explicación del algoritmo

El enfoque consiste en una **gramática de estados finitos** a nivel de token aplicada a los logits del modelo durante la decodificación codiciosa.

### 1. Construcción del prompt y forzado del prefijo

`Processor.improve_prompt` construye un prompt con las funciones disponibles (en formato JSON), un conjunto de reglas y la solicitud del usuario. A continuación, la generación se **inicia** con un prefijo fijo:

```
{"prompt": "<user request>", "name":
```

Por construcción, el campo `prompt` se copia literalmente (nunca depende del modelo), y el modelo solo genera lo que viene después: el nombre de la función y los parámetros.

### 2. Máquina de estados JSON a nivel de carácter (`JsonTokenizer`)

`JsonTokenizer` valida el JSON carácter por carácter utilizando una pila (para `{`/`[` anidados) y un estado (`OBJ_OPEN`, `KEY_STRING`, `AFTER_COLON`, `STRING_VALUE`, `NUMBER`, `BOOL`, `AFTER_VALUE`, `DONE`, ...). También gestiona las secuencias de escape dentro de las cadenas, prohíbe saltos de línea/tabulaciones sin escapar dentro de las cadenas y limita los espacios en blanco a un único carácter consecutivo para evitar que el modelo se quede atascado introduciendo espacios.

No conoce nada sobre los esquemas de las funciones. En su lugar, expone cuatro **callbacks**: `on_key_closed`, `on_value_enter`, `on_value_char` y `on_value_closed`, que permiten a una gramática externa aceptar o rechazar cada evento.

### 3. Gramática consciente del esquema (`FunctionCallGrammar`)

`FunctionCallGrammar` se conecta a estos callbacks y realiza un seguimiento de una *fase*:

| Fase            | Qué se aplica                                                                                                                            |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `EXPECT_NAME`   | El valor debe ser una cadena que sea un **prefijo de un nombre de función real**, y debe coincidir exactamente con uno cuando se cierre. |
| `IN_PARAMETERS` | Las claves y valores siguen el esquema de la función seleccionada. El tipo JSON de cada valor debe coincidir con el tipo declarado.      |
| `OTHER`         | No se aplica ninguna restricción adicional más allá de un JSON válido.                                                                   |

Una vez cerrado el nombre de la función, se conoce el esquema activo y cada valor de parámetro se valida mediante una **restricción de valor** elegida según su tipo:

* `PrefixTrieConstraint`: el búfer debe ser siempre un prefijo de algún candidato (nombres de funciones, `true`/`false`).
* `NumberConstraint`: permite opcionalmente un `-` inicial, dígitos y como máximo un `.`, con límites para los dígitos enteros y decimales y sin ceros iniciales.
* `IntConstraint`: lo mismo, pero sin decimales.
* `FreeStringConstraint`: ninguna regla adicional (el tokenizer ya valida las comillas y los escapes).

Cuando se han procesado todos los parámetros del esquema, se establece `no_more_parameters`, lo que prohíbe una coma final y obliga a cerrar el objeto.

### 4. Enmascaramiento de logits

En cada paso de generación:

1. El modelo devuelve los logits del siguiente token.

2. `calculate_valid_logits` recorre el vocabulario y conserva los tokens para los que `FunctionCallGrammar.check_step(token)` tiene éxito. La comprobación se realiza sobre una **copia** de la gramática, por lo que el estado real nunca se modifica debido a un token rechazado.

3. `process_valid_logits` establece los logits de todos los demás tokens en `-inf`.

4. El siguiente token se selecciona mediante `argmax` (decodificación codiciosa), se añade a la secuencia y se aplica a la gramática real.

5. El bucle termina cuando el JSON alcanza el estado `DONE` (o después de un número máximo de iteraciones o de un tiempo de espera).

Como un token solo se acepta si **todos** sus caracteres son aceptados, el modelo puede utilizar tokens de varios caracteres libremente mientras la estructura se mantiene válida.

## Decisiones de diseño

* **La gramática y el tokenizer están separados.** `JsonTokenizer` solo conoce JSON; `FunctionCallGrammar` conoce los esquemas. La interfaz de callbacks mantiene ambos componentes sencillos y permite probarlos de forma aislada.

* **Clonación para anticipación (lookahead).** `check_step` prueba cada token candidato sobre una copia del estado. Esto es más sencillo y seguro que implementar un mecanismo de deshacer, a cambio de cierta sobrecarga (mitigada por la caché descrita a continuación).

* **Caché de máscaras.** El conjunto de tokens válidos depende únicamente del estado de la gramática (estado del tokenizer, pila, fase, parámetro actual, búfer de restricciones y función activa). Se almacena en caché utilizando esa clave, por lo que los estados repetidos no tienen coste después de la primera vez.

* **Forzado del prefijo.** Inyectar `{"prompt": ..., "name":` elimina toda una clase de errores (prompts modificados o parafraseados) y acorta la generación.

* **Decodificación codiciosa.** Produce una salida determinista, lo que hace que los resultados sean reproducibles y las pruebas fiables. Como la gramática elimina las opciones no válidas, el muestreo añade variabilidad sin aportar beneficios.

* **Las restricciones de tipos están contenidas en clases pequeñas.** Añadir un tipo significa añadir una subclase de `ValueConstraint` y una línea en el mapeo de tipos.

* **Pydantic para esquemas y rutas.** `FunctionDef`/`ParamType` normalizan alias (`int`, `bool`, ...), y `Parser` valida las rutas y los permisos antes de ejecutar nada.

* **Tipado estricto y linting.** El código pasa `flake8` y `mypy --strict`, con docstrings en todas las clases y métodos públicos.

## Análisis de rendimiento

> **Completa esta sección con tus propias mediciones antes de entregar el proyecto.**

* **Precisión:** [X/Y prompts eligen la función correcta; X/Y tienen todos los argumentos correctos, medido sobre el conjunto de pruebas proporcionado.]

* **Fiabilidad:** La salida es un JSON válido con un nombre de función conocido y parámetros correctamente tipados **por construcción**, ya que los tokens no válidos son inalcanzables. Los errores restantes son *semánticos* (se elige una función incorrecta o se extrae un valor incorrecto), no sintácticos.

* **Velocidad:** [Promedio de segundos por prompt y tiempo con caché fría frente a caché caliente.] El coste principal es el primer recorrido del vocabulario en un estado nuevo de la gramática, que clona la gramática una vez por token. La caché elimina este coste para los estados repetidos. La inferencia del modelo es el otro coste principal.

* **Límites:** Cada prompt tiene un límite de tiempo (10 s por defecto) y un máximo de 500 tokens generados.

## Desafíos encontrados

* **Tokens de varios caracteres.** Los tokens pueden abarcar varios caracteres (por ejemplo, unas comillas seguidas de una coma). La gramática los valida carácter por carácter sobre una copia y solo los acepta si todos los caracteres pasan la validación.

* **Marcador de espacio BPE.** El vocabulario codifica los espacios como `Ġ`. El tokenizer lo convierte en un espacio real antes de realizar la validación.

* **Secuencias de escape.** El hecho de que un carácter esté escapado depende del carácter *anterior*, incluso entre límites de tokens, por lo que el indicador de escape se conserva entre caracteres y tokens.

* **Números.** Un número no tiene un delimitador de cierre, por lo que se cierra con el siguiente carácter (`,` o `}`). Por ello, las restricciones tienen un paso `close()`, y se aplican límites a los ceros iniciales y al número de dígitos para evitar números interminables.

* **Coste de la clonación.** Recorrer ~150k tokens realizando una clonación completa de la gramática por cada token era lento. La caché basada en el estado de la gramática solucionó la mayor parte del problema.

* **Corrección de la clonación.** La copia del estado interno (pila, búfer de claves y búfer de restricciones) tenía que ser completa; de lo contrario, una copia podría desviarse silenciosamente del original.

* **Escapado en parámetros similares a expresiones regulares.** Las barras invertidas en los valores de tipo cadena requerían una regla específica en el prompt y un manejo cuidadoso de los escapes en el tokenizer.

## Estrategia de pruebas

> **Adapta esta sección a las pruebas que realmente hayas ejecutado.**

* **Pruebas unitarias para la máquina de estados:** cadenas JSON válidas e inválidas introducidas carácter por carácter (escapes, anidamiento, espacios en blanco, cierre prematuro).

* **Pruebas unitarias para las restricciones:** números (`-1`, `0.5`, `01`, `1.`, `1..2`), enteros, booleanos y coincidencia de prefijos.

* **Pruebas de la gramática:** para un esquema determinado, comprobar que los nombres de funciones desconocidos, los parámetros desconocidos, los tipos incorrectos y los parámetros adicionales son rechazados.

* **Ejecuciones de extremo a extremo:** las definiciones de funciones y prompts proporcionados, comprobando que cada salida pueda analizarse mediante `json.loads`, que los nombres existan en las definiciones y que los tipos de los parámetros coincidan con el esquema.

* **Casos límite:** prompts vacíos, prompts muy largos, comillas y barras invertidas en la solicitud, números negativos y decimales, archivos inexistentes o ilegibles y rutas fuera de `data/`.

* **Comprobaciones estáticas:** `flake8` y `mypy --strict` sobre todo el proyecto.

## Ejemplo de uso

Ejecución básica:

```bash
python -m src
```

Archivos personalizados:

```bash
python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

Ejemplo de prompt y resultado:

```
Prompt: "Greet John"

Result: {"prompt": "Greet John", "name": "fn_greet", "parameters": {"name": "John"}}
```

Ruta no válida (rechazada antes de ejecutar):

```bash
python -m src --input /etc/passwd

# ValueError: input parameters must not start with /
```

## Recursos

### Referencias

* Willard & Louf, *Efficient Guided Generation for Large Language Models* (2023): https://arxiv.org/abs/2307.09702
* Documentación de la biblioteca Outlines (generación estructurada): https://dottxt-ai.github.io/outlines/
* Documentación y tarjetas de modelos de Qwen: https://huggingface.co/Qwen
* Hugging Face, *Tokenizer summary* (tokens BPE y de nivel de bytes): https://huggingface.co/docs/transformers/tokenizer_summary
* RFC 8259, *The JSON Data Interchange Format*: https://datatracker.ietf.org/doc/html/rfc8259
* Documentación de Pydantic: https://docs.pydantic.dev
* Documentación de NumPy: https://numpy.org/doc/
* Documentación de mypy: https://mypy.readthedocs.io
* flake8 y PEP 8: https://flake8.pycqa.org, https://peps.python.org/pep-0008/

### Uso de IA

Se utilizó IA (Claude) como herramienta de apoyo para:

* **Calidad del código:** reescritura de módulos existentes para que superaran `flake8` y `mypy --strict` (anotaciones de tipos, tipado de `Protocol`/ABC, longitud de líneas e imports no utilizados).

* **Documentación:** escritura de docstrings en inglés para todas las clases y métodos, y elaboración de este README.

* **Revisión del código:** identificación de posibles errores y casos límite (por ejemplo, en `NumberConstraint.close()`, el `clone()` del tokenizer y la gestión del tiempo de espera), que posteriormente fueron revisados y decididos manualmente.

* **Preguntas generales:** tiempos de espera y manejo de excepciones en Python.

El diseño del enfoque de decodificación restringida y las decisiones finales sobre el código fueron realizados por los autores, quienes revisaron y probaron todos los cambios realizados con asistencia de IA. [Adapta este párrafo para reflejar tu uso real.]
