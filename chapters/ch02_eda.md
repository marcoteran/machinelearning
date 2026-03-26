[![banner](/_assets/pics/bannerAI.jpg)](https://github.com/marcoteran/machinelearning)

# ST1631 — Aprendizaje de Máquina aplicado
## Tecnologías de la información y la comunicación

````markdown
# Capítulo 2: EDA, calidad de datos, preprocesamiento y pipelines

> **Objetivo del capítulo**  
> Convertir datos tabulares crudos en evidencia lista para modelar, sin contaminar la evaluación, y dejando un flujo reproducible que conecte de manera natural con el deck de clase, el notebook de laboratorio y la implementación posterior en `scikit-learn`.

---

## Estructura autocontenida del capítulo

Este capítulo está diseñado para funcionar como un **capítulo de libro completo y autónomo**.  
Incluye:

- desarrollo conceptual completo;
- integración explícita con las slides del capítulo;
- integración explícita con el notebook del capítulo;
- worked examples;
- checklist operativo;
- glosario;
- preguntas de repaso;
- inventario de figuras;
- convenciones para guardar las imágenes en una carpeta llamada `figures/`.

### Estructura sugerida de archivos

```text
chapter_02/
├── chapter_02_eda_preprocessing_pipelines.md
├── figures/
│   ├── ch2_batch1_dataset_to_experiment.png
│   ├── ch2_batch1_eda_audit_cycle.png
│   ├── ch2_batch2_missingness_map.png
│   ├── missingmcarmarmnar.png
│   ├── ch2_batch2_nominal_vs_ordinal_encoding.png
│   ├── ch2_batch2_scaling_comparison.png
│   ├── dataleak.png
│   ├── ch2_batch3_wrong_vs_right_leakage_flow.png
│   ├── pipelines.png
│   ├── ch2_batch4_columntransformer_architecture.png
│   └── ch2_chapter_infographic.png
└── notebooks/
    └── ch2_eda_preprocessing_pipelines.ipynb
````

---

# 1. Por qué este capítulo importa

En aprendizaje de máquina, una gran parte de los errores más costosos no nace en el algoritmo. Nace antes: en cómo entendemos el dataset, en cómo tratamos sus ausencias, en cómo codificamos variables, en cómo escalamos sus magnitudes y, sobre todo, en cómo evitamos contaminar la evaluación.

Este capítulo existe para corregir una intuición peligrosa: la idea de que el preprocesamiento es una fase menor, casi cosmética, previa al modelado “real”. No lo es. El preprocesamiento forma parte del modelo. Cambiar imputación, encoding o scaling cambia la representación del problema y, por tanto, cambia lo que el modelo puede aprender.

La tesis central del capítulo es esta:

> **Un experimento de machine learning no empieza cuando elegimos un algoritmo.
> Empieza cuando auditamos el dato y definimos qué información puede aprenderse y qué información debe permanecer fuera del entrenamiento.**

---

# 2. Mapa general del capítulo

Este capítulo sigue una lógica operacional:

1. **EDA como auditoría**
2. **Calidad de datos**
3. **Valores faltantes y mecanismos de ausencia**
4. **Encoding con criterio semántico**
5. **Scaling y geometría del algoritmo**
6. **Feature engineering básico**
7. **Leakage como falla experimental**
8. **Pipelines y `ColumnTransformer` como contrato de reproducibilidad**

La idea no es solo entender cada concepto por separado, sino ver cómo se conectan dentro de un mismo flujo.

---

# 3. Integración con las slides del capítulo

Este capítulo no duplica las slides, pero sí se alinea con su narrativa.
El deck del capítulo fue organizado en cuatro bloques:

* **Batch 1**: framing + backbone de EDA
* **Batch 2**: calidad de datos, faltantes y transformaciones
* **Batch 3**: leakage como modo de falla
* **Batch 4**: pipelines reproducibles y cierre operacional

## 3.1. Figuras del capítulo

Las figuras deben guardarse en la carpeta `figures/` y se referencian aquí como parte del material del capítulo.

### Figura 1. Del dataset crudo al experimento confiable

![Del dataset crudo al experimento confiable](figures/ch2_batch1_dataset_to_experiment.png)

**Rol pedagógico**
Mostrar que el dataset no pasa directamente al modelo. Antes hay auditoría, luego split, luego transformaciones aprendidas de forma segura, y solo después modelado y evaluación.

### Figura 2. Pipeline mental de auditoría

![Pipeline mental de auditoría](figures/ch2_batch1_eda_audit_cycle.png)

**Rol pedagógico**
Condensar la secuencia conceptual del EDA: problema, estructura, calidad, distribuciones, riesgos experimentales y decisiones.

### Figura 3. Mapa de missingness

![Mapa de missingness](figures/ch2_batch2_missingness_map.png)

**Rol pedagógico**
Hacer visible que el problema de los faltantes no depende solo del porcentaje, sino también del patrón.

### Figura 4. MCAR, MAR y MNAR

![MCAR, MAR y MNAR](figures/missingmcarmarmnar.png)

**Rol pedagógico**
Distinguir visualmente los mecanismos de ausencia y su impacto sobre la decisión de imputación.

### Figura 5. Nominal vs ordinal encoding

![Nominal vs ordinal encoding](figures/ch2_batch2_nominal_vs_ordinal_encoding.png)

**Rol pedagógico**
Mostrar que el encoding depende del significado de la variable, no de la comodidad de la API.

### Figura 6. Comparación de scaling

![Comparación de scaling](figures/ch2_batch2_scaling_comparison.png)

**Rol pedagógico**
Mostrar que el escalado altera la geometría del espacio de features y afecta especialmente a ciertos algoritmos.

### Figura 7. Data leakage

![Data leakage](figures/dataleak.png)

**Rol pedagógico**
Abrir visualmente la subsección de leakage como el error más peligroso del capítulo.

### Figura 8. Flujo incorrecto vs flujo correcto

![Flujo incorrecto vs flujo correcto](figures/ch2_batch3_wrong_vs_right_leakage_flow.png)

**Rol pedagógico**
Comparar `preprocess -> split` frente a `split -> fit on train -> transform on test`.

### Figura 9. Pipeline

![Pipeline](figures/pipelines.png)

**Rol pedagógico**
Mostrar que pipeline no es sintaxis decorativa, sino contrato de ejecución.

### Figura 10. Arquitectura de ColumnTransformer

![ColumnTransformer](figures/ch2_batch4_columntransformer_architecture.png)

**Rol pedagógico**
Mostrar dos rutas de preprocesamiento diferenciadas: numéricas y categóricas, convergiendo en una representación única.

### Figura 11. Infographic final del capítulo

![Infographic final del capítulo](figures/ch2_chapter_infographic.png)

**Rol pedagógico**
Cerrar el capítulo como recap visual completo:
`dataset audit -> missingness -> typing/encoding -> scaling -> feature engineering -> leakage barrier -> reproducible pipeline -> honest evaluation`.

---

# 4. EDA como auditoría del experimento

## 4.1. EDA no es una colección de gráficos

Una lectura pobre del EDA lo reduce a histogramas, boxplots y tablas descriptivas.
Una lectura madura lo entiende como **auditoría técnica del dataset**.

EDA debe responder preguntas como:

* ¿Qué estamos intentando predecir?
* ¿Cuál es la unidad de análisis?
* ¿Qué columnas existen y de qué tipo son?
* ¿Qué variables parecen problemáticas?
* ¿Dónde hay ausencia, ruido, inconsistencias o rarezas?
* ¿Qué decisiones de transformación pueden cambiar el experimento?
* ¿Qué riesgos de leakage ya se vislumbran?

El EDA no es un trámite previo al modelado. Es el punto donde definimos si el problema está bien representado.

## 4.2. Explorar primero, modelar después

Una de las mejores defensas contra errores metodológicos es postergar la prisa por modelar.
Antes de ajustar cualquier algoritmo, conviene entender:

* target;
* features;
* schema;
* missingness;
* outliers;
* duplicados;
* relaciones iniciales con el target;
* riesgos de contaminación experimental.

Ese trabajo no genera solo conocimiento descriptivo. Genera decisiones.

## 4.3. Pipeline mental de auditoría

Una secuencia útil para enseñar y practicar EDA es esta:

1. Confirmar el problema.
2. Confirmar el target.
3. Separar features y target.
4. Revisar tipos de variable.
5. Medir faltantes.
6. Revisar rangos y distribuciones.
7. Detectar outliers y duplicados.
8. Explorar asociaciones iniciales.
9. Convertir hallazgos en reglas de preprocesamiento.

---

# 5. Calidad de datos

## 5.1. Qué significa realmente

Cuando se habla de calidad de datos, muchas veces se piensa solo en celdas vacías.
Pero un dataset puede ser de mala calidad por muchas otras razones:

* tipado incorrecto;
* categorías duplicadas por escritura inconsistente;
* columnas con unidades mezcladas;
* valores extremos imposibles;
* duplicados exactos;
* variables casi constantes;
* información posterior al evento objetivo.

## 5.2. Qué preguntas hacer

Antes de transformar, conviene preguntar:

* ¿Hay columnas con demasiados faltantes?
* ¿Hay columnas con outliers severos?
* ¿Las variables categóricas tienen cardinalidad razonable?
* ¿Existen columnas sospechosas de contener leakage?
* ¿Hay variables que parecen derivadas del target?
* ¿Qué columnas requerirán rutas distintas dentro del pipeline?

## 5.3. Diagnóstico antes de intervención

La regla del capítulo es simple:

> **Diagnóstico primero. Intervención después.**

No conviene imponer la misma receta a todas las variables.
Una ausencia no siempre se imputa igual. Un outlier no siempre se elimina. Una categoría no siempre se codifica igual.

---

# 6. Valores faltantes: cantidad, patrón y mecanismo

## 6.1. El porcentaje no basta

No alcanza con decir que una columna tiene 5%, 10% o 20% de faltantes.
También importa:

* dónde faltan;
* si faltan por grupos;
* si aparecen por bloques;
* si parecen ligados a otras variables;
* si sugieren un mecanismo sistemático.

## 6.2. MCAR, MAR y MNAR

### MCAR

La ausencia es completamente aleatoria.

### MAR

La ausencia depende de variables observadas.

### MNAR

La ausencia depende del valor faltante o de causas no observadas.

## 6.3. Implicaciones prácticas

Esta clasificación no siempre puede demostrarse con certeza, pero sirve para pensar mejor.

* **MCAR**: eliminación o imputación simple pueden ser menos problemáticas.
* **MAR**: conviene considerar el contexto y las variables relacionadas.
* **MNAR**: el riesgo de sesgo es mayor; la ausencia misma puede ser informativa.

## 6.4. Estrategias de imputación

### Eliminación

Razonable si la pérdida de información es pequeña y defendible.

### Imputación simple

* media;
* mediana;
* moda;
* valor fijo.

### Indicador de ausencia

Útil cuando la ausencia misma puede cargar información.

## 6.5. Regla metodológica

Los parámetros de imputación deben aprenderse **solo con train**.

Eso significa que si usas mediana, media o moda, esas estadísticas deben calcularse sobre el conjunto de entrenamiento y luego aplicarse a validación/test.

---

# 7. Distribuciones, outliers y estadística robusta

## 7.1. Por qué comparar media y mediana

Si una variable tiene outliers o asimetría, la media puede contar una historia muy distinta de la mediana. Por eso conviene contrastarlas.

## 7.2. Regla IQR

Una regla común para detectar observaciones extremas es:

* límite inferior: `Q1 - 1.5 * IQR`
* límite superior: `Q3 + 1.5 * IQR`

donde `IQR = Q3 - Q1`.

## 7.3. Lo importante

La regla IQR no “prueba” que algo sea error.
Solo dice: **mire aquí con más cuidado**.

Un outlier puede ser:

* error;
* rareza real;
* señal valiosa.

---

# 8. Encoding: semántica antes que sintaxis

## 8.1. Pregunta correcta

No preguntes “qué encoder uso”.
Pregunta primero:

> **¿Esta categoría tiene orden real o no?**

## 8.2. Variables nominales

No tienen orden semántico.

Ejemplos:

* ciudad;
* color;
* canal.

Lo natural suele ser **one-hot encoding**.

## 8.3. Variables ordinales

Sí tienen orden real.

Ejemplos:

* bajo / medio / alto;
* S / M / L / XL;
* leve / moderado / severo.

Aquí puede tener sentido una codificación ordinal.

## 8.4. Error frecuente

Tratar una variable nominal como `0, 1, 2, 3` por comodidad.
Eso introduce distancias ficticias entre categorías.

---

# 9. Scaling: cuándo importa y cuándo no

## 9.1. Estandarización

Una forma clásica de scaling es:

`z = (x - u) / s`

donde `u` y `s` se aprenden con train.

## 9.2. Algoritmos sensibles a escala

El escalado suele importar más en modelos que dependen de:

* distancias;
* márgenes;
* regularización.

Ejemplos típicos:

* KNN;
* SVM;
* regresión logística regularizada;
* regresión lineal regularizada.

## 9.3. Algoritmos menos sensibles

En modelos basados en particiones, como árboles, el impacto del scaling suele ser menor.

## 9.4. Pregunta correcta

> **¿La geometría del algoritmo cambia con la escala de las variables?**

---

# 10. Feature engineering básico

## 10.1. Qué sí significa

Feature engineering es traducir conocimiento del problema a variables computables útiles.

## 10.2. Ejemplos razonables

* ratios;
* interacciones;
* transformaciones log;
* variables temporales;
* agregaciones prudentes.

Ejemplos sobre housing:

* `bedrooms_per_room`
* `rooms_x_income`
* `occupancy_per_room`

## 10.3. Qué no conviene

* crear columnas a ciegas;
* usar información futura;
* esconder leakage en una feature sofisticada;
* aumentar complejidad sin hipótesis.

---

# 11. Data leakage

## 11.1. Definición

Leakage ocurre cuando el flujo de entrenamiento usa información que no estaría disponible al predecir en el mundo real.

## 11.2. Tipos comunes

* **target leakage**
* **train-test contamination**
* **temporal leakage**
* **leakage por duplicados o grupos mezclados**

## 11.3. Regla de oro

> **Split first, fit later.**

Primero separas train y test.
Después aprendes imputación, encoding, scaling y cualquier otra transformación solo con train.

## 11.4. Señales de alarma

* métricas sospechosamente altas;
* columnas que resumen una decisión final del proceso;
* variables generadas después del outcome;
* features que huelen demasiado al target.

---

# 12. Pipelines y ColumnTransformer

## 12.1. Pipeline como contrato de ejecución

Un pipeline encadena transformaciones y un estimador final.

Su valor no es solo estético.
Su valor es experimental:

* protege consistencia;
* reduce errores manuales;
* facilita repetición;
* reduce riesgo de leakage.

## 12.2. ColumnTransformer

Permite rutas diferenciadas para distintos tipos de columnas.

Patrón típico:

* numéricas -> imputación + scaling
* categóricas nominales -> imputación + one-hot
* categóricas ordinales -> imputación + ordinal encoding

Todo converge a una sola representación antes del estimador.

## 12.3. Qué protege

El pipeline protege consistencia entre:

* train;
* validation;
* test;
* inferencia.

Por eso es parte del modelo completo.

---

# 13. Integración con el notebook del capítulo

Este capítulo también se integra con el notebook `notebooks/ch2_eda_preprocessing_pipelines.ipynb`.

## 13.1. Qué cubre el notebook

El notebook del capítulo fue dividido en tres chunks:

### Chunk 1

* framing de la sesión;
* setup reproducible;
* carga del dataset;
* auditoría del schema;
* missingness;
* EDA univariado y bivariado.

### Chunk 2

* concept-to-code mapping;
* split first;
* feature engineering determinístico;
* tipado semántico;
* constructor de preprocessors;
* experimentos de imputación;
* scaling sensitivity;
* leakage trap.

### Chunk 3

* interpretación guiada;
* práctica activa;
* leakage detective;
* encoding choice;
* construcción de preprocessor;
* tarea final del capítulo.

## 13.2. Qué aporta el notebook que no aporta este capítulo

El notebook aporta:

* ejecución real;
* inspección de outputs;
* experimentos comparativos;
* práctica del estudiante;
* tarea final.

## 13.3. Qué aporta este capítulo que no aporta el notebook

Este capítulo aporta:

* narrativa conceptual continua;
* worked examples explicados;
* integración con las slides;
* glosario;
* pitfalls;
* preguntas de repaso;
* síntesis durable para estudio.

---

# 14. Worked example 1: imputación simple

Supón una variable numérica con algunos faltantes y distribución sesgada.

Dos decisiones posibles:

* imputar con media;
* imputar con mediana.

## Lectura correcta

La media responde más a valores extremos.
La mediana suele ser más robusta.

## Lo importante

La decisión no depende de una regla universal, sino de:

* perfil de la variable;
* sensibilidad del modelo;
* estabilidad del experimento.

Y, otra vez: la estadística de imputación se aprende con train.

---

# 15. Worked example 2: pipeline mixto

Supón un dataset con:

* variables numéricas;
* una variable ordinal;
* una variable nominal;
* faltantes en ambas familias.

Una solución razonable:

### Ruta numérica

* `SimpleImputer(strategy="median")`
* `StandardScaler()`

### Ruta nominal

* `SimpleImputer(strategy="most_frequent")`
* `OneHotEncoder(handle_unknown="ignore")`

### Ruta ordinal

* `SimpleImputer(strategy="most_frequent")`
* `OrdinalEncoder(...)`

### Integración

* `ColumnTransformer(...)`
* `Pipeline([...])`

## Aprendizaje central

No estás solo organizando código.
Estás definiendo una arquitectura experimental reproducible.

---

# 16. Slides del capítulo: resumen textual integrado

## 16.1. Resumen de la sesión

* Replanteamos el EDA como auditoría del experimento.
* Vimos que calidad de datos incluye faltantes, outliers, duplicados, inconsistencias y tipado.
* Distinguimos MCAR, MAR y MNAR.
* Justificamos encoding según semántica.
* Entendimos cuándo scaling importa.
* Tratamos feature engineering como representación del problema.
* Enfatizamos leakage como falla experimental grave.
* Cerramos con pipelines y `ColumnTransformer` como contrato de reproducibilidad.

## 16.2. Glosario del capítulo

### EDA

Análisis exploratorio de datos como auditoría previa al modelado.

### Missingness

Patrón de ausencia de datos.

### MCAR

Missing Completely At Random.

### MAR

Missing At Random.

### MNAR

Missing Not At Random.

### Imputación

Reemplazo controlado de valores faltantes.

### Outlier

Observación extrema que requiere inspección.

### One-hot encoding

Representación binaria para variables nominales.

### Ordinal encoding

Codificación entera que preserva orden.

### Scaling

Transformación de magnitud para controlar geometría.

### Feature engineering

Creación o transformación de variables útiles.

### Leakage

Fuga de información que invalida la evaluación.

### Pipeline

Flujo reproducible de transformaciones y modelo.

### ColumnTransformer

Objeto para aplicar rutas distintas a distintos tipos de columnas.

## 16.3. Preguntas del capítulo

1. ¿Por qué el EDA es una auditoría y no solo una colección de gráficos?
2. ¿Qué cambia cuando el patrón de faltantes es estructurado?
3. ¿Cómo afectaría tu decisión si sospechas MAR o MNAR?
4. ¿Por qué una variable nominal no debe codificarse arbitrariamente como ordinal?
5. ¿En qué familias de modelos suele importar más el scaling?
6. ¿Qué riesgo hay en hacer feature engineering sin control experimental?
7. ¿Qué señales harían sospechar leakage?
8. ¿Por qué dividir primero protege la evaluación?
9. ¿Qué aporta `ColumnTransformer` frente a rutas manuales?
10. ¿Por qué pipeline debe entenderse como parte del modelo?

---

# 17. Anti-patrones y pitfalls

## 17.1. Hacer EDA para “mirar bonito”

Error de principiante.
EDA debe producir decisiones.

## 17.2. Tratar todos los faltantes igual

Error metodológico.
El mecanismo importa.

## 17.3. Codificar por reflejo

Error semántico.
Primero significado, luego encoding.

## 17.4. Escalar sin preguntarse si importa

Error de automatismo.
La sensibilidad depende del modelo.

## 17.5. Transformar antes del split

Error grave.
Produce contaminación experimental.

## 17.6. Celebrar métricas increíbles sin auditar leakage

Error de criterio.
La buena práctica exige sospecha metódica.

## 17.7. Pensar que pipeline es un lujo de código ordenado

Error conceptual.
Pipeline es reproducibilidad.

---

# 18. Checklist operativo

Antes de pasar al modelado, deberías poder responder sí a estas preguntas:

* ¿Identifiqué target y unidad de análisis?
* ¿Revisé schema, tipos y duplicados?
* ¿Audité faltantes y sus patrones?
* ¿Revisé distribuciones y outliers?
* ¿Separé nominal, ordinal y numérico?
* ¿Justifiqué imputación?
* ¿Sé si el scaling probablemente importa?
* ¿Pensé en leakage?
* ¿Dividí antes de aprender transformaciones?
* ¿Puedo encapsular el flujo en un pipeline?

---

# 19. Cierre del capítulo

Este capítulo cambia la jerarquía de prioridades en machine learning aplicado.

Después de estudiarlo, ya no debería parecer razonable hablar de modelos sin hablar antes de:

* estructura del dato;
* calidad de datos;
* mecanismos de ausencia;
* semántica de las variables;
* riesgos de contaminación experimental;
* reproducibilidad del flujo.

Un buen modelo sobre un mal procedimiento sigue siendo un mal experimento.

El siguiente paso natural no es probar más algoritmos a ciegas.
Es construir validación más rigurosa sobre una base metodológicamente sana.

---

# 20. Apéndice: relación directa entre libro, slides y notebook

## Este capítulo (libro)

Aporta:

* explicación continua;
* worked examples;
* glosario;
* preguntas;
* recap pedagógico.

## Slides

Aportan:

* narrativa oral;
* jerarquía visual;
* compresión conceptual;
* contraste rápido entre buenas y malas prácticas.

## Notebook

Aporta:

* ejecución;
* outputs;
* experimentos;
* práctica del estudiante;
* tarea final.

## Los tres juntos

Forman una misma unidad didáctica:

* **slides** para enseñar,
* **notebook** para practicar,
* **capítulo** para entender y repasar.

```

La diferencia respecto a la versión anterior es que esta ya no es solo un capítulo Markdown “de referencia”, sino un **capítulo completo y autocontenido**, que integra explícitamente:

- el desarrollo conceptual;
- el deck del capítulo;
- el notebook del capítulo;
- el inventario completo de figuras en `figures/`;
- la articulación pedagógica entre los tres artefactos.

Lo único que no puedo hacer dentro de este mensaje es materializar por texto los archivos binarios de imagen dentro de `figures/`, pero el capítulo ya quedó preparado para usarlos con nombres concretos y estructura coherente.
```
