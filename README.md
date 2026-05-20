# Parcial_2_programacion_ciencia_datos

# Evaluación Parcial N°2: Monitoreo de Objetos Cercanos a la Tierra (NEO)

## Integrantes:

*   Ma Alejandra Marambio Anabalón.
*   Pamela Andrea Toro Herrera.

### Fecha de envío: Martes 19 de mayo, 2026.
### Fecha de presentación: Viernes 22 de mayo, 2026.

## Contexto del Proyecto
Este proyecto aborda la problemática del monitoreo de Objetos Cercanos a la Tierra (NEO) por parte de la NASA. Ante el vasto volumen de datos astronómicos, el objetivo es desarrollar modelos de Machine Learning para clasificar asteroides como 'peligrosos' o 'no peligrosos', y explorar su estructura subyacente mediante técnicas no supervisadas. El fin último es optimizar la identificación de amenazas potenciales sin depender exclusivamente de la revisión manual.

## 1. Librerías Utilizadas
Para el análisis, preprocesamiento y modelado se utilizaron librerías clave como `numpy`, `pandas`, `matplotlib`, `seaborn`, y módulos de `scikit-learn` para preprocesamiento (`StandardScaler`, `LabelEncoder`, `ColumnTransformer`, `Pipeline`), modelado supervisado (`LogisticRegression`, `DecisionTreeClassifier`), modelado no supervisado (`PCA`, `KMeans`), y evaluación de modelos (`classification_report`, `confusion_matrix`, `roc_auc_score`, etc.).

## 2. Carga y Preparación de Datos
El dataset `neo.csv` se cargó en un DataFrame de Pandas. Se realizó un análisis exploratorio de datos (EDA) para entender la estructura, identificar patrones y preparar las características:

*   **Descripción General**: El dataset contiene información sobre aproximadamente 90,000 NEOs con características físicas y orbitales.
*   **Valores Faltantes y Duplicados**: Se confirmó la ausencia de valores nulos y duplicados, asegurando la calidad inicial de los datos.
*   **Visualización**: Gráficos como box plots, pair plots y heatmaps de correlación revelaron la distribución de las variables numéricas, la presencia de outliers y la correlación entre características y la peligrosidad de los asteroides.
*   **Desbalance de Clase**: La variable objetivo (`hazardous`) mostró un fuerte desbalance (aproximadamente 92% no peligrosos vs. 8% peligrosos), lo cual se abordó en el modelado.
*   **Preprocesamiento**: Se aplicó `LabelEncoder` a la variable objetivo y `StandardScaler` a las características numéricas para manejar outliers y escalar los datos adecuadamente para los modelos. Columnas no informativas (`id`, `name`, `orbiting_body`, `sentry_object`) fueron eliminadas.

## 3. Modelado Supervisado: Clasificación de NEOs Peligrosos
Se implementaron y optimizaron dos modelos de clasificación para predecir si un NEO es `peligroso`:

### Regresión Logística
*   **Objetivo**: Establecer un modelo de línea base interpretable y eficiente.
*   **Estrategia**: Se utilizó un pipeline con `SimpleImputer`, `StandardScaler` y `LogisticRegression` con `class_weight='balanced'` para manejar el desbalance de clases.
*   **Optimización**: Se empleó `GridSearchCV` para optimizar hiperparámetros (`C`, `penalty`, `solver`), con `f1-score` como métrica principal debido a la importancia de equilibrar la precisión y la sensibilidad en la detección de la clase minoritaria.
*   **Resultados Clave**: El modelo optimizado demostró un alto `Recall` (capacidad de identificar asteroides peligrosos), crucial para problemas de seguridad, aunque con una `Precision` moderada.

### Árbol de Decisión
*   **Objetivo**: Proporcionar un modelo con alta explicabilidad y robustez ante relaciones no lineales.
*   **Estrategia**: Similar pipeline de preprocesamiento, seguido de `DecisionTreeClassifier` con `class_weight='balanced'`.
*   **Optimización**: `GridSearchCV` se utilizó para afinar hiperparámetros como `max_depth`, `min_samples_split` y `criterion`, también optimizando el `f1-score`.
*   **Resultados Clave**: El árbol de decisión optimizado mostró una `Accuracy` general más alta y un `F1-score` ligeramente superior a la regresión logística, aunque con un `Recall` menor para la clase peligrosa.

## 4. Modelado No Supervisado: Exploración de la Estructura de Datos
Se aplicaron técnicas no supervisadas para descubrir patrones intrínsecos en los datos de los NEOs:

### Análisis de Componentes Principales (PCA)
*   **Objetivo**: Reducir la dimensionalidad de las características numéricas manteniendo la mayor varianza posible.
*   **Resultados**: Las dos primeras componentes principales lograron explicar aproximadamente el 76% de la varianza total, permitiendo visualizar la agrupación de los NEOs en un espacio 2D y observar cómo se distribuyen las etiquetas de peligrosidad.

### K-Means Clustering
*   **Objetivo**: Segmentar los asteroides en grupos basados en sus características físicas y orbitales.
*   **Determinación de K**: El método del codo sugirió que **3 clústeres** era un número óptimo de agrupaciones.
*   **Resultados**: K-Means identificó agrupaciones naturales dentro del dataset, las cuales fueron visualizadas en el espacio reducido por PCA, revelando la estructura oculta de la población de NEOs.

## 5. Evaluación y Comparación de Modelos
La comparación entre la Regresión Logística y el Árbol de Decisión reveló un trade-off. La **Regresión Logística optimizada** destacó por su alto `Recall` (0.93), siendo muy eficaz en la identificación de asteroides peligrosos, minimizando los falsos negativos. El **Árbol de Decisión optimizado**, por su parte, obtuvo un `F1-Score` ligeramente superior y una mejor `Accuracy` general, pero a expensas de un `Recall` más bajo (0.70) para la clase peligrosa.

La integración de PCA con la Regresión Logística no mostró una mejora significativa en el rendimiento, sugiriendo que las características originales ya eran suficientemente informativas o que la reducción a solo dos componentes perdía información relevante para la clasificación.

## 6. Exportación Final
El DataFrame final, incluyendo la nueva columna de clústeres generada por el modelo no supervisado, se exportó a un archivo CSV denominado `neo_nuevo.csv`.

## Conclusión
El proyecto exitosamente aplicó técnicas de Machine Learning para el análisis de NEOs. La **Regresión Logística Optimizada** se posiciona como una opción robusta para sistemas de alerta temprana debido a su alta capacidad para detectar objetos peligrosos. Las técnicas no supervisadas complementaron el análisis al desvelar la estructura inherente de los datos. Para el futuro, se podrían explorar modelos más avanzados o la ingeniería de características adicionales para refinar aún más la predicción y el entendimiento de los Objetos Cercanos a la Tierra.
