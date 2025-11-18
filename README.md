# ✈️ Análisis 360º del tráfico aéreo de San Francisco (SFO)

**Análisis integral del Aeropuerto Internacional de San Francisco: de la exploración de datos al pronóstico predictivo**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PySpark](https://img.shields.io/badge/PySpark-3.0+-red.svg)](https://spark.apache.org/)

---

## 📊 Sobre el proyecto

Este proyecto presenta un análisis exhaustivo del tráfico aéreo del Aeropuerto Internacional de San Francisco (SFO) durante el período 2005-2016. A través de técnicas avanzadas de ciencia de datos, machine learning y análisis de series temporales, se identifican patrones operativos clave, tendencias de crecimiento y se construyen modelos predictivos para la planificación estratégica.

**Desarrollado por:** María Luisa Ros Bolea  
**Fecha:** Noviembre 2025  
**Dataset:** Air Traffic Passenger Statistics - San Francisco International Airport
Análisis en profundidad en el PDF

--[Análisis 360º del tráfico aereo de San Francisco DEF.pdf](https://github.com/user-attachments/files/23601859/Analisis.360.del.trafico.aereo.de.San.Francisco.DEF.pdf)
-

## 🎯 Objetivos del análisis

1. **Caracterizar el perfil operativo** del aeropuerto SFO identificando sus patrones de tráfico dominantes
2. **Identificar los factores clave** que determinan el volumen de pasajeros y la distribución del tráfico
3. **Segmentar aerolíneas** según sus perfiles operativos utilizando técnicas de clustering
4. **Desarrollar modelos predictivos** para:
   - Estimar el volumen de pasajeros basado en características del vuelo
   - Clasificar vuelos como domésticos o internacionales
   - Pronosticar el tráfico futuro considerando tendencia y estacionalidad
5. **Generar insights estratégicos** para la optimización de recursos y planificación operativa

---

## 🛠️ Stack tecnológico

### Lenguajes y entornos
- **Python 3.8+**: Lenguaje principal del análisis
- **Jupyter Notebook**: Entorno de desarrollo interactivo
- **Google Colab**: Plataforma de ejecución en la nube

### Librerías de análisis y visualización
```python
pandas>=1.3.0           # Manipulación y análisis de datos
numpy>=1.21.0           # Operaciones numéricas
matplotlib>=3.4.0       # Visualizaciones estáticas
seaborn>=0.11.0         # Visualizaciones estadísticas avanzadas
plotly>=5.0.0           # Visualizaciones interactivas y mapas
```

### Machine Learning y estadística
```python
scikit-learn>=0.24.0    # Modelos de ML (Random Forest, Regresión Logística, K-Means)
statsmodels>=0.13.0     # Series temporales (SARIMA, pruebas de estacionariedad)
```

### Big Data
```python
pyspark>=3.0.0          # Procesamiento distribuido de datos a gran escala
```

---

## 📁 Estructura del proyecto

```
├── Análisis_360º_de_tráfico_aéreo_de_San_Francisco.ipynb
├── air_traffic_data.csv
└── README.md
```

---

## 🔍 Metodología del análisis

### 1. Exploración y limpieza de datos

**Acciones realizadas:**
- Carga y exploración inicial del dataset (dimensiones, tipos de datos, valores nulos)
- Estandarización de nombres de columnas
- Análisis y tratamiento de valores faltantes (decisión: eliminación de filas con nulos)
- Validación de integridad de datos

**Hallazgos iniciales:**
- Dataset con información de vuelos, pasajeros, aerolíneas, terminales y regiones geográficas
- Presencia de valores nulos en códigos IATA de aerolíneas
- Variables categóricas y numéricas combinadas que requieren preprocesamiento

### 2. Análisis exploratorio de datos (EDA)

#### 2.1 Distribución de pasajeros por terminal
- **Resultado:** La Terminal 3 concentra el mayor volumen de tráfico
- **Insight:** Cualquier optimización debe priorizar esta terminal por su papel central

#### 2.2 Análisis por región geográfica
- **Top rutas:** El tráfico doméstico (US) domina con más de 339 millones de pasajeros
- **Tráfico internacional:** Asia es la región internacional más importante
- **Conclusión:** SFO es primordialmente un aeropuerto doméstico con fuerte conexión asiática

#### 2.3 Análisis de aerolíneas
- **Aerolínea dominante:** United Airlines lidera en volumen total y frecuencia de vuelos
- **Top 10 aerolíneas** identificadas por total de pasajeros y número de operaciones
- **Eficiencia:** American Airlines y Southwest muestran la mayor eficiencia por vuelo

#### 2.4 Distribución por tipo de vuelo
- **Hallazgo clave:** SFO funciona como aeropuerto de Origen y Destino (O&D)
- **Tráfico de tránsito:** Insignificante comparado con embarques y desembarques
- **Implicación:** Los esfuerzos deben centrarse en optimizar procesos de terminal

### 3. Ingeniería de características

**Creación del perfil operativo por aerolínea:**
- Total de vuelos
- Total de pasajeros
- Media de pasajeros por vuelo
- Porcentaje de operaciones internacionales
- Número de destinos únicos

**Visualizaciones generadas:**
- Top 10 aerolíneas por frecuencia de vuelos
- Top 10 aerolíneas por volumen de pasajeros
- Aerolíneas con mayor capacidad media por vuelo
- Aerolíneas con mayor porcentaje de operaciones internacionales

### 4. Análisis de series temporales

#### 4.1 Preparación de la serie temporal
- Conversión de `activity_period` (formato YYYYMM) a índice datetime
- Agregación mensual del total de pasajeros ajustados

#### 4.2 Descomposición temporal
**Componentes identificados:**
- **Tendencia:** Crecimiento constante y sostenido durante todo el período
- **Estacionalidad:** Patrón anual muy marcado con picos en verano (junio-agosto)
- **Residuos:** Mayormente aleatorios, indicando que la descomposición captura bien los patrones

#### 4.3 Prueba de estacionariedad (Dickey-Fuller Aumentada)
- **Resultado:** Serie NO estacionaria (como era esperado)
- **Motivo:** Presencia clara de tendencia creciente y estacionalidad
- **Acción:** Aplicación de diferenciación para modelado SARIMA

### 5. Modelado predictivo

#### 5.1 Regresión: Random Forest Regressor

**Objetivo:** Predecir el volumen de pasajeros (`adjusted_passenger_count`)

**Características utilizadas:**
- `operating_airline` (One-Hot Encoded)
- `geo_region` (One-Hot Encoded)
- `terminal` (One-Hot Encoded)
- `month` (extraído de `activity_period`)
- `activity_type_code` (One-Hot Encoded)
- `price_category_code` (One-Hot Encoded)
- `year`

**Resultados:**
- **RMSE:** 38,734.59 pasajeros
- **Interpretación:** Error promedio aceptable dado el rango de volúmenes (cientos de miles/millones)

**Top 10 características más importantes:**
1. `GEO_Region_US` (0.176) - Factor más relevante
2. `Terminal` específicas
3. `Activity_Type_Code`
4. `Month` (captura estacionalidad)
5. Aerolíneas principales

**Conclusión:** El modelo valida que el carácter doméstico de SFO y la terminal son los predictores clave del volumen.

#### 5.2 Clasificación: Regresión Logística

**Objetivo:** Clasificar vuelos como domésticos (0) o internacionales (1)

**Variable objetivo:** Creada a partir de `geo_region` (US = 0, resto = 1)

**Resultados:**
- **Accuracy:** >90%
- **Precision:** ~93% (cuando predice internacional, casi siempre acierta)
- **Recall:** ~92% (encuentra la mayoría de vuelos internacionales reales)
- **F1-Score:** Balance excelente entre precision y recall

**Matriz de confusión:** Confirmó la alta capacidad discriminatoria del modelo

#### 5.3 Clustering: K-Means

**Objetivo:** Segmentar aerolíneas según su perfil operativo

**Características utilizadas:**
- Total de vuelos
- Total de pasajeros
- Media de pasajeros por vuelo
- Porcentaje de operaciones internacionales

**Proceso:**
1. **Método del codo:** Determinación del número óptimo de clusters (k=3 o k=4)
2. **Estandarización:** StandardScaler aplicado a las características
3. **Aplicación del algoritmo:** K-Means con k óptimo

**Segmentos identificados:**
- **Operadores 100% Internacionales:** Bajo volumen, alta especialización geográfica
- **Aerolíneas Regionales de Alta Frecuencia:** Muchos vuelos, bajo volumen por vuelo
- **Carriers Nacionales de Gran Volumen:** Dominio doméstico, alta eficiencia
- **Aerolíneas de Nicho:** Perfiles mixtos o especializados

#### 5.4 Pronóstico: SARIMA

**Objetivo:** Predecir el tráfico de pasajeros para los próximos 24 meses

**Modelo:** SARIMA (Seasonal AutoRegressive Integrated Moving Average)
- **Componente estacional:** Captura los picos anuales de verano
- **Componente autoregresivo:** El tráfico de un mes depende de meses anteriores
- **Integración:** Manejo de la no estacionariedad

**Resultados:**
- **Pronóstico generado:** 24 meses futuros con intervalos de confianza al 95%
- **Picos pronosticados:** Verano >5 millones de pasajeros mensuales
- **Valles pronosticados:** Febrero ~3.7-3.9 millones de pasajeros mensuales

**Aplicación práctica:** Herramienta clave para:
- Planificación de personal
- Gestión de capacidad en terminales
- Optimización de recursos estacionales

### 6. Análisis con PySpark

**Objetivo:** Procesamiento distribuido y análisis avanzado a escala

**Análisis realizados:**
1. **Agregaciones por aerolínea:** Total de pasajeros, frecuencia de vuelos, destinos
2. **Análisis descriptivo:** Estadísticas por aerolínea
3. **Matriz de correlación:**
   - `Total_Passengers` vs `Flight_Frequency`: 0.90 (muy fuerte positiva)
   - `Total_Passengers` vs `Num_Destinations`: 0.84 (fuerte positiva)
4. **Cuadrante estratégico:** Segmentación dual de aerolíneas:
   - **Negocio del VOLUMEN:** Domésticas, alta frecuencia, muchos destinos
   - **Negocio del VALOR:** Internacionales, baja frecuencia, alto valor por pasajero

**Insight clave:** SFO debe gestionarse como cuatro negocios distintos según terminal y tipo de tráfico.

---

## 📈 Principales hallazgos

### Perfil operativo de SFO

✅ **Aeropuerto predominantemente doméstico:** Más del 85% del tráfico es dentro de Estados Unidos  
✅ **Función O&D (Origen y Destino):** Tráfico de tránsito/conexión es mínimo  
✅ **Tendencia de crecimiento sostenido:** Incremento constante de 2005 a 2016  
✅ **Estacionalidad anual marcada:** Picos en verano (junio-agosto), valles en invierno

### Factores críticos de operación

🔑 **Terminal 3:** Epicentro del tráfico, concentra la mayor actividad  
🔑 **United Airlines:** Dependencia estratégica crítica (volumen y frecuencia dominantes)  
🔑 **Ruta US (doméstica):** Principal driver del volumen de pasajeros  
🔑 **Conectividad Asia-Europa:** Pilares de tráfico internacional

### Insights estratégicos

💡 **Segmentación dual necesaria:**
   - Operación doméstica de alto volumen y eficiencia
   - Operación internacional de alto valor y servicio premium

💡 **Inversión prioritaria:** Terminal 3 y procesos de embarque/desembarque

💡 **Gestión de capacidad:** Planificación estacional crítica para absorber picos de verano

💡 **Diversificación de riesgo:** Reducir dependencia de United Airlines atrayendo más carriers

---

## 📊 Visualizaciones destacadas

El notebook incluye múltiples visualizaciones profesionales:

- 📊 Gráficos de barras (distribución por terminal, aerolíneas, regiones)
- 📦 Box plots (distribución de pasajeros por tipo de vuelo)
- 🔥 Mapas de calor (correlaciones, matrices de confusión)
- 📈 Gráficos de series temporales (tendencia, estacionalidad, pronósticos)
- 🎯 Gráficos de importancia de características (Random Forest)
- 🗺️ Mapas interactivos (rutas aéreas con Plotly)
- 📊 Dashboards de insights clave
- 📉 Método del codo (selección de k en K-Means)

---

## 🚀 Cómo ejecutar el proyecto

### Requisitos previos

1. Python 3.8 o superior instalado
2. Cuenta de Google (para ejecutar en Google Colab)
3. Dataset `air_traffic_data.csv`

### Opción 1: Google Colab (recomendado)

1. Sube el archivo `Análisis_360º_de_tráfico_aéreo_de_San_Francisco.ipynb` a tu Google Drive
2. Abre el notebook con Google Colab
3. Sube el dataset `air_traffic_data.csv` a Colab o súbelo a tu Drive
4. Ejecuta las celdas secuencialmente

### Opción 2: Entorno local

```bash
# Clonar el repositorio
git clone https://github.com/malurosbolea-ux/analisis-trafico-aereo-sfo.git
cd analisis-trafico-aereo-sfo

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install pandas numpy matplotlib seaborn plotly scikit-learn statsmodels pyspark jupyter

# Abrir Jupyter Notebook
jupyter notebook
```

---

## 📝 Estructura del notebook

El análisis está organizado en las siguientes secciones principales:

1. **Configuración inicial y carga de datos**
2. **Limpieza y preprocesamiento**
3. **Análisis exploratorio de datos (EDA)**
   - Distribución por terminal
   - Análisis de regiones geográficas
   - Análisis de aerolíneas
   - Tipos de vuelo
4. **Ingeniería de características**
5. **Análisis de series temporales**
   - Descomposición temporal
   - Prueba de estacionariedad
6. **Modelado predictivo**
   - Random Forest Regressor
   - Regresión Logística
   - K-Means Clustering
   - SARIMA
7. **Análisis con PySpark**
8. **Dashboard de insights clave**
9. **Conclusiones generales**

---

## 🎓 Competencias técnicas aplicadas

Este proyecto demuestra dominio en:

- ✅ **Python avanzado** para ciencia de datos
- ✅ **Manipulación y limpieza de datos** con Pandas
- ✅ **Análisis exploratorio de datos (EDA)** profundo
- ✅ **Visualización de datos** con Matplotlib, Seaborn y Plotly
- ✅ **Ingeniería de características** (feature engineering)
- ✅ **Machine Learning supervisado** (regresión y clasificación)
- ✅ **Machine Learning no supervisado** (clustering)
- ✅ **Análisis de series temporales** (SARIMA, estacionariedad)
- ✅ **Big Data con PySpark** (procesamiento distribuido)
- ✅ **Interpretación de modelos** y generación de insights de negocio
- ✅ **Storytelling con datos** (narración analítica clara)

---

## 🌟 Conclusiones clave del análisis

El análisis 360º del tráfico aéreo de SFO proporciona una base empírica sólida para la toma de decisiones estratégicas:

1. **Perfil operativo claro:** SFO es un aeropuerto doméstico de O&D con crecimiento sostenido y alta estacionalidad

2. **Factores críticos identificados:** Terminal 3, United Airlines, ruta doméstica US y conectividad Asia-Europa

3. **Modelos predictivos validados:** 
   - Random Forest explica el volumen con RMSE aceptable
   - Regresión Logística clasifica vuelos con >90% de precisión
   - SARIMA proporciona pronósticos estacionales confiables

4. **Segmentación estratégica:** Las aerolíneas operan bajo perfiles distintivos que requieren gestión diferenciada

5. **Planificación estacional crítica:** Los picos de verano (>5M pasajeros) exigen planificación proactiva de recursos

6. **Oportunidad de diversificación:** Reducir dependencia de United Airlines es un objetivo estratégico de largo plazo

Este análisis demuestra cómo las técnicas avanzadas de ciencia de datos pueden transformar datos operativos brutos en inteligencia estratégica accionable para la gestión aeroportuaria.

---

## 📧 Contacto

**María Luisa Ros Bolea**

📧 Email: malurosbolea@gmail.com  
💼 LinkedIn: [María Luisa Ros Bolea](https://www.linkedin.com/in/maría-luisa-ros-bolea-400780160/)  
🐙 GitHub: [@malurosbolea-ux](https://github.com/malurosbolea-ux)  
🌐 Portfolio: [Portfolio Profesional](https://marialuisarosboleaportfolio.my.canva.site/porfolio-profesional-mar-a-luisa-ros-bolea-actualizado)

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

---

## 🙏 Agradecimientos

Agradezco a la comunidad de ciencia de datos y a todas las fuentes de conocimiento que han hecho posible este análisis. El dataset utilizado corresponde a información pública del Aeropuerto Internacional de San Francisco.

---

**⭐ Si este proyecto te resulta útil o interesante, considera darle una estrella en GitHub!**

*Desarrollado con 💜 por María Luisa Ros Bolea - Noviembre 2025*
