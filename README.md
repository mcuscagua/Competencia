# Competencia
Competencia de Métodos Estadísticos Avanzados.

## Modelo Binario

En la carpeta binaria hay 4 notebooks:

1.	Binary_set feauture selection: donde se realiza el análisis descriptivo y la selección de variables.
2.	Gridsearch3: se utiliza para la selección de modelos
3.	TestModelSelected: y aqui finalmente de hace el grid search y se corre 500 veces cada conjunto y finalmente se escribe el archivo 20191011_ResultadosCV_scores con el que finalmente se toman las decisiones
4.	Model_Predict vf: En este archivo se ingresa la bd de registros desconocidos para realizar la predicción.

Y 7 archivos csv:
1.	Databinarystudents
2.	datasetx
3.	salida_bin (predicción de “yL” y probabilidades)



## Modelo Continuo
En la carpeta carpeta continua hay 4 notebooks:

1. Análisis Descriptivo: donde se hace un ánalisis descriptivo de los datos
2. Selección de variables: es donde se corren los diferentes métodos de selección de variables y se escribe el archivo: variablesToProof
3. Selección del modelo: se parte del archivo variablesToProof y se corre modelos lineales 1000 veces para cada conjunto y de ahi salen los conjutnos que se van a probar en la siguiente fase y se escribe el archivo 20190924 ResultadosCV_scores.
4. TestModelSelected: y aqui finalmente de hace el grid search y se corre 500 veces cada conjunto y finalmente se escribe el archivo 20191011_ResultadosCV_scores con el que finalmente se toman las decisiones

Y 7 archivos csv:

1. 20191011_ResultadosCV_scores
2. continuous_test
3. continuous_train
4. datacontinuosstudents
5. datasetx
6. variablesToProof
7.20190924 ResultadosCV_scores

## Modelo Conteo
En la carpeta carpeta Count hay 4 notebooks:

1. Selección de Variables - Sistema de Votación por Modelos: Se implementa la función SelectFromModel de sklearn para encontrar las variables más relevantes a partir de la importancia de los pesos dados por el modelo a cada variable.
2. Selección de variables - Exploración Aleatoria: Donde se evalúan 2 millones de combinaciones diferentes para la selección de los regresores.
3. Selección del modelo: se exploran los algoritmos usando las variables seleccionadas en la búsqueda aleatoria.
4. TestModelSelected: Aquí se hace una revisión de la estabilidad del XGBoost con diferentes conjuntos de variables con resultados similares a las seleccionadas para revisar la estabilidad. 

4 archivos csv:

1. count_test
2. count_train
3. datasetx
4. datacountstudents

Y el archivo modelCount.sav contiene el modelo entrenado.
