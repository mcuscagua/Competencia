# Competencia
Competencia de Métodos Estadísticos Avanzados.

## Modelo Continuo
En la carpeta carpeta continua hay 4 notebooks:

1. Análisis Descriptivo: donde se hace un ánalisis descriptivo de los datos
2. Selección de variables: es donde se corren los diferentes métodos de selección de variables y se escribe el archivo: variablesToProof
3. Selección del modelo: se parte del archivo variablesToProof y se corre modelos lineales 1000 veces para cada conjunto y de ahi salen los conjutnos que se van a probar en la siguiente fase
4. TestModelSelected: y aqui finalmente de hace el grid search y se corre 500 veces cada conjunto y finalmente se escribe el archivo 20191011_ResultadosCV_scores con el que finalmente se toman las decisiones

Y 7 archivos csv:

1. 20191011_ResultadosCV_scores
2. continuous_test
3. continuous_train
4. datacontinuosstudents
5. datasetx
6. variablesToProof
