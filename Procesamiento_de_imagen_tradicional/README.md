# Procesamiento de imagen para APNR por métodos tradicionales


Dentro de las diferentes alternativas que existen para realizar el proceso de APNR (Automatic Plate Number Recognition), se exploraron en primera instancia técnicas de procesamiento de imagen por métodos tradicionales.

Debido a la complejidad del proceso, se decidió dividirlo en dos etapas principales: reconocimiento de la zona de la placa en una imagen y reconocimiento de los caractéres de la zona de la placa.

A continuación se describe el proceso y cada uno de los pasos seguidos en la estructuración de un programa para la detección de la zona de la placa usando únicamente métodos tradicionales. 

## Programa para el reconocimiento de la zona de la placa en una imagen

El diseño de un programa para el reconocimiento de la zona de la placa haciendo uso de únicamente métodos tradicionales presenta un reto debido a la variedad de variables y factores que pueden afectar la extracción de características. Factores como la iluminación, rotación, resolución o color de la imagen afectan de manera considerable el desempeño de algoritmos tradicionales. Por este motivo, se implementó un sistema en cascada en donde se implementan diferentes algoritmos permitiendo así una mayor robustez en el programa. Un diagrama del funcionamiento del sistema se muestra a continuación.

<p align="center">
<img src= https://user-images.githubusercontent.com/102924128/197409652-b9fccb8a-1c54-4e21-b353-f917aeb48643.png alt="Sistema en cascada" title="Sistema en cascada" >
</p>



### Algoritmo 1: Contornos

### Algoritmo de verificación

### Algoritmo 2: Bordes verticales

### Otras alternativas exploradas


## Conclusiones



