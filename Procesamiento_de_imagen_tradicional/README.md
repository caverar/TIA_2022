# Procesamiento de imagen para APNR por métodos tradicionales


Dentro de las diferentes alternativas que existen para realizar el proceso de APNR (Automatic Plate Number Recognition), se exploraron en primera instancia técnicas de procesamiento de imagen por métodos tradicionales.

Debido a la complejidad del proceso, se decidió dividirlo en dos etapas principales: reconocimiento de la zona de la placa en una imagen y reconocimiento de los caractéres de la zona de la placa.

A continuación se describe el proceso y cada uno de los pasos seguidos en la estructuración de un programa para la detección de la zona de la placa usando únicamente métodos tradicionales. 

## Programa para el reconocimiento de la zona de la placa en una imagen

El diseño de un programa para el reconocimiento de la zona de la placa haciendo uso de únicamente métodos tradicionales presenta un reto debido a la variedad de variables y factores que pueden afectar la extracción de características. Factores como la iluminación, rotación, resolución o color de la imagen afectan de manera considerable el desempeño de algoritmos tradicionales. Por este motivo, se implementó un sistema en cascada en donde se implementan diferentes algoritmos permitiendo así una mayor robustez en el programa. Un diagrama del funcionamiento del sistema se muestra a continuación.

<p align="center">
<img src= https://user-images.githubusercontent.com/102924128/197409652-b9fccb8a-1c54-4e21-b353-f917aeb48643.png alt="Sistema en cascada" title="Sistema en cascada" >
</p>

Tal como se ve en la figura, el programa recibe una imagen y  la procesa con el algoritmo 1 el cual intenta encontrar la zona de la placa. Esa zona se pasa como argumento al algoritmo de verificación de la placa en donde se determina si fue detectada una placa o no. En caso de que una placa haya sido detectada, esa zona se retorna. En caso contrario, cuando una placa no es detectada, se hace uso del algoritmo 2 para detectar la zona de la placa en la imagen. De nuevo en este caso se verifica si la zona escogida corresponde efectivamente a una placa. En caso afirmativo se retorna esa zona. En caso contrario, se retorna la imagen original.

En las siguientes secciones se explicará el funcionamiento del algoritmo 1 y 2 así como el algoritmo de verificación.

### Algoritmo 1: Contornos

### Algoritmo de verificación

### Algoritmo 2: Bordes verticales

### Programa completo

### Otras alternativas exploradas

Además de los algoritmos previamente mencionados, se
exploraron otras alternativas. A continuación se describen
brevemente dos de ellas.

* **Filtrado por color:** Con este enfoque se buscó limitar el
dataset de placas a aquellas que fuesen de color amarillo.
Las imágenes se guardaron en formato HSV y se filtró
por el valor de Hue (conocido también como matiz o
tono) seleccionando el rango de grados que segmentara
esa tonalidad.


```python
# Cargamos la imagen
img_original = cv2.imread('Proc_imagen/Fig/Placa5.jpg')
img = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

# Convertimos la imagen a HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

# Definimos los colores que queremos detectar   
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

print(np.max(h), np.min(h))

for i in range(h.shape[0]):
    for j in range(h.shape[1]):
        if h[i,j] < 90 or h[i, j] > 110:
            img[i,j] = 0
```


## Conclusiones

La exploración de resultados mediante técnicas tradicionales permite un entendimiento a profundidad de cada
uno de los algoritmos que se implementan. Pese a su
rigidez y la alta dependencia de las condiciones de captura de la imagen, en comparación a sistemas que usan redes neuronales, se puede obtener un sistema con mayor robustez al usar un enfoque de métodos en cascada.

