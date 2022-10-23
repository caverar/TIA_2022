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

El primer algoritmo usado se basa en la detección de contornos. Para esto primero se aplicó un efecto difuso a la imagen en escala de grises con la función bilateralBlur con el fin de eliminar ruido pero conservar los bordes. 

```python
plate_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bilateral_blur = cv2.bilateralFilter(plate_gray,11,17,17)
```

Imagen original:

![image](https://user-images.githubusercontent.com/102924128/197419670-19a5e2bb-002c-4270-a3dd-5fd874a2822b.png)

Imagen después de efecto bilateral blur:

![image](https://user-images.githubusercontent.com/102924128/197419750-f1f3b523-c256-4aa3-bf26-d0b833a797c8.png)

Seguido de esto se uso el algoritmo de detección de bordes de Canny.

```python
edged = cv2.Canny(bilateral_blur, 30, 150)  
```

Imagen después de aplicar algoritmo de Canny:

![image](https://user-images.githubusercontent.com/102924128/197419769-680ce1ff-208c-417d-b3c0-a1505146cb83.png)

Después se hallaron los contornos y se realizó una aproximación poligonal filtrando únicamente aquellos polígonos conformados por 4 puntos. Lo anterior se realizó teniendo en cuenta que la placa en las imágenes suele tener una forma rectangular.

```python
contours = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

location = []
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 8, True)
    if(len(approx) == 4):
        location.append(approx)
```
![image](https://user-images.githubusercontent.com/102924128/197419791-318e267c-dabc-476f-ac1f-05df144cdf8d.png)

Finalmente se escoge el contorno con mayor área y esa es la región que se toma como posible placa.

```python
location = sorted(location, key = cv2.contourArea, reverse = True)
selected_contour = 0
white_background = np.zeros_like(plate_gray) + 255
cv2.drawContours(white_background, location, selected_contour, (0), 1)
```

![image](https://user-images.githubusercontent.com/102924128/197419813-4cf25fc5-7ef6-4643-a77a-8be1d7e7c3c3.png)


El código completo que muestra únicamente el resultado del algoritmo 1 se encuentra en el archivo [first_test.py](first_test.py). En el jupyter notebook [test1.ipynb](test1.ipynb) se encuentra el código con las imágenes paso a paso de lo previamente explicado.

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

El código completo se encuentra en el archivo [color.py](color.py).
Un ejemplo del resultado después de aplicar filtrado por color se puede ver a continuación.

Imagen original:


![image](https://user-images.githubusercontent.com/102924128/197416746-1ab3325d-42ae-490c-9709-84c32d00597d.png)


Imagen después de filtrado por color:

![image](https://user-images.githubusercontent.com/102924128/197416703-05f3f172-30b9-4224-939e-5120568b0dfb.png)

* **Contornos con hijos:** 

En este caso se hallaron primero los contornos de la imagen después de realizar una detección de bordes mediante el Algoritmo de Canny. Seguido de esto, se seleccionaron  ́unicamente aquellos contornos que tuviesen hijos, es decir, aquellos que tuviesen contornos en su interior. Esto se hace debido en la zona de la placa es muy probable que los caracteres, letras y números de la placa, aparezcan al interior de un contorno rectangular. A continuación se muestra un ejemplo de este proceso. El código completo se encuentra en el archivo [contornos_con_hijos.ipynb](contornos_con_hijos.ipynb).

Imagen con todos los contornos:

![image](https://user-images.githubusercontent.com/102924128/197416927-18923094-5cc7-40b6-a84f-7d2dc09a13b7.png)

Imagen únicamente con contornos con hijos:

![image](https://user-images.githubusercontent.com/102924128/197416999-dd64e507-76d5-4b42-aa36-9872ee144867.png)



## Conclusiones

La exploración de resultados mediante técnicas tradicionales permite un entendimiento a profundidad de cada
uno de los algoritmos que se implementan. Pese a su
rigidez y la alta dependencia de las condiciones de captura de la imagen, en comparación a sistemas que usan redes neuronales, se puede obtener un sistema con mayor robustez al usar un enfoque de métodos en cascada.

