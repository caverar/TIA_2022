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

Después de obtener la posible zona de la placa, se procede a verificar si efectivamente corresponde a una placa. Para esto, se evalúan cinco diferentes condiciones en donde se tienen como parámetros la altura y ancho de la imagen, la cantidad de píxeles blancos y negros después de realizar una umbralización y la cantidad de contornos presentes en la región.

* Altura de la imagen > 10 píxeles
* Ancho > 50 píxeles
* (Altura / Ancho) > 20
* Contornos > 10 
* (Píxeles Blancos - Píxeles negros) > 0

A continuación un fragmento del código que evalúa esas condiciones en la imagen de la posible zona de la placa que se recibe.

```python
plate_zone_gray = cv2.cvtColor(plate_zone, cv2.COLOR_BGR2GRAY)
plate_zone_binary = cv2.threshold(plate_zone_gray, 127, 255, cv2.THRESH_BINARY)[1]
h_plate, w_plate  = plate_zone_binary.shape

min_height = 10
bool_h = h_plate > min_height

min_width = 50
bool_w = w_plate > min_width

height_width_relation = (h_plate / w_plate) * 100
bool_h_w = height_width_relation > 20 

plate_contours = cv2.findContours(plate_zone_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]  
bool_contours = len(plate_contours) > 10

hist = cv2.calcHist([plate_zone_binary],[0],None,[256],[0,256])
white_pixels = hist[-1]
black_pixels = hist[0]
black_white_factor = 0

bool_white_black = (white_pixels - black_pixels) > black_white_factor
```
Esta verificación se asemeja a una compuerta lógica AND dado que retorna un valor de *true*, es decir que una placa fue encontrada, únicamente cuando las cinco condiciones se cumplen.

```python
plate_found = bool_h and bool_w and bool_h_w and bool_contours and  bool_white_black
```

El código completo se puede encontrar en el archivo [verify_plate_image.py](verify_plate_image.py).

### Algoritmo 2: Bordes verticales

En el algoritmo 2 se buscan los bordes verticales de la región rectangular que comprende a la placa. Para esto se toma la imagen original y se aplica un efecto difuso seguido de una detección de bordes mediante el algoritmo de Canny.

Imagen original:
![image](https://user-images.githubusercontent.com/102924128/197420620-f2f344ab-91d6-4729-8457-a5a69f15a798.png)

```python
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

bilateral_blur = cv2.bilateralFilter(img_gray,11,17,17)
edged = cv2.Canny(bilateral_blur, 240, 250)
```
Imagen después de bilateral blur y algoritmo de Canny:
![image](https://user-images.githubusercontent.com/102924128/197420636-72cf3cdd-1dba-4feb-822e-8abb66f5ef4b.png)


La imagen resultante es tratada con transformaciones morfológicas. Se realiza una erosión con un elemento estructurante en forma de vector de 5 x 1 para poder detectar bordes verticales. Seguido de esto se realiza un cierre con una matriz de 5x5 píxeles con el fin de eliminar ruido para finalmente volver a aplicar una erosión con un vector de 5 x 1. 

```python
kernel_v_line = np.ones((5, 1), np.uint8)

erode_v = cv2.erode(edged, kernel_v_line) 

kernel = np.ones((5,5),np.uint8)
closing = cv2.morphologyEx(erode_v, cv2.MORPH_CLOSE, kernel)

processed_image = cv2.erode(closing, kernel_v_line) 
```
Imagen después de transformaciones morfológicas:

![image](https://user-images.githubusercontent.com/102924128/197420658-49b07480-8e70-4572-ba76-859ed72117b2.png)

Después de las transformaciones morfológicas se realiza un barrido de la imagen y se seleccionan aquellas columnas y filas que tengan mayor número de píxeles blancos. El área encerrada por esas filas y columnas es la posible zona de la placa.

```python
white_in_rows = np.zeros(processed_image.shape[0])
for i, row in enumerate(processed_image):
    for value in row:
        if value == 255: white_in_rows[i]+=1
        
white_in_columns = np.zeros(processed_image.shape[1])
for i in range(processed_image.shape[1]):
    for value in processed_image[:, i]:
        if value == 255: white_in_columns[i]+=1

min_white_pixels_columns = 20
min_white_pixels_rows = 6
selected_rows = [i for i, row in enumerate(white_in_rows) if row > min_white_pixels_rows]
selected_columns = [i for i, column in enumerate(white_in_columns) if column > min_white_pixels_columns]
```
Antes de retornar la posible región de la imagen que corresponde a la zona de la placa, se hace una traslación de las columnas y filas seleccionadas con el fin de tener un factor de seguridad que permita encerrar la zona de la placa con cierta tolerancia y evitar que algún borde quede recortado.

```python
row_security_factor = 13
column_security_factor = 3

x1 = min(selected_rows)
x2 = max(selected_rows)
y1 = min(selected_columns)
y2 = max(selected_columns)

if x1 - row_security_factor > 0: x1 -= row_security_factor
if y1 - column_security_factor > 0: y1 -= column_security_factor
if x2 + row_security_factor > 0: x2 += row_security_factor
if y2 + column_security_factor > 0: y2 += column_security_factor

plate_zone2 = img[x1:x2, y1:y2]
```
Región de la imagen con posible zona de la placa:

![image](https://user-images.githubusercontent.com/102924128/197420724-faa2fa66-fd8e-43cb-8ffa-370452dc84db.png)

El código completo que muestra únicamente el resultado del algoritmo 1 se encuentra en el archivo [second_test.py](second_test.py). En el jupyter notebook [test2.ipynb](test2.ipynb) se encuentra el código con las imágenes paso a paso de lo previamente explicado.

### Programa completo

El programa completo se encuentra en el archivo [plate_zone_detector.py](plate_zone_detector.py) que realiza la función de wrapper de tanto el algoritmo 1 y 2 como del algoritmo de verificación de la placa.

A continuación se muestra la función correspondiente:

```python
def detect_plate(img):
  """
  Detects zone plate from an image using two different methods (contours and vertical boundaries). 
  If zone plate is not found, original image is returned

  Parameters:
      img (ndarray): image in BGR

  Returns:
      bool: Found plate
      ndarray: possible zone plate
      string: info of plate verification process

  """
  b1, i1, m1 = test_plate_method_1(img)
  if not b1:
    return test_plate_method2(img)
  return b1, i1, m1
```

Como entrada a la función se tiene la imagen de la cual quiere obtenerse la placa. En primera instancia, se busca detectar la placa mediante el algoritmo 1, usando contornos y aproximación poligonal. En caso de que no se detecte la placa por este método, se procede a usar el algoritmo 2 enfocado en encontrar bordes verticales y se verifica si la región hallada corresponde a la placa. Si se comprueba que la zona corresponde a una placa, se retorna la imagen de esa región. En caso contrario, se retorna la imagen original.

Esta función recibe la imagen como arreglo ndarray en BGR y retorna tres valores.

* boolean: Este valor retorna True si la zona de la placa fue encontrada y False si no se detectó la placa en la imagen
* ndarray: Retorna la zona de la placa como imagen. Si la placa no fue detectada, retorna la imagen original
* string: Reporta un mensaje con el algoritmo por el cual se encontró la placa, o la razón por la cual no se halló, y los resultados de la verificación de la placa.

Un ejemplo del uso de esta función se encuentra en el archivo [example_test.ipynb](example_test.ipynb).

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

