import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Problema 1

def funcion1_tp_imagen(tamanio_ven : int) -> np.ndarray:
    """
    Ecualiza el histograma de una imagen, localmente.

    Parameters
    ----------
    imagen : Imagen la cual su histograma va a ser ecualizada.

    tamanio_ven : Tamaño de la ventana para la ecualización del histograma.
    
    Returns
    ----------
    imagen_equalizada : Imagen con su histograma ecualizado.
    """


    imagen = cv2.imread('Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE) # Leemos la imagen

    imagen_estirada = cv2.copyMakeBorder(imagen, tamanio_ven//2, tamanio_ven//2, tamanio_ven//2, tamanio_ven//2, cv2.BORDER_REPLICATE) # Se agrega un borde para evitar problemas con los indices fuera de rango

    imagen_equalizada = np.zeros_like(imagen) # Creamos una imagen de ceros del alto y ancho igual a la imagen original

    for i in range(imagen.shape[0]):
        for j in range(imagen.shape[1]):

            ventana_local = imagen_estirada[i:i+tamanio_ven, j:j+tamanio_ven] # Agarramos una imagen con los tamaños recortados

            equalizado_local = cv2.equalizeHist(ventana_local) # Ecualizamos el histograma

            imagen_equalizada[i, j] = equalizado_local[tamanio_ven//2, tamanio_ven//2] # Asiganmos la imagen ecualizada en los rangos dados a una nueva iamgen
    
    imagen_sin_ruido = cv2.medianBlur(imagen_equalizada, 3) # Aplicamos un filtro para el ruido salt and pepper.

    _, imagen_bianria = cv2.threshold(imagen_sin_ruido,70, 255, cv2.THRESH_BINARY) # binarisamos la imagen para corregir los valores que toadavia son grises, y pasarlos a blanco

    return imagen_bianria




# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------

# Problema 2


def encontrar_mayor_componente_horizontal(image_binaria : np.ndarray) -> tuple[int, tuple[int,int]]:
    """
    Esta función encuentra el componente conectado horizontal más grande.

    Parameters
    -----------
    image_binaria : Imagen a encontrar su mayor componente en formato np.ndarray.

    Returns
    -----------
    Componente_mayor : Recorte de la imagen del componente de mayor tamaño.
    max_ancho : Ancho del componente.
    centroid_max : Punto donde se encuentra el centroide del componente.
    """

    image_invertida = cv2.bitwise_not(image_binaria) # Invertimos la imagen binaria.

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(image_invertida) 

    max_ancho = 0   # Inicializar variables para almacenar el componente más ancho

    for i in range(1, num_labels):  # Recorrer todos los componentes conectados (excluyendo el fondo)
        ancho = stats[i, cv2.CC_STAT_WIDTH] # Obtenemos el ancho
        alto = stats[i, cv2.CC_STAT_HEIGHT] # Obtenemos el alto

        if ancho > alto and ancho > max_ancho: # Encontramos el ancho más grande de los componentes conectados
            max_ancho = ancho
            centroid_max = centroids[i] # Guardamos el centroide para luego facilitar los calculos

    return max_ancho, centroid_max



# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------

def resolver_examenes() -> dict[str : list[dict[str : str], dict[str : str], np.ndarray]]:
    """
    Esta función devuelve un diccionario con todos los examenes corregidos y la validación de los datos de encabezado.

    Returns
    -----------
    examenes :  Un diccionario donde la clave es cada examen y los valores son:
                                                                - Un diccionario donde la clave es el nro de pregunta y el valor su corrección.
                                                                - Un diccionario donde la clave son: Name, Date, Class, Condicion y sus
                                                                    respectivos valores: "OK","MAL"
                                                                - Un np.ndarray que es el recorte del nombre de la imagen del examen
    """
    

    notas = ['C','B','A','D','B','B','A','B','D','D']       # Lista con las respuestas correctas del examen

    imagenes = []                                           # Lista donde se van a guardar las imagenes de los examenes

    num_elementos = len(os.listdir('./examenes'))
    
    for i in range(1,num_elementos+1):
        imagenes.append(cv2.imread(f'./examenes/examen_{i}.png', cv2.IMREAD_GRAYSCALE)) # Recorremos todas las imagenes de la carpeta y las guardamos en una lista

    examenes = {}
    indx = 0

    for examen in imagenes:
        indx += 1                       # Indice auto-incremental
        recuadros_lista = []            # Lista para guardar los recortes de las preguntas de la imagen del examen
        condicion_alumno = {}           # Vamos a guardar: Nombre, Fecha y Clase. 
        letra_respondida = ''           # Variable para comparar la respuesta correcta y la seleccionada
        respuesta_examen = {}           # Diccionario para guardar las correcciones de cada pregunta
        cant_respuestas_buenas = 0      # Contador de respuestas buenas    

        # ----------------------------------------------------------------
        # ----------------------------------------------------------------
        # ----------------------------------------------------------------

        img = examen                                                                    # Guardamos la imagen
        img_bin_col = img<90                                                            # Aplicamos Umbralizado
        _, img_th1 = cv2.threshold(img,150,255,cv2.THRESH_BINARY)                       # Binarizamos 
        
        
        sumas_column = np.sum(img_bin_col, axis=0)                                      # Calcula la suma de los elementos de img_bin_col a lo largo del eje 0, resultando en un array que contiene la suma de cada columna
        lineas_verticales = np.where(sumas_column >= (np.max(sumas_column)) *0.8)[0]    # Encuentra los índices de las columnas donde la suma es mayor o igual al 80% del valor máximo de  
        sumas_fil = np.sum(img_th1, axis=1)                                             # Calcula la suma de los elementos de img_th1 a lo largo del eje 1
        lineas_horiz = np.where(sumas_fil <= np.min(sumas_fil) *2)[0][1:]               # Encuentra los índices de las filas donde la suma es menor o igual al doble del valor mínimo de sumas_fil, omitiendo el primer índice encontrado (por el uso de [0][1:])

        # ----------------------------------------------------------------
        # ----------------------------------------------------------------
        # ----------------------------------------------------------------

        encabezado =  img_th1[0:lineas_horiz[0]-3,0:lineas_verticales[-1]]   # Cortamos la matrix en una sub-matirz, esta contiene contiene todas las filas antes de la primer linea horizontal y todas las columnas hasta la última línea vertical
        _ , cord_central_guiones_encabezado = encontrar_mayor_componente_horizontal(encabezado) 
        
        titulo_linea = img_th1[int(cord_central_guiones_encabezado[1]),:]   # Recortamos el pixel donde se encuentran las lineas de los datos del alumno 
        
        inicios_lineas = np.diff(titulo_linea)                  # Hacemos las diferencias del pixel para encontrar principio y final de cada renglon de los datos. 
        cords_inicios_lineas = np.where(inicios_lineas != 0)[0] # Encontramos en que puntos se encuentran estos renglones

        nombre_examen = img_th1[int(cord_central_guiones_encabezado[1])-20:int(cord_central_guiones_encabezado[1])-2, cords_inicios_lineas[0]:cords_inicios_lineas[1]]  # Extraigo una sub-matriz de img_th1 20 pixeles por encima y 2 por debajo de su coordenada central de guión, horizontalmente desde el principio hasta el final
        fecha_examen = img_th1[int(cord_central_guiones_encabezado[1])-20:int(cord_central_guiones_encabezado[1])-2,cords_inicios_lineas[2]:cords_inicios_lineas[3]]    
        clase_examen = img_th1[int(cord_central_guiones_encabezado[1])-20:int(cord_central_guiones_encabezado[1])-2,cords_inicios_lineas[4]:cords_inicios_lineas[5]]    
        
        nombre_examen_alt = cv2.bitwise_not(nombre_examen)  # Invertimos los valores de la imagen binaria
        fecha_examen_alt = cv2.bitwise_not(fecha_examen)
        clase_examen_alt = cv2.bitwise_not(clase_examen)

        cant_caracteres_nombre , _ = cv2.connectedComponents(nombre_examen_alt, connectivity=8) # Encuentro los componentes 8 conectados de la imagen recortada
        cant_caracteres_fecha, _ = cv2.connectedComponents(fecha_examen_alt, connectivity=8)
        cant_caracteres_clase, _ = cv2.connectedComponents(clase_examen_alt, connectivity=8)


        if (cant_caracteres_nombre-1) > 25 or (cant_caracteres_nombre-1) == 0:  # Verificamos si la cantidad de caracteres es cero o mayor a 25
            condicion_alumno['Name'] = 'MAL'
        else:    
            letra_aplastada = np.sum(nombre_examen_alt,axis=0,dtype=np.bool)        # Buscamos sobre el eje x (en formato imagen) pixeles donde haya valores mayores a 0, es decir, donde hay letras.
            principio_final_letra = np.diff(letra_aplastada)                        # Haciendo las diferencias podemos reconocer el principio y final de cada letra, además reconocemos los espacios entre palabras.
            letras_aplastada_cords = np.where(principio_final_letra != False)[0]    # Encontramos las coordenadas de las letras.

            for indx_fl in range(1,len(letras_aplastada_cords)-1,2):                          # Recorremos las cordenadas de las letras-1 en saltos de rana de 2 en 2 XD
                espacio = letras_aplastada_cords[indx_fl+1] - letras_aplastada_cords[indx_fl] # Realizamos distancia euclidia de coordenadas
                if espacio > 3:                                                               # Y asumimos que si dicha distancia es mayor a 3 pixles se trata de dos palabras distintas
                    condicion_alumno['Name'] = 'OK'
            if not condicion_alumno:                                                          # En caso de no encontrar al menos 2 palabras decimos que el nombre está mal
                condicion_alumno['Name'] = 'MAL'
        
        if (cant_caracteres_fecha-1) == 8:          # Comprobamos los 8 caracteres del campo fecha
            condicion_alumno['Date'] = 'OK'
        else:
            condicion_alumno['Date'] = 'MAL'
        
        if (cant_caracteres_clase-1) == 1:          # Comprobamos el caracter de la clase.
            condicion_alumno['Class'] = 'OK'
        else:
            condicion_alumno['Class'] = 'MAL'


        # ----------------------------------------------------------------
        # ----------------------------------------------------------------
        # ----------------------------------------------------------------


        for i in range(0,len(lineas_verticales),2): # Iterramos sobre los recuadros, primero sobre las lineas verticales, para seguir el orden de las preguntas
            for j in range(0,len(lineas_horiz)-1):
                img_recuadro = img_th1[lineas_horiz[j]+1:lineas_horiz[j+1]-2,lineas_verticales[i]+2:lineas_verticales[i+1]] # Recortamos los recuadros de la imagen del examen
                recuadros_lista.append(img_recuadro)    # Lo guardamos en una lista, manteniendo su orden.


        for respuesta in range(len(recuadros_lista)):   # Iteramos las respuestas
            ancho_maximo, cord_central = encontrar_mayor_componente_horizontal(recuadros_lista[respuesta])   # Encontramos el renglon de la respuesta.
            cord_linea = recuadros_lista[respuesta][int(cord_central[1])-14:int(cord_central[1]),int(cord_central[0])-int(ancho_maximo//2):int(cord_central[0])+int(ancho_maximo//2)]   # Recortamos el recuadro para conseguir solamente el espacio donde va la respuesta.
            cord_linea_alt = cv2.bitwise_not(cord_linea)    # Invertimos la imagen.

            num_labels, _, _, _ = cv2.connectedComponentsWithStats(cord_linea_alt, connectivity=8)  # Encontramos los componentes 8 conectados, es decir, la o las letras respondidas.
            if num_labels <= 1 or num_labels >= 3:  # Como el fondo es un componente conectado la cantidad de componentes deberia ser 2.
                respuesta_examen[f'Pregunta {respuesta+1}'] = 'MAL'
            else:
                contornos, _ = cv2.findContours(cord_linea_alt,cv2.RETR_LIST ,cv2.CHAIN_APPROX_NONE)    # Encontramos los contornos de las letras.
                if len(contornos) == 1: # Si tiene 1 solo contorno es la letra C.
                    letra_respondida = 'C'
                elif len(contornos) == 3:   # Si tiene 3 contornos es la letra B.
                    letra_respondida = 'B'
                else:   # Si tiene 2 contornos puede ser la letra A o D.
                    pixeles_verticales = np.sum(cord_linea_alt,axis=0)  # Sumamos verticalmente los pixeles del recorte de la letra.
                    primer_pixel_dif = np.where(pixeles_verticales > 0)[0]  # Encontramos las cordenadas de la letra.
                    if primer_pixel_dif.any():  # Si hay alguna coordenada de letra, es decir, si se respondió.
                        if pixeles_verticales[primer_pixel_dif[0]] > 255*5: # Si tiene la primera columna de la letra más de 5 pixeles pintados
                            letra_respondida = 'D'
                        else:   # Si no los tiene es la letra A
                            letra_respondida = 'A'

                if letra_respondida == notas[respuesta]:    # Corregimos la respuesta.
                    respuesta_examen[f'Pregunta {respuesta+1}'] = 'OK'
                    cant_respuestas_buenas += 1                    
                else: 
                    if letra_respondida != notas[respuesta]:
                        respuesta_examen[f'Pregunta {respuesta+1}'] = 'MAL'

        if cant_respuestas_buenas >= 6: # Si el contador de respuestas correctas es mayor o igual a 6 esta aprobado
            condicion_alumno['Condicion'] = 'A'
        else:   # En el caso contrario esta reprobado.
            condicion_alumno['Condicion'] = 'R'

        examenes[f'examen {indx}'] = [respuesta_examen,condicion_alumno,nombre_examen]  # Guardamos toda la información del examen, alumno y recorte del nombre del alumno en el diccionario.
        alto_img = nombre_examen.shape[0]*(len(imagenes))   # Guardamos el alto de la imagen a generar.
        ancho_img = nombre_examen.shape[1]                  # Guardamos el ancho de la imagen a generar.




    imagen_respuestas_alumnos = np.ones((alto_img,ancho_img+30,3),np.uint8)*255 # Creamos una imagen blanca de tamaño igual a la suma del tamaño de los nombres, al ancho le sumamos 30 pixeles

    imagen_respuestas_alumnos_un_canal = np.zeros((alto_img,ancho_img+30),np.uint8) # Por conveniencia creamos la misma imagen que antes pero con valores 0

    indx_img = 0

    for _, examen_evaluado_value in examenes.items():   # Recorremos todos los valores de nuestro diccionario (examen_evaluado_value[2] es un np.array con el nombre del alumno)
        imagen_respuestas_alumnos[examen_evaluado_value[2].shape[0]*indx_img:examen_evaluado_value[2].shape[0]*(indx_img+1),0:examen_evaluado_value[2].shape[1]] = cv2.cvtColor(examen_evaluado_value[2],cv2.COLOR_GRAY2RGB)    # En la nueva imagen insertams el nombre del alumno y lo pasamos a 3 canales
        cv2.line(imagen_respuestas_alumnos, (0,examen_evaluado_value[2].shape[0]*indx_img), (imagen_respuestas_alumnos.shape[1],examen_evaluado_value[2].shape[0]*(indx_img)),(0,0,0),1,cv2.LINE_4) # trazamos lineas horizontales para crear un cuadro
        recuadro_condicion = imagen_respuestas_alumnos_un_canal[examen_evaluado_value[2].shape[0]*indx_img:examen_evaluado_value[2].shape[0]*(indx_img+1),ancho_img:imagen_respuestas_alumnos.shape[1]] # Hacemos un recorte en los "recuadros" donde vamos a señalizar la condición del alumno 

        _ , _, _, centroide_condicion = cv2.connectedComponentsWithStats(recuadro_condicion)    # Obtenemos el centroide del recuadr

        if examen_evaluado_value[1]['Condicion'] == 'A':    # Graficamos el circulo según la condición del alumno
            cv2.circle(imagen_respuestas_alumnos,(int(centroide_condicion[0][0])+ancho_img,int(centroide_condicion[0][1]+examen_evaluado_value[2].shape[0]*indx_img)+1),7,(0,255,0),-1) #Graficamos el circulo verde para las respuestas correctas de 7 de radio
        else:
            cv2.circle(imagen_respuestas_alumnos,(int(centroide_condicion[0][0])+ancho_img,int(centroide_condicion[0][1]+examen_evaluado_value[2].shape[0]*indx_img)+1),7,(255,0,0),-1) #Graficamos el circulo rojo para las respuestas incorrectas de 7 de radio

        indx_img += 1

    cv2.line(imagen_respuestas_alumnos, (ancho_img,0), (ancho_img,imagen_respuestas_alumnos.shape[0]),(0,0,0),1,cv2.LINE_4) #Creamos la linea que separa la imagen de los nombres con el circulo de aprobacion o desaprobacion

    for nro_examen , info in examenes.items():  # Imprimimos toda la información requerida por el ejercicio.
        print(f'Correcciones del {nro_examen}')
        for pregunta, correc in info[0].items():
            print(f'{pregunta}: {correc}')
        for dato_encabezado, valid in info[1].items():
            print(f'{dato_encabezado}: {valid}')

    plt.imshow(imagen_respuestas_alumnos),plt.show()

    return examenes



# Menu

while True:
    print('1: Ver problema 1.\n2: Resolver examenes\n3: Salir del código\n')
    opcion = int(input("Ingrese su opción: "))
    match opcion:
        case 1:
            imagen_equalizada = funcion1_tp_imagen(20)
            plt.imshow(imagen_equalizada,cmap='gray'),plt.show()

        case 2:
            datos = resolver_examenes()

        case 3:
            break