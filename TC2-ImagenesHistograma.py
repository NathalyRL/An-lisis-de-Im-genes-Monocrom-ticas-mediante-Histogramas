import cv2
import matplotlib.pyplot as plt
import numpy as np

# Cargar la imagen a color y frames N y N1
ubicacion = r"D:\Documentos\COM MULTIMEDIA\Trabajo 2\imagen_color.jpg"
ruta_N = r"D:\Documentos\COM MULTIMEDIA\Trabajo 2\N.jpg"
ruta_N1 = r"D:\Documentos\COM MULTIMEDIA\Trabajo 2\N1.jpg"

#Leer las imágenes
N = cv2.imread(ruta_N)
N1 = cv2.imread(ruta_N1)

# cv2.imread carga la imagen en orden B-G-R, no R-G-B.
# Por eso se usa el comando COLOR_BGR2GRAY

imagen = cv2.imread(ubicacion)

# ---------- IMAGEN EN ESCALA DE GRISES ----------
# Convertir a blanco y negro (escala de grises) BGR
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Para Matplotlib R-G-B
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

# Graficar la imagen original y en escala de grises
fig, ejes = plt.subplots(1, 2, figsize=(12, 6))

ejes[0].imshow(imagen_rgb)
ejes[0].set_title("Imagen original")
ejes[0].axis("off")

ejes[1].imshow(gris, cmap="gray")
ejes[1].set_title("Escala de grises")
ejes[1].axis("off")

# ----- HISTOGRAMA DE LA IMAGEN EN ESCALA DE GRISES -------
hist = cv2.calcHist([gris], [0], None, [256], [0, 256])

# Grafica del histograma
fig_hist, ejes_hist = plt.subplots(
	2, 1, figsize=(10, 6), sharex=True,
	gridspec_kw={"height_ratios": [5, 0.35], "hspace": 0.08}
)

ejes_hist[0].bar(
    range(256),
    hist.ravel(),
    color="plum",
    edgecolor="purple",
    linewidth=0.8,
    width=1
)

ejes_hist[0].set_title("Histograma de la imagen en escala de grises")
ejes_hist[0].set_xlim(0, 255)

#Barra de color de referencia del histograma
barra_grises = np.arange(256, dtype=np.uint8).reshape(1, -1)
ejes_hist[1].imshow(barra_grises, cmap="gray", aspect="auto", extent=[0, 255, 0, 1])
ejes_hist[1].set_yticks([])
ejes_hist[1].set_xlabel("Nivel de gris: negro (0) a blanco (255)")

fig_hist.tight_layout()

plt.tight_layout()

# ---------- IMAGEN NEGATIVA ----------
negativo = cv2.bitwise_not(gris)

#Grafica de imagen negativa
fig_neg, eje_neg = plt.subplots(figsize=(6, 6))
eje_neg.imshow(negativo, cmap="gray", vmin=0, vmax=255)
eje_neg.set_title("Imagen negativa")
eje_neg.axis("off")
fig_neg.tight_layout()

# ---------- HISTOGRAMA DE LA IMAGEN NEGATIVA ----------
hist_neg = cv2.calcHist([negativo], [0], None, [256], [0, 256])

#Grafica de histograma
fig_hist_neg, ejes_hist_neg = plt.subplots(
	2, 1, figsize=(10, 6), sharex=True,
	gridspec_kw={"height_ratios": [5, 0.35], "hspace": 0.08}
)

ejes_hist_neg[0].bar(
    range(256),
    hist_neg.ravel(),
    color="skyblue",
    edgecolor="blue",
    linewidth=0.8,
    width=1
)
ejes_hist_neg[0].set_title("Histograma de la imagen negativa")
ejes_hist_neg[0].set_xlim(0, 255)

ejes_hist_neg[1].imshow(barra_grises, cmap="gray", aspect="auto", extent=[0, 255, 0, 1])
ejes_hist_neg[1].set_yticks([])
ejes_hist_neg[1].set_xlabel("Nivel de gris: negro (0) a blanco (255)")

fig_hist_neg.tight_layout()


# ------------ RESTA DE LA IMAGEN ------------
# Verificar que se cargaron los frames
if N is None or N1 is None:
    raise FileNotFoundError("No se pudo leer N o N1")

# Escala de grises
N_gris = cv2.cvtColor(N, cv2.COLOR_BGR2GRAY)
N1_gris = cv2.cvtColor(N1, cv2.COLOR_BGR2GRAY)

# Resta de los frames
residual = cv2.absdiff(N1_gris, N_gris)

# Grafica del residual
fig_res, ejes_res = plt.subplots(1, 1, figsize=(12, 6))
ejes_res.imshow(residual, cmap="gray", vmin=0, vmax=255)
ejes_res.set_title("Frame residual")
ejes_res.axis("off")
fig_res.tight_layout()

plt.show()