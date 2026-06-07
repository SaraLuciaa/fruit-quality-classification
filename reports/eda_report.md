# Reporte de Analisis Exploratorio de Datos (EDA) - Datos y Metricas

Este reporte fue generado de manera automatica por el script de analisis exploratorio de datos y contiene los datos puros recopilados.

---

## 1. Estructura del Dataset y Distribucion de Clases

### Distribucion Detallada por Subcarpeta:
```
      Fruit Quality  Count
      Apple     Bad   1294
     Banana     Bad   1038
      Guava     Bad    859
       Lime     Bad    806
     Orange     Bad   1086
Pomegranate     Bad   1747
      Apple    Good   1293
     Banana    Good   1155
      Guava    Good    464
       Lime    Good    758
     Orange    Good   1144
Pomegranate    Good   1188
      Apple Regular    890
     Banana Regular    350
      Guava Regular    813
       Lime Regular    743
     Orange Regular    510
Pomegranate Regular    720
```

### Analisis del Desbalance por Clase de Calidad:
- **Bad**: 6830 imagenes (40.51%)
- **Good**: 6002 imagenes (35.60%)
- **Regular**: 4026 imagenes (23.88%)


---

## 2. Propiedades Fisicas de las Imagenes y Metricas de Calidad
Muestra aleatoria representativa analizada de 810 imagenes.

### Estadisticas Descriptivas de las Imagenes:
```
             Width       Height  Brightness    Contrast     Sharpness
count   810.000000   810.000000  810.000000  810.000000    810.000000
mean    691.091358   647.176543  140.885626   53.603679    392.572218
std     939.420687   903.030232   33.639458   13.397653    951.804242
min      22.000000    21.000000   38.254094   18.249194      1.716021
25%     192.000000   256.000000  115.484237   44.041464     67.484064
50%     256.000000   256.000000  140.354207   53.558945    137.307131
75%     416.000000   416.000000  165.463160   62.912187    309.112875
max    3120.000000  4032.000000  240.241857   94.465162  10084.640759
```

### Formatos de Archivo Encontrados:
```
Format
JPEG    692
PNG     118
```

---

## 3. Analisis Cromatico (Perfiles RGB de Referencia)
- **Ruta de imagen de muestra Buena**: `dataset\Good Quality_Fruits\Apple_Good\20190809_115439.jpg`
- **Ruta de imagen de muestra Mala**: `dataset\Bad Quality_Fruits\Apple_Bad\IMG-20260603-WA0146.jpg`
