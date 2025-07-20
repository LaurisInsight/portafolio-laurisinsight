#  Análisis exploratorio de datos de Airbnb Buenos Aires​



#Diccionario de datos

*   Index - Variable: Descripción | Dtype
*   0 - id: Identificador único de Airbnb para el anuncio | int64
*   1 - name: Nombre del anuncio | object
*   2 - host_id: Identificador único de Airbnb para el anfitrión | int64
*   3 - host_name: Nombre del anfitrión | object
*   4 - neighbourhood_group: Distrito | float64
*   5 - neighbourhood: Barrio | object
*   6 - latitude: Latitud geográfica | float64
*   7 - longitude: Longitud geográfica | float64
*   8 - room_type: Tipo de alojamiento (propiedad completa, habitación privada, habitación compartida, habitación de hotel) | object
*   9 - price: Precio por noche moneda local | float64
*   10 - minimum_nights: Número mínimo de noches de estadía (estipulado por el host) | int64
*   11 - number_of_reviews: Número de reseñas | int64
*   12 - last_review: Fecha de la última reseña | object
*   13 - reviews_per_month: Número de reseñas por mes | float64
*   14 - calculated_host_listings_count: Número de anuncios que tiene el anfitrión en el scrape actual  en la ciudad/región geográfica | int64
*   15 - availability_365: Disponibilidad del anuncio en días en el futuro (sobre los siguientes 365 días posteriores al scrapeo) | int64
*   16 - number_of_reviews_ltm: Número de reseñas en los últimos 12 meses | int64
*   17 - license: Número de licencia/permiso/registro | object
*   18 - host_is_superhost: Indica si el host es superhost | object
*   19 - bedrooms: Número de habitaciones | float64
*   20 - review_scores_rating: Calificación promedio (experiencia general) | float64
*   21 - review_scores_accuracy: Calificación promedio (precisión) | float64
*   22 - review_scores_cleanliness: Calificación promedio (limpieza) | float64
*   23 - review_scores_checkin: Calificación promedio (llegada) | float64
*   24 - review_scores_communication: Calificación promedio (comunicación) | float64
*   25 - review_scores_location: Calificación promedio (ubicación) | float64
*   26 - review_scores_value: Calificación promedio (relación calidad-precio) | float64
*

# Carga del dataset

## Importamos las librerías que vamos a utilizar
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import geopandas as gpd

"""## Consumo de los datos"""

ruta = "listings.csv" #dataset resumindo
data_res = pd.read_csv(ruta)
data_res_raw = pd.read_csv(ruta)

ruta_det = "listings2.csv.gz" #dataset completo (solo usaremos algunas variables)
data_det = pd.read_csv(ruta_det)
data_det_raw = pd.read_csv(ruta_det)

# Configuración para ver todas las columnas en la consola
pd.set_option('display.max_columns', None)   # muestra todas las columnas
pd.set_option('display.width', None)         # no limita el ancho total

# Lista de columnas a traer desde data_det
cols_to_merge = [
    "id",  # clave para unir
    "host_is_superhost",
    "bedrooms",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value"
]

# Hacemos el merge, unimos al dataset con que vamos a trabajar las columnas del dataset completo
data = data_res.merge(data_det[cols_to_merge], on="id", how="left")

"""# Inspección inicial del dataset

## Tamaño del dataset
"""

filas, columnas = data.shape
print(f"El dataset tiene {filas} instancias y {columnas} variables.")

"""## Visualización previa de instancias"""

# Visualización de las primeras filas del dataset
data.head()

# Visualización de las 5 últimas filas del dataset.
data.tail()

"""# Limpieza del dataset

## Busqueda de instancias repetidas
"""

# Verificamos que no haya instancias repetidas

duplicated_rows = data.duplicated()

if duplicated_rows.any()==False:
  print('No hay instancias repetidas')
else:
  print('Hay '+duplicated_rows.sum()+ ' instancias repetidas')

"""Convertimos las variables con los identificadores de anuncios y anunciantes a categóricas."""

# Convertir 'id' y 'host_id' a tipo categórico
data['id'] = data['id'].astype('category')
data['host_id'] = data['host_id'].astype('category')

"""## Columnas irrelevantes

* Eliminamos las siguientes columnas:

- name: nombre del anuncio, consideramos **irrelevante**.
- host_name: nombre de pila del anunciantes, _a priori_ la consideramos **irrelevante**.
- number_of_reviews y number_of_reviews_ltm: estas dos variables, al igual que reviews_per_month pueden interpretarse como indicadores del volumen de visitas a un alojamiento (a más reseñas, más visitas). Dado que son muy similares entre sí, optamos por conservar solo reviews_per_month. Esta elección se basa en que la consideramos menos susceptible a sesgos relacionados con la antigüedad del alojamiento (las otras dos podrían mostrar pocos comentarios porque el anuncio es más reciente y no por ser poco visitado).
- calculated_host_listings_count: consideramos **irrelevante** para nuestra pregunta
"""

# Eliminamos las columnas que no aportan información relevante para el análisis.
data.drop(['name', 'host_name', 'number_of_reviews', 'calculated_host_listings_count'], axis=1, inplace=True)

# Chequeamos que haya cambiado el número de columnas
filas, columnas = data.shape
print(f"El dataset tiene {filas} instancias y {columnas} variables.")

"""## Busqueda de missing values"""

# Buscamos valores missing values en el dataset.

total_filas = data.shape[0]
columnas_con_nulos = []
columnas_sin_nulos = []

for col in data.columns:
    nulos = total_filas - data[col].notnull().sum()
    if nulos > 0:
        columnas_con_nulos.append((col, nulos))
    else:
        columnas_sin_nulos.append(col)

# Chequeamos si hay missing values
if not columnas_con_nulos:
    print("No hay valores nulos en el dataset.")
else:
    for col, nulos in columnas_con_nulos:
        print(f"La columna {col} tiene {nulos} valores nulos.")
    if columnas_sin_nulos:
        print(f"\nLas columnas {', '.join(columnas_sin_nulos)} no tienen valores nulos.")

# Resumen de tipos de datos y porcentaje de valores no nulos por columna
def resumen_info(df):
    total = len(df)
    resumen = pd.DataFrame({
        'Tipo de dato': df.dtypes,
        'No nulos': df.notnull().sum(),
        '% No nulos': df.notnull().mean() * 100
    })
    return resumen

resumen_info(data)

"""Se observa que existen algunas variables con porcentajes diversos de valores no nulos, el tramiento que recibirán (p.e. eliminación de instancias, eliminación de variables, etc.) será diferente en cada caso.

### Tratameinto de columnas con missing values

#### Eliminación de columnas: neighbourhood_group y license

Dada la gran cantidad de missing values, eliminamos dos columnas más:

- neighbourhood_group: Columna vacia.
- licence: 98,9% de missing values.
"""

# Eliminamos las columnas mencionadas
data.drop(['neighbourhood_group', 'license'], axis=1, inplace=True)

# Chequeamos que haya cambiado el número de columnas
filas, columnas = data.shape
print(f"El dataset tiene {filas} instancias y {columnas} variables.")

"""#### Eliminación de instancias: price, bedrooms, host_is_superhost

Dada la importancia de ests variables para nuestro análisis, y el momento del análisis decidimos eliminar las filas con missing values en alguna de estas tres variables.
"""

data.dropna(subset=['price', 'bedrooms', 'host_is_superhost'], inplace=True)

"""#### Tratamiento de variables: last_review

Los missing values en esta variable no son errores de carga sino publicaciones sin reviews. Para salvar esto y a la vez pasar de una variable que continene fechas a una categorica, calculamos una nueva variable con el número de días desde el ultimo reviews (conocienod la fecha de realizado el scraping) y luego la bineamos.
"""

# Creamos variable days_since_last_review
fecha_scraping = pd.to_datetime('2025-01-30')
data['last_review'] = pd.to_datetime(data['last_review'], errors='coerce')
data['days_since_last_review'] = (fecha_scraping - data['last_review']).dt.days

# Bineamos la variable days_since_last_review
bins_days_since_last_review = [-1, 30, 90, 180, 365, float('inf')]
labels_days_since_last_review = ['0-30','31-90','91-180', '181-365', '>365']
data['days_since_last_review_binned'] = pd.cut(data['days_since_last_review'], bins=bins_days_since_last_review, labels=labels_days_since_last_review, right=True)

# Asignamos los nulos (sin reviews) a una nueva categoria
data['days_since_last_review_binned'] = data['days_since_last_review_binned'].cat.add_categories('sin reviews')
data['days_since_last_review_binned'] = data['days_since_last_review_binned'].fillna('sin reviews')

# Eliminamos las columnas last_review y days_since_last_review
data.drop(['last_review', 'days_since_last_review'], axis=1, inplace=True)

data['days_since_last_review_binned'].value_counts()

"""#### Tratamiento de variables: reviews_per_month
En esta variable decidimos reemplazar los missing values por ceros, ya que no se trata de errores de carga sino de publicaciones sin reviews.
"""

data['reviews_per_month'] = data['reviews_per_month'].fillna(0)

"""#### Tratamiento de variables: Scores
Buscamos nuevamente missing values
"""

# Buscamos missing values en el dataset.

total_filas = data.shape[0]
columnas_con_nulos = []
columnas_sin_nulos = []

for col in data.columns:
    nulos = total_filas - data[col].notnull().sum()
    if nulos > 0:
        columnas_con_nulos.append((col, nulos))
    else:
        columnas_sin_nulos.append(col)


if not columnas_con_nulos:
    print("No hay valores nulos en el dataset.")
else:
    for col, nulos in columnas_con_nulos:
        print(f"La columna {col} tiene {nulos} valores nulos.")
    if columnas_sin_nulos:
        print(f"\nLas columnas {', '.join(columnas_sin_nulos)} no tienen valores nulos.")

"""En el caso de las variables de scores, decidimos reemplazar los missing values de los scores por -1"""

# Lista de columnas de scores a tratar
score_columns = [
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value"
]

# Reemplazar los valores nulos en las columnas de scores por -1
for col in score_columns:
    data[col] = data[col].fillna(-1)

# Verificar si aún hay nulos en esas columnas (debería mostrar 0 nulos)
for col in score_columns:
    nulos = data[col].isnull().sum()
    print(f"La columna {col} tiene {nulos} valores nulos.")

"""Para finalizar, comprobamos que no hayan quedado missing values"""

# Buscamos valores missing values en el dataset.

total_filas = data.shape[0]
columnas_con_nulos = []
columnas_sin_nulos = []

for col in data.columns:
    nulos = total_filas - data[col].notnull().sum()
    if nulos > 0:
        columnas_con_nulos.append((col, nulos))
    else:
        columnas_sin_nulos.append(col)

# Chequeamos si hay missing values
if not columnas_con_nulos:
    print("No hay valores nulos en el dataset.")
else:
    for col, nulos in columnas_con_nulos:
        print(f"La columna {col} tiene {nulos} valores nulos.")
    if columnas_sin_nulos:
        print(f"\nLas columnas {', '.join(columnas_sin_nulos)} no tienen valores nulos.")

"""## Revisión de variables categóricas

Determinamos el número de niveles en las diferentes columnas categóricas, y vemos que tienen más de un nivel (son informativas).
"""

cols_cat = ['neighbourhood', 'room_type', 'days_since_last_review_binned', 'host_is_superhost', ]

# Utilizamos el método nunique para conocer el número de elementos distintos encontrados a lo largo de un eje
for col in cols_cat:
    print(f'La columna {col}', 'tiene',data[col].nunique(), 'niveles')

"""Comprobamos que no hay errores tipograficos en las variables categoricas"""

for col in cols_cat:
    valores_unicos = sorted(data[col].dropna().unique())
    print(f"{col}: {valores_unicos}")

"""Para mayor facilidad de interpretación de los graficos recodificamos los valores de host_is_superhost"""

# Reemplzamos f por No y t por Si
data['host_is_superhost'] = data['host_is_superhost'].replace({'f': 'No', 't': 'Si'})

# Comprobamos los cambios
print(data['host_is_superhost'].value_counts())
print(data['host_is_superhost'].unique())

"""## Revisión de variables continuas

### Columnas con un unico valor (no informativas)
**Chequeamos** que las variables continuas tengan desvío estándar (std) diferente de cero (de esta manera nos aseguramos que tienen datos que varían, y por lo tanto son informativas)
"""

# Identificamos las variables numericas
numeric_cols = data.select_dtypes(include=np.number).columns

print("Columnas numéricas en el dataset:", list(numeric_cols))

# Comparamos las desviaciones estandar respecto a cero
cols_with_zero_std = []
for col in numeric_cols:
    if data[col].std() == 0:
        cols_with_zero_std.append(col)

# Mostrar resultados
if not cols_with_zero_std:
    print("\nNo hay columnas numéricas con un único valor (todas las desviaciones estandar son distintas a cero)")
else:
    print(f"\nLas columnas {', '.join(cols_with_zero_std)} tienen un unico valor (desviacion estandar igual a cero)")

"""### Outliers"""

data.describe()

"""Realizamos boxplots de las variables numericas para identificar posibles outleirs"""

cols_num = ['price', 'minimum_nights', 'reviews_per_month', 'availability_365', 'number_of_reviews_ltm', 'bedrooms', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value'] # Excluimos latitud y longitud

fig, ax = plt.subplots(nrows=len(cols_num), ncols=1, figsize=(8, len(cols_num)*6))
fig.subplots_adjust(hspace=0.5)
for i, col in enumerate(cols_num):
    sns.boxplot(x=col, data=data, ax=ax[i])
    ax[i].set_title(col)

"""Al analizar los valores de las variables minimum_nights y bedrooms, en lugar de tratarlas como variables numéricas con outliers, decidimos transformarlas en variables categóricas, ya que presentan distribuciones fuertemente asimétricas y valores extremos poco frecuentes pero válidos.
Para el resto de las variables numéricas, aplicamos un filtro eliminando aquellas filas que se encuentren por fuera del percentil 95 en al menos una de esas columnas. Sin embargo, excluimos de este proceso a las variables de review_scores, ya que aplicar el mismo criterio implicaría eliminar publicaciones con bajas calificaciones o sin reviews (a las cuales les asignamos un valor de -1 para evitar valores faltantes).
"""

# Lista de columnas numéricas
cols_num = ['price', 'reviews_per_month', 'number_of_reviews_ltm', 'availability_365']

# Calcular el percentil 95 para cada columna
percentiles_95 = data[cols_num].quantile(0.95)

# Mostrar cuántas filas se eliminarían por cada columna
print("Filas por encima del percentil 95 por variable:")
for col in cols_num:
    cantidad = (data[col] > percentiles_95[col]).sum()
    print(f"- {col}: {cantidad} filas")

# Filtrar las filas que estén por debajo o igual al percentil 95 en todas las columnas
condicion = (data[cols_num] <= percentiles_95).all(axis=1)
data_p95 = data[condicion].copy()

# Mostrar cuántas filas se eliminaron en total
eliminadas = len(data) - len(data_p95)
print(f"\nSe eliminaron {eliminadas} filas que estaban por encima del percentil 95 en al menos una variable.")

data_clean = data_p95

cols_num = ['price', 'reviews_per_month', 'number_of_reviews_ltm', 'availability_365' ]
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(8,30))
fig.subplots_adjust(hspace=0.5)
for i, col in enumerate(cols_num):
    sns.boxplot(x=col, data=data_clean, ax=ax[i])
    ax[i].set_title(col)

"""Si bien todavía se observan valores por debajo del bigote y/o por arriba del bigote en dos de las variables, decidimos no seguir eliminando valores extremos para no perder demasiadas instancias

# Transformación de variables

##minimum_nights

Convertimos la variable en binaria, llamando 'Short-Term Rentals' a las publicaciones con un minimo de dias menor o igual a 30 y 'Long-Term Rentals' cuando el minimo es mayor a 30 dias
"""

data_clean['minimum_nights'] = data_clean['minimum_nights'].apply(lambda x: 'short-term_rentals' if x <= 30 else 'long-term_rentals')

"""##bedrooms"""

print(data_clean['bedrooms'].value_counts().sort_index())

"""Convertimos la variable numerica bedrooms en una categorica por cantidad de ambientes."""

data_clean['bedrooms'] = data_clean['bedrooms'].apply(lambda x: '1 amb' if x == 0 else ('2 amb' if x == 1 else ('3 amb' if x == 2 else ('4 amb' if x == 3 else '5 o más amb'))))

print("\nConteo de valores para 'bedrooms':")
print(data_clean['bedrooms'].value_counts())

"""# Análisis exploratorio

## Análisis univariado
"""

data_clean.info()

"""### Variables categoricas

####neighbourhood
"""

col = "neighbourhood"
plt.figure(figsize=(12, 6))
orden = data_clean[col].value_counts().index  # Orden descendente por cantidad de anuncios
ax = sns.countplot(x=col, data=data_clean, order=orden)

# Agregamos etiquetas de porcentaje
total = len(data_clean[col])
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    x = p.get_x() + p.get_width() / 4 + 0.02
    y = p.get_height()
    ax.annotate(
        percentage,
        (x, y),
        ha='left',
        va='bottom',
        rotation=45,
        fontsize=8
    )

plt.xticks(rotation=75, ha='right')
plt.title('Distribución de Publicaciones por Barrio (%)')
plt.xlabel('Barrio')
plt.tight_layout()
plt.show()

col = "neighbourhood"
top_cats = data_clean[col].value_counts().nlargest(10).index
plt.figure(figsize=(10, 6))

ax = sns.countplot(x=col, data=data_clean, order=top_cats) # Use data_clean here

# Add percentage labels
total = len(data_clean[col]) # Calculate percentage based on data_clean
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')


plt.xticks(rotation=45)
plt.title('Top 10 de barrios con más avisos (%)')
plt.xlabel('Barrio')
plt.tight_layout()
plt.show()

total_listings = len(data_clean)
top_barrios = data_clean['neighbourhood'].value_counts()

# Porcentaje de los 5 barrios con más listings
top_5_barrios = top_barrios.nlargest(5)
print("Porcentaje de listings por cada uno de los 5 barrios con más listings:")
for barrio, count in top_5_barrios.items():
    percentage = (count / total_listings) * 100
    print(f"- {barrio}: {percentage:.2f}%")

# Porcentaje total de los top 3, 5 y 10 barrios
top_3_percentage = (top_barrios.nlargest(3).sum() / total_listings) * 100
top_5_percentage = (top_barrios.nlargest(5).sum() / total_listings) * 100
top_10_percentage = (top_barrios.nlargest(10).sum() / total_listings) * 100

print("\nPorcentaje total de listings para los top barrios:")
print(f"- Top 3 barrios: {top_3_percentage:.2f}%")
print(f"- Top 5 barrios: {top_5_percentage:.2f}%")
print(f"- Top 10 barrios: {top_10_percentage:.2f}%")

"""Se observa que el barrio con más avisos es Palermo y que los primeros tres barrios con más avisos concentran más del 50 % de los avisos y los barrios del top 10 concentran más del 80 % de las publicaciones"""

# Calcular el porcentaje de listings por barrio
neighbourhood_counts = data_clean['neighbourhood'].value_counts()
neighbourhood_percentages = (neighbourhood_counts / total_listings) * 100

# Identificar los barrios con 1% o menos de listings
barrios_con_bajo_porcentaje = neighbourhood_percentages[neighbourhood_percentages <= 1.0]

# Mostrar cuántos barrios tienen el 1% o menos
print(f"\nNúmero de barrios con 1% o menos de listings: {len(barrios_con_bajo_porcentaje)}")

# Mostrar los nombres de estos barrios y su porcentaje
print("\nBarrios con 1% o menos de listings:")
for barrio, percentage in barrios_con_bajo_porcentaje.items():
    print(f"- {barrio}: {percentage:.2f}%")

"""En el extremo opuesto podemos ver que hay más de 30 barrios con el 1 % o menos de las publicaciones

Podemos observar lo mismo empleando mapas
"""

# Cargar geometría de barrios de CABA
url_shapefile = "https://data.insideairbnb.com/argentina/ciudad-aut%C3%B3noma-de-buenos-aires/buenos-aires/2025-01-29/visualisations/neighbourhoods.geojson"
barrios = gpd.read_file(url_shapefile)

# Calcular métricas por barrio
listing_counts = data_clean["neighbourhood"].value_counts()
listing_pct = listing_counts / listing_counts.sum() * 100

# Unir métricas
summary_df = pd.DataFrame({
    "listings": listing_counts,
    "listing_pct": listing_pct
}).reset_index().rename(columns={"index": "neighbourhood"})

# Unir con datos geográficos
barrios_merge = barrios.merge(summary_df, on="neighbourhood", how="left")

# Crear paleta de colores azul clara (usando sólo parte de Blues)
blues = plt.cm.Blues
colors = blues(np.linspace(0.2, 0.9, 256))  # recorta extremos muy oscuros o claros
from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(colors)

# Crear figura
fig, ax = plt.subplots(figsize=(12, 10))
barrios_merge.plot(column="listing_pct", cmap=custom_cmap, legend=True,
                   edgecolor="black", linewidth=0.4, ax=ax)

# Agregar texts a cada barrio
for _, row in barrios_merge.iterrows():
    if pd.notna(row["listing_pct"]):  # Check for missing listing_pct values
        x, y = row["geometry"].centroid.x, row["geometry"].centroid.y
        barrio = row["neighbourhood"]
        count = int(row["listings"])
        pct = row["listing_pct"]

        ax.text(x, y + 0.001, f"{barrio}", fontsize=8, ha='center', weight='bold', color='black')
        ax.text(x, y - 0.001, f"{count} ({pct:.1f}%)", fontsize=7, ha='center', color='black')

# Título y referencia textual
plt.title("Porcentaje de anuncios por barrio", fontsize=14)

# Referencia textual (arriba derecha, texto centrado en el recuadro)
text = (
    "Referencia:\n"
    "Nombre del barrio\n"
    "Cantidad de anuncios (%)"
)
plt.gcf().text(
    0.755, 0.955, text,
    fontsize=9, va='top', ha='center',
    family='monospace',
    bbox=dict(facecolor='white', alpha=0.8)
)

plt.axis("off")
plt.tight_layout()
plt.show()

# Cargar geometría de barrios de CABA
url_shapefile = "https://data.insideairbnb.com/argentina/ciudad-aut%C3%B3noma-de-buenos-aires/buenos-aires/2025-01-29/visualisations/neighbourhoods.geojson"
barrios = gpd.read_file(url_shapefile)

# Calcular métricas por barrio
listing_counts = data_clean["neighbourhood"].value_counts()
listing_pct = listing_counts / listing_counts.sum() * 100

# Unir métricas
summary_df = pd.DataFrame({
    "listings": listing_counts,
    "listing_pct": listing_pct
}).reset_index().rename(columns={"index": "neighbourhood"})

# Unir con datos geográficos
barrios_merge = barrios.merge(summary_df, on="neighbourhood", how="left")

# --- Custom Color Mapping ---
# Assign a specific color for barrios with <= 1%
gray_color = '#D3D3D3'  # A light gray color

# Create a new column to represent the coloring value
# Assign 0 for barrios with <= 1% and their actual percentage for others
barrios_merge['coloring_pct'] = barrios_merge['listing_pct'].apply(lambda x: 0 if x <= 1.0 else x)

# Create a colormap for the rest of the percentages (excluding the 0)
# We'll use a standard blue scale, ensuring it starts effectively after the 0 value
blues = plt.cm.Blues
# Create a list of colors. The first color will be our specific gray.
# The rest will be from the Blues colormap, mapped to the range > 1%.
cmap_list = [gray_color] + [blues(i) for i in np.linspace(0.2, 0.9, 255)]
custom_cmap = ListedColormap(cmap_list)


# Create figure
fig, ax = plt.subplots(figsize=(12, 10))

# Plot the map using the 'coloring_pct' for color mapping
# We need to specify vmin and vmax for the legend to make sense with the 0 value
vmin = 0
vmax = barrios_merge['listing_pct'].max() # The max percentage for the rest

barrios_merge.plot(column="coloring_pct", cmap=custom_cmap, legend=True,
                   edgecolor="black", linewidth=0.4, ax=ax,
                   vmin=vmin, vmax=vmax)


# Agregar texts a cada barrio
for _, row in barrios_merge.iterrows():
    # Only add text for barrios that have data
    if pd.notna(row["listing_pct"]):
        x, y = row["geometry"].centroid.x, row["geometry"].centroid.y
        barrio = row["neighbourhood"]
        count = int(row["listings"])
        pct = row["listing_pct"]

        ax.text(x, y + 0.001, f"{barrio}", fontsize=8, ha='center', weight='bold', color='black')
        ax.text(x, y - 0.001, f"{count} ({pct:.1f}%)", fontsize=7, ha='center', color='black')

# Título y referencia textual
plt.title("Porcentaje de anuncios por barrio", fontsize=14)

# Referencia textual (arriba derecha, texto centrado en el recuadro)
text = (
    "Referencia:\n"
    "Nombre del barrio\n"
    "Cantidad de listings (%)\n"
    "Barrios con <= 1%: color gris" # Update note about the gray color
)
plt.gcf().text(
    0.74, 0.955, text,
    fontsize=9, va='top', ha='center',
    family='monospace',
    bbox=dict(facecolor='white', alpha=0.8)
)

plt.axis("off")
plt.tight_layout()
plt.show()

"""####room_type"""

col = "room_type"
plt.figure(figsize=(8, 6))
orden = data_clean[col].value_counts().index
ax = sns.countplot(x=col, data=data_clean, order=orden)
plt.title('Tipo de habitación')

# Agregamos etiquetas con porcentajes
total = len(data_clean[col])
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')

plt.tight_layout()
plt.show()

"""Se observa que el 91 % de las publicaciones corresponden a propiedades enteras. Luego hay un 8,5 % de publicaciones de habitaciones privadas mientras que el porcentaje de habitaciones compartidas y habitaciones de hotel suman menos del 1%.

####minimum_nights
"""

col = "minimum_nights"
plt.figure(figsize=(8, 6))
orden = data_clean[col].value_counts().index
ax = sns.countplot(x=col, data=data_clean, order=orden)
plt.title('Cantidad mínima de noches')

# Agregamos etiquetas con porcentajes
total = len(data_clean[col])
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')

plt.tight_layout()
plt.show()

"""Casi la totalidad de las publicaciones son alquileres de corto plazo, es decir que tienen estadias minimas menores o iguales a 30 dias. Alrededor del 1 % de las publicaciones tienen estadias minimas largas. Se trata de hosts que buscan alquilar por periodos relativamente largos.

####host_is_superhost
"""

col = "host_is_superhost"
plt.figure(figsize=(8, 6))
ax = sns.countplot(x=col, data=data_clean, order=data_clean[col].value_counts().index)
plt.title('Es Superhost')
plt.xlabel('Es Superhost')
plt.tight_layout()

# Agregamos etiquetas con porcentajes
total = len(data_clean[col])
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')


plt.show()

"""Poco más del 42 % de las publicaciones pertenecen a superhosts.

####bedrooms
"""

col = "bedrooms"
plt.figure(figsize=(10, 6))

orden_bedrooms = ['1 amb', '2 amb', '3 amb', '4 amb', '5 o más amb']
ax = sns.countplot(x=col, data=data_clean, order=orden_bedrooms)
plt.title('Cantidad de Ambientes')
plt.xlabel('Cantidad de Ambientes')
plt.tight_layout()

# Agregamos etiquetas con porcentajes
total = len(data_clean[col])
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')

plt.show()

"""Casi el 70 % de las publicaciones son de 2 ambientes, un 15 % son de 3 ambientes y un 13 % de 1 ambiente.
Las propiedades de mayor tamaño son muy pocas.

####days_since_last_review_binned
"""

col = "days_since_last_review_binned"

# Definimos el orden deseado para las categorías
labels_days_since_last_review = ['0-30', '31-90', '91-180', '181-365', '>365', 'sin reviews']

plt.figure(figsize=(10, 6))

# Usamos la lista labels_days_since_last_review directamente como el orden para el grafico
ax = sns.countplot(x=col, data=data_clean, order=labels_days_since_last_review)
plt.title('Días desde el último review (binned)')
plt.tight_layout()

# Agregamos etiquetas con porcentajes
total = len(data_clean[col])
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')

plt.show()

total_listings = len(data_clean)
days_since_counts = data_clean[col].value_counts()

"""Se observa que alrededor de un tercio de las publicaciones recibieron su ultimo review entre 0 y 30 dias antes de la captura de los datos y más del 50 % en los 3 meses previos. Poco más del 10 % de las propiedades tienen su ultimo review anterior a un año (propiedades poco alquiladas)

###Variables continuas

####price
"""

# Histograma del precio
plt.figure(figsize=(10, 6))
sns.histplot(data_clean['price'], bins=30, kde=True)
plt.title('Distribución del precio')
plt.xlabel('Precio')
plt.ylabel('Frecuencia')

# Configurar las marcas del eje x de 10 mil en 10 mil
max_price = data_clean['price'].max()
plt.xticks(np.arange(0, max_price + 10001, 10000))
plt.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

# Box-plot del precio
plt.figure(figsize=(10, 6))
sns.boxplot(x=data_clean['price'])
plt.title('Box-plot del precio')
plt.xlabel('Precio')

# Configurar las marcas del eje x de 10 mil en 10 mil
plt.xticks(np.arange(0, max_price + 10001, 10000))
plt.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

# Descripción estadística de la columna price
print("\nDescripción estadística de la columna 'price':")
print(data_clean['price'].describe())

# Mediana y Moda de la columna price
mediana_price = data_clean['price'].median()
moda_price = data_clean['price'].mode()

print(f'Moda del precio: {moda_price.iloc[0]}')
print(f"Mediana del precio: {mediana_price}")

"""Se observa una distribución asimetrica sesgada a la derecha, es decir que hay más publicaciones con precios bajos o medios que con precios altos. Los valores están entre 260 y 126772. Sin embargo cabe destacar que el rango intercuartil, que concentras el 50 % de los anuncios, va de 29406 a 54464 pesos.

####reviews_per_month
"""

# Histograma de reviews_per_month
plt.figure(figsize=(10, 6))
sns.histplot(data_clean['reviews_per_month'], bins=30, kde=True)
plt.title('Distribución de Reviews por mes')
plt.xlabel('Reviews por mes')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

# Box-plot de reviews_per_month
plt.figure(figsize=(10, 6))
sns.boxplot(x=data_clean['reviews_per_month'])
plt.title('Box-plot de Reviews por mes')
plt.xlabel('Reviews por mes')
plt.tight_layout()
plt.show()

# Descripción estadística de la columna reviews_per_month
print("\nDescripción estadística de la columna 'reviews_per_month':")
print(data_clean['reviews_per_month'].describe())

# Mediana y moda de la columna reviews_per_month
mediana_reviews_month = data_clean['reviews_per_month'].median()
moda_reviews_month = data_clean['reviews_per_month'].mode()

print(f'Moda de reviews por mes: {moda_reviews_month.iloc[0]}')
print(f"Mediana de reviews por mes: {mediana_reviews_month}")

"""Observamos una distribución muy sesgada ala derecha. La mayoria de las publicaciones no tienne reviews o reciben muy pocas por mes.El 75% de los casos tiene hasta 1,48 *reviews por mes*. El promedio está en 0,93, la mediana 0,65 y la moda es 0 con casi 6000 avisos sin reviews sobre más de 26000.

####availability_365
"""

# Histograma de availability_365
plt.figure(figsize=(10, 6))
sns.histplot(data_clean['availability_365'], bins=30, kde=True)
plt.title('Distribución de disponibilidad en 365 dias')
plt.xlabel('Días disponibles en 365 dias')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

# Box-plot de availability_365
plt.figure(figsize=(10, 6))
sns.boxplot(x=data_clean['availability_365'])
plt.title('Box-plot de disponibilidad en 365 dias')
plt.xlabel('Días disponibles en 365')
plt.tight_layout()
plt.show()

# Descripción estadística de la columna availability_365
print("\nDescripción estadística de la columna 'availability_365':")
print(data_clean['availability_365'].describe())

# Mediana y moda de la columna availability_365
mediana_availability = data_clean['availability_365'].median()
moda_availability = data_clean['availability_365'].mode()

print(f'Moda de disponibilidad_365: {moda_availability.iloc[0]}')
print(f"Mediana de disponibilidad_365: {mediana_availability}")

"""El promedio de la variable *días disponibles durante el año* se encontró en 224 días, el 50% de los valores está por debajo 249 días y como máximo es un año, lo que indica que anuncios cuentan con alta disponibilidad.

####number_of_reviews_ltm
"""

# Histograma de number_of_reviews_ltm
plt.figure(figsize=(10, 6))
sns.histplot(data_clean['number_of_reviews_ltm'], bins=30, kde=True)
plt.title('Distribución del número de reviews en el último año')
plt.xlabel('Número de reviews en el último año')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

# Box-plot de number_of_reviews_ltm
plt.figure(figsize=(10, 6))
sns.boxplot(x=data_clean['number_of_reviews_ltm'])
plt.title('Box-plot del número de reviews en el último año')
plt.xlabel('Número de reviews en el último año')
plt.tight_layout()
plt.show()

# Descripción estadística de la columna number_of_reviews_ltm
print("\nDescripción estadística de la columna 'number_of_reviews_ltm':")
print(data_clean['number_of_reviews_ltm'].describe())

# Mediana y moda de la columna number_of_reviews_ltm
mediana_reviews_ltm = data_clean['number_of_reviews_ltm'].median()
moda_reviews_ltm = data_clean['number_of_reviews_ltm'].mode()

print(f'Moda del número de reviews en el último año: {moda_reviews_ltm.iloc[0]}')
print(f"Mediana del número de reviews en el último año: {mediana_reviews_ltm}")

"""La distribución de la variable number_of_reviews_ltm es asimétrica positiva con sesgo a la derecha, es decir que hay muchos alojamientos que recibieron pocas o ninguna review, y unos pocos que concentraron muchas. La moda es 0, indicando que muchas de las publicaciones no recibieron ninguna review en el último año.
El 50 % de las publicaciones recibieron 4 o menos reviews en el último año y el 75 % tuvo menos de 12.

####scores
"""

# Lista de columnas de scores para graficar
score_columns_to_plot = [
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value"
]

# Crear histogramas para cada columna de score
for col in score_columns_to_plot:
    plt.figure(figsize=(8, 5))
    sns.histplot(data_clean[data_clean[col] != -1][col], bins=20, kde=True) # Excluir los -1 para el histograma
    plt.title(f'Distribución de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.show()

# Crear boxplots para cada columna de score
for col in score_columns_to_plot:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=data_clean[data_clean[col] != -1][col]) # Excluir los -1 para el boxplot
    plt.title(f'Box-plot de {col}')
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

# Descripción estadística de las columnas de scores (excluyendo los -1)
print("\nDescripción estadística de las columnas de scores (excluyendo -1):")
print(data_clean[data_clean['review_scores_rating'] != -1][score_columns_to_plot].describe())

# Mediana y Moda de las columnas de scores (excluyendo los -1)
print("\nMediana y Moda de las columnas de scores (excluyendo -1):")
for col in score_columns_to_plot:
    # Filtrar valores -1 antes de calcular mediana y moda
    filtered_data = data_clean[data_clean[col] != -1][col]
    if not filtered_data.empty:
        mediana_score = filtered_data.median()
        moda_score = filtered_data.mode()
        print(f"{col}:")
        if not moda_score.empty:
             print(f"  Moda: {moda_score.iloc[0]}")
        else:
             print(f"  Moda: No hay moda única")
        print(f"  Mediana: {mediana_score}")
    else:
        print(f"{col}: No hay datos (después de excluir -1)")

# Análisis de la proporción de listings con scores faltantes (-1)
print("\nProporción de listings sin scores (valores -1):")
total_listings = len(data_clean)
for col in score_columns_to_plot:
    count_missing_score = (data_clean[col] == -1).sum()
    percentage_missing_score = (count_missing_score / total_listings) * 100
    print(f"- {col}: {count_missing_score} ({percentage_missing_score:.2f}%)")

"""En general se observa que las calificaciones son todas bastante cercanas a 5 (calificación maxima). Los scores de limpieza y relacion precio-calidad son ligeramente más bajos que los demas.

## Análisis bivariado

Cruzamos las distintas variables contra el número de reviews por mes

###reviews_per_month
A continuación cruzamos las distintas variables con el número de reviews por mes.
El número de reviews por mes es un indicador de cuan visitado es un alojamiento.

####neighbourhood
"""

# Cakcykanis ka nedua de número de reviews por mes para cada barrio
neighbourhood_reviews_mean = data_clean.groupby('neighbourhood')['reviews_per_month'].mean().sort_values(ascending=False)

# Grafico de barras
plt.figure(figsize=(15, 8))
sns.barplot(x=neighbourhood_reviews_mean.index, y=neighbourhood_reviews_mean.values)
plt.xticks(rotation=90)
plt.title('Promedio del número de reviews por mes por barrio')
plt.xlabel('Neighbourhood')
plt.ylabel('Mean Reviews per Month')
plt.tight_layout()
plt.show()

"""Podemos observar que los barrios de Puerto Madero, San Nicolas, Monte Castro y Palermo son los unicos que tienen en promedio más de un review por mes."""

# Cargar geometría de barrios de CABA
url_shapefile = "https://data.insideairbnb.com/argentina/ciudad-aut%C3%B3noma-de-buenos-aires/buenos-aires/2025-01-29/visualisations/neighbourhoods.geojson"
barrios = gpd.read_file(url_shapefile)

# Calcular métricas por barrio
reviews_mean = data_clean.groupby("neighbourhood")["reviews_per_month"].mean()
listing_counts = data_clean["neighbourhood"].value_counts()
listing_pct = listing_counts / listing_counts.sum() * 100

# Unir métricas
summary_df = pd.DataFrame({
    "reviews_per_month": reviews_mean,
    "listings": listing_counts,
    "listing_pct": listing_pct
}).reset_index().rename(columns={"index": "neighbourhood"})

# Unir con datos geográficos
barrios_merge = barrios.merge(summary_df, on="neighbourhood", how="left")

# Crear paleta de colores azul clara (usando sólo parte de Blues)
blues = plt.cm.Blues
colors = blues(np.linspace(0.2, 0.9, 256))  # recorta extremos muy oscuros o claros
from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(colors)

# Crear figura
fig, ax = plt.subplots(figsize=(12, 10))
barrios_merge.plot(column="reviews_per_month", cmap=custom_cmap, legend=True,
                   edgecolor="black", linewidth=0.4, ax=ax)

# Agregar textos a cada barrio
for _, row in barrios_merge.iterrows():
    if pd.notna(row["reviews_per_month"]):
        x, y = row["geometry"].centroid.x, row["geometry"].centroid.y
        barrio = row["neighbourhood"]
        reviews = row["reviews_per_month"]
        count = int(row["listings"])
        pct = row["listing_pct"]

        ax.text(x, y + 0.002, f"{barrio}", fontsize=8, ha='center', weight='bold', color='black')
        ax.text(x, y, f"{reviews:.2f}", fontsize=8, ha='center', color='black')
        ax.text(x, y - 0.002, f"{count} ({pct:.1f}%)", fontsize=7, ha='center', color='black')

# Título y referencia textual
plt.title("Promedio de reviews por mes y anuncios por barrio", fontsize=14)

# Referencia textual (arriba derecha, texto centrado en el recuadro)
text = (
    "Referencia:\n"
    "Nombre del barrio\n"
    "Reviews por mes (prom)\n"
    "Cantidad de listings (%)"
)
plt.gcf().text(
    0.755, 0.955, text,
    fontsize=9, va='top', ha='center',
    family='monospace',
    bbox=dict(facecolor='white', alpha=0.8)
)

plt.axis("off")
plt.tight_layout()
plt.show()

# Calcular métricas por barrio (usando la mediana de reviews_per_month)
reviews_median = data_clean.groupby("neighbourhood")["reviews_per_month"].median()
listing_counts = data_clean["neighbourhood"].value_counts()
listing_pct = listing_counts / listing_counts.sum() * 100

# Unir métricas
summary_df_median = pd.DataFrame({
    "reviews_per_month": reviews_median,
    "listings": listing_counts,
    "listing_pct": listing_pct
}).reset_index().rename(columns={"index": "neighbourhood"})

# Unir con datos geográficos
barrios_merge_median = barrios.merge(summary_df_median, on="neighbourhood", how="left")

# Crear paleta de colores azul clara (usando sólo parte de Blues)
blues = plt.cm.Blues
colors = blues(np.linspace(0.2, 0.9, 256))  # recorta extremos muy oscuros o claros
custom_cmap = ListedColormap(colors)

# Crear figura
fig, ax = plt.subplots(figsize=(12, 10))
barrios_merge_median.plot(column="reviews_per_month", cmap=custom_cmap, legend=True,
                          edgecolor="black", linewidth=0.4, ax=ax)

# Agregar textos a cada barrio
for _, row in barrios_merge_median.iterrows():
    if pd.notna(row["reviews_per_month"]):
        x, y = row["geometry"].centroid.x, row["geometry"].centroid.y
        barrio = row["neighbourhood"]
        reviews = row["reviews_per_month"]
        count = int(row["listings"])
        pct = row["listing_pct"]

        ax.text(x, y + 0.002, f"{barrio}", fontsize=8, ha='center', weight='bold', color='black')
        ax.text(x, y, f"{reviews:.2f}", fontsize=8, ha='center', color='black')
        ax.text(x, y - 0.002, f"{count} ({pct:.1f}%)", fontsize=7, ha='center', color='black')

# Título y referencia textual
plt.title("Mediana de reviews por mes y anuncios por barrio", fontsize=14)

# Referencia textual (arriba derecha, texto centrado en el recuadro)
text = (
    "Referencia:\n"
    "Nombre del barrio\n"
    "Reviews por mes (mediana)\n"
    "Cantidad de listings (%)"
)
plt.gcf().text(
    0.755, 0.955, text,
    fontsize=9, va='top', ha='center',
    family='monospace',
    bbox=dict(facecolor='white', alpha=0.8)
)

plt.axis("off")
plt.tight_layout()
plt.show()

"""Los barrios cuyos avisos tienen más reviews se concentran en la zona del bajo y en el norte de la ciudad. En la zona sur el número de reviews es muy bajo pero a la vez son barrios con muy pocas publicaciones. El caso del barrio de Monte Castro resulta llamativo por su elevado numero de reviews por mes pero debe ser interpretado con precaución ya que se trata de un barrio con muy pocas publicaciones."""

# Seleccionamos los 10 barrios con mas anuncios
top_10_neighbourhoods = data_clean['neighbourhood'].value_counts().nlargest(10).index.tolist()

# Filtramos esos barrios
data_top10 = data_clean[data_clean['neighbourhood'].isin(top_10_neighbourhoods)]

# Ordnamos los barrios segun reviews por mes
order_top10 = data_top10.groupby('neighbourhood')['reviews_per_month'].mean().sort_values(ascending=False).index

# Crear box plot
plt.figure(figsize=(14, 7))
sns.boxplot(x='neighbourhood', y='reviews_per_month', data=data_top10, order=order_top10)
plt.xticks(rotation=45, ha='right')
plt.title('Box Plot de reviews por mes vs top 10 de barrios')
plt.xlabel('barrio')
plt.ylabel('reseñas por mes')
plt.tight_layout()
plt.show()

# crear Strip Plot para los 10 barrios
plt.figure(figsize=(14, 7))
sns.stripplot(x='neighbourhood', y='reviews_per_month', data=data_top10, order=order_top10, alpha=0.5, jitter=0.3)
plt.xticks(rotation=45, ha='right')
plt.title('Strip Plot de reviews por mes vs top 10 de barrios')
plt.xlabel('barrio')
plt.ylabel('reseñas por mes')
plt.tight_layout()
plt.show()

"""Entre los 10 barrios con más publicaciones podemos observar que en todos los casos las distribuciones están sesgadas hacia la izquierda (más publicaciones con pocos reviews). Empleando el coeficiente de variación para independizarnos de las medias, podemos observas diferencias sutiles en la dispersión, siendo Palermo y San Nicolas los de menor dispersión en el número de reviews mientras que Balvanera es el de mayor dispersión."""

# Calculamos desvio estandar y coeficiente de variación para reseñas por mes
neighbourhood_stats_top10 = data_top10.groupby('neighbourhood')['reviews_per_month'].agg(['std', 'mean']).reset_index()
neighbourhood_stats_top10['cov'] = neighbourhood_stats_top10['std'] / neighbourhood_stats_top10['mean']

print("Estadísticas de reviews_per_month para los 10 barrios con más anuncios:")
print(neighbourhood_stats_top10.sort_values(by='mean', ascending=False))

"""####room_type"""

# Calculamos media de reseñas por mes por barrio
neighbourhood_reviews_mean = data_clean.groupby('room_type')['reviews_per_month'].mean().sort_values(ascending=False)

# Grafico de barras de las medias
plt.figure(figsize=(15, 8))
ax = sns.barplot(x=neighbourhood_reviews_mean.index, y=neighbourhood_reviews_mean.values) # Assign the plot to a variable


plt.title('Promedio del número de reseñas por mes por tipo de habitación')
plt.xlabel('Room Type')
plt.ylabel('media de reseñas por mes')

# Agregamos etiquetas con las medias
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.show()

"""Podemos observar que los alojamientos completos son los que cuentan con el mayor promedio de reseñas al mes.

####minimun_nights
"""

# Ordenar por la mediana de reviews_per_month según minimum_nights
order_min_nights = data_clean.groupby('minimum_nights')['reviews_per_month'].median().sort_values(ascending=False).index

# Calcular medias por minimum_nights y ordenar por mediana
mean_reviews = data_clean.groupby('minimum_nights')['reviews_per_month'].mean()
order_min_nights = data_clean.groupby('minimum_nights')['reviews_per_month'].median().sort_values(ascending=False).index

# Gráfico de barras con seaborn
plt.figure(figsize=(15, 8))
ax = sns.barplot(
    x=mean_reviews.loc[order_min_nights].index,
    y=mean_reviews.loc[order_min_nights].values,
)
plt.title('Promedio de reseñas por mes vs tipo de estadia (short-term vs long-term)')
plt.xlabel('cantidad mínima de noches')
plt.ylabel('Media de Reseñas por Mes')
plt.xticks(rotation=0)
plt.tight_layout()

# Agregamos etiquetas con las medias
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.show()

# Box Plot
plt.figure(figsize=(15, 8))
sns.boxplot(x='minimum_nights', y='reviews_per_month', data=data_clean, order=order_min_nights)
plt.xticks(rotation=0)  # Etiquetas horizontales
plt.title('Box Plot de reseñas por mes vs tipo de estadia (short-term vs long-term)')
plt.xlabel('cantidad mínima de noches')
plt.ylabel('Reseñas por Mes')
plt.tight_layout()
plt.show()

# Strip Plot
plt.figure(figsize=(15, 8))
sns.stripplot(x='minimum_nights', y='reviews_per_month', data=data_clean,
              order=order_min_nights, alpha=0.5, jitter=0.2)
plt.xticks(rotation=0)
plt.title('Strip Plot de reviews por mes vs tipo de estadia (short-term vs long-term)')
plt.xlabel('cantidad minima de noches')
plt.ylabel('Reseñas por Mes')
plt.tight_layout()
plt.show()

"""Las propiedades disponibles para estadias cortas tienen muchas más reviews que aquellas unicamente disponibles para estadias largas (gran parte de las cuales tienen cero reviews). La diferencia en reviews por mes entre short-term y long-term rentals era esperable ya que por ejemplo una propiedad con minimo de 31 dias jamas podria tener más de un review por mes, por más exitosa que sea.

####host_is_superhost
"""

# Calcular la media de reviews_per_month para cada categoría de host_is_superhost
mean_reviews_superhost = data_clean.groupby('host_is_superhost')['reviews_per_month'].mean().sort_values(ascending=False)

# Gráfico de barras del promedio, ordenado por promedio (descendente)
plt.figure(figsize=(8, 6))

ax = sns.barplot(x=mean_reviews_superhost.index, y=mean_reviews_superhost.values)
plt.title('Promedio de reviews por mes vs host_is_superhost')
plt.xlabel('Es Superhost')
plt.ylabel('Promedio de Reviews por Mes')
plt.tight_layout()

# Agregamos etiquetas con las medias sobre las barras
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.show()

# Box Plot
plt.figure(figsize=(8, 6))
# Usamos el mismo orden que en el barplot
sns.boxplot(x='host_is_superhost', y='reviews_per_month', data=data_clean, order=mean_reviews_superhost.index)
plt.title('Box Plot de reviews por mes vs host_is_superhost')
plt.xlabel('Es Superhost')
plt.ylabel('Reviews por Mes')
plt.tight_layout()
plt.show()

# Strip Plot
plt.figure(figsize=(8, 6))
# Usamos el mismo orden que en el barplot
sns.stripplot(x='host_is_superhost', y='reviews_per_month', data=data_clean,
              order=mean_reviews_superhost.index, alpha=0.5, jitter=0.2) # Ajusta jitter si es necesario
plt.title('Strip Plot de reviews por mes vs host_is_superhost')
plt.xlabel('Es Superhost')
plt.ylabel('Reviews por Mes')
plt.tight_layout()
plt.show()

"""Podemos observar que las publicaciones pertenecientes a superhosts tienen alrededor del doble de reviews por mes que aquellas cuyos hosts no son superhosts.

####bedrooms
"""

# Ordenamos las categorías de 'bedrooms' según el número de ambientes
order_bedrooms = ['1 amb', '2 amb', '3 amb', '4 amb', '5 o más amb']

# Calculamos la media de reviews_per_month para cada categoría de bedrooms
mean_reviews_bedrooms = data_clean.groupby('bedrooms')['reviews_per_month'].mean().reindex(order_bedrooms) # Reindex para mantener el orden

# Gráfico de barras del promedio
plt.figure(figsize=(10, 6))
sns.barplot(x=mean_reviews_bedrooms.index, y=mean_reviews_bedrooms.values)
plt.title('Promedio de reviews por mes vs Cantidad de Ambientes (Bedrooms)')
plt.xlabel('Cantidad de Ambientes')
plt.ylabel('Promedio de Reviews por Mes')
plt.tight_layout()
plt.show()

# Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='bedrooms', y='reviews_per_month', data=data_clean, order=order_bedrooms)
plt.title('Box Plot de reviews por mes vs Cantidad de Ambientes (Bedrooms)')
plt.xlabel('Cantidad de Ambientes')
plt.ylabel('Reviews por Mes')
plt.tight_layout()
plt.show()

# Strip Plot
plt.figure(figsize=(10, 6))
sns.stripplot(x='bedrooms', y='reviews_per_month', data=data_clean,
              order=order_bedrooms, alpha=0.5, jitter=0.2)
plt.title('Strip Plot de reviews por mes vs Cantidad de Ambientes (Bedrooms)')
plt.xlabel('Cantidad de Ambientes')
plt.ylabel('Reviews por Mes')
plt.tight_layout()
plt.show()

"""Las propiedades de 5 o más ambientes son muy pocas, y tiene su distribucion de reseñas por mes marcadamenete desplazadas a muy pocas reseñas por mes, esta tendencia se observa también en los anuncios de 4 ambientes. Entre las de 1 a 3 ambientes no se observan grandes diferencias, aunque las de 1 ambiente tienen en promedio más reviews que las de 2 y 3.

####days_since_last_review_binned
"""

# Calculamos la media reviews por mes para cada valor de days_since_last_review_binned
reviews_mean_binned = data_clean.groupby('days_since_last_review_binned')['reviews_per_month'].mean().sort_values(ascending=False)

# Bar Plot
plt.figure(figsize=(15, 6))
ax = sns.barplot(x=reviews_mean_binned.index, y=reviews_mean_binned.values)
plt.title('Promedio de reviews por mes vs Tiempo desde la última review')
plt.xlabel('Tiempo desde la última review')
plt.ylabel('Media de Reviews por Mes')
plt.tight_layout()

# Agregamos etiquetas sobre las barras
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
plt.show()

# Ordenamos los bins por media de reviews por mes
order_binned = data_clean.groupby('days_since_last_review_binned')['reviews_per_month'].mean().sort_values(ascending=False).index

# Box Plot
plt.figure(figsize=(15, 8))
sns.boxplot(x='days_since_last_review_binned', y='reviews_per_month', data=data_clean, order=order_binned)
plt.title('Box Plot de reviews por mes vs Tiempo desde la última review')
plt.xlabel('Tiempo desde la última review')
plt.ylabel('Reviews per Month')
plt.tight_layout()
plt.show()

# Strip Plot
plt.figure(figsize=(15, 8))
sns.stripplot(x='days_since_last_review_binned', y='reviews_per_month', data=data_clean,
              order=order_binned, alpha=0.5, jitter=0.2)
plt.xticks(rotation=45, ha='right')
plt.title('Strip Plot de reviews por mes vs Tiempo desde la última review')
plt.xlabel('Tiempo desde la última review')
plt.ylabel('Reviews per Month')
plt.tight_layout()
plt.show()

"""Se observa que las publicaciones con más reviews en general recibieron su ultimo review recientemente. Son muy pocas las publicaciones con alta número de reviews que no reciben un review hace más de un año.

####price
"""

plt.figure(figsize=(10, 6))
sns.scatterplot(x='price', y='reviews_per_month', data=data_clean, alpha=0.5)
plt.title('Scatter Plot de Reviews por mes vs Precio')
plt.xlabel('Precio')
plt.ylabel('Reviews por mes')
plt.show()

"""En este scatter plot se puede ver que la mayoría de los alojamientos tienen precio bajo/medio y abarcan todo el rango de reseñas por mes, mientras que hay menos densidad de alojamientos que tengan precio alto y muchas reviews por mes"""

# Crear rangos de precios
data_clean['price_bin'] = pd.qcut(data_clean['price'], q=5)  # o usar pd.cut si querés rangos fijos

# Calcular media de reviews por mes en cada rango de precio
mean_reviews = data_clean.groupby('price_bin')['reviews_per_month'].mean().reset_index()

# Gráfico de barras
plt.figure(figsize=(10, 6))
sns.barplot(data=mean_reviews, x='price_bin', y='reviews_per_month')

plt.xticks(rotation=45)
plt.ylabel('Promedio de reviews por mes')
plt.xlabel('Rango de precios')
plt.title('Promedio de reviews por mes según rango de precio')
plt.tight_layout()
plt.show()

"""En este gráfico podemos ver que los precios con mayor cantidad de reseñas por mes son entre medios y bajos

####availability_365
"""

# Crear rangos de availability_365
# Usamos pd.cut para crear bins con rangos fijos o aproximados
# Definir los límites de los bins (0 días, 1-90 días, 91-180 días, 181-270 días, 271-365 días)
bins_availability = [0, 90, 180, 270, 365]
labels_availability = ['0-90', '91-180', '181-270', '271-365']

# Asegurarse de que el primer límite sea inclusivo si hay valores de 0
# Ajustamos los bins y labels para que incluyan el 0 y el final
bins_availability = [-1, 90, 180, 270, 365]
labels_availability = ['0-90', '91-180', '181-270', '271-365'] # O considerar '0' como una categoría separada si hay muchos

# Si hay muchas filas con 0 disponibilidad, podríamos hacer un bin de 0-0, 1-90, etc.
# Vamos a verificar la distribución de 0s
# print(data_clean['availability_365'].value_counts().sort_index().head())

# Dado el histograma previo, parece que hay valores en 0 y en el rango alto.
# Usemos qcut para dividir por cuartiles si no queremos rangos fijos.
data_clean['availability_bin'] = pd.qcut(data_clean['availability_365'], q=5, duplicates='drop') # 'drop' maneja casos con pocos valores únicos

# Calcular media de reviews por mes en cada rango de disponibilidad
mean_reviews_availability = data_clean.groupby('availability_bin')['reviews_per_month'].mean().reset_index()

# Gráfico de barras
plt.figure(figsize=(10, 6))
sns.barplot(data=mean_reviews_availability, x='availability_bin', y='reviews_per_month')

plt.xticks(rotation=45)
plt.ylabel('Promedio de reviews por mes')
plt.xlabel('Rango de disponibilidad en 365 días')
plt.title('Promedio de reviews por mes según rango de disponibilidad')
plt.tight_layout()
plt.show()

"""En este gráfico, si bien no hay grandes diferencias en el promedio de reseñas por mes para los distintos bines de la variable rango de disponibilidad, se ve que los alojamientos con más reseñas muestran una disponibilidad de las más altas (entre 270 y 348 días al año), lo que parece indicar que no hay una relación entre la cantidad de reseñas y la disponibilidad de ese alojamiento en el futuro.

####number_of_reviews_ltm
"""

# Gráfico de dispersión (Scatter Plot) de number_of_reviews_ltm vs reviews_per_month
plt.figure(figsize=(10, 6))
sns.scatterplot(x='number_of_reviews_ltm', y='reviews_per_month', data=data_clean, alpha=0.5)
plt.title('Gráfico de Dispersión: Reviews por Mes vs Número de Reviews en el Último Año')
plt.xlabel('Número de Reviews en el Último Año (number_of_reviews_ltm)')
plt.ylabel('Reviews por Mes (reviews_per_month)')
plt.show()

# Crear rangos (bins) para number_of_reviews_ltm
# Usemos qcut para dividir en 5 grupos aproximadamente iguales.
data_clean['reviews_ltm_bin'] = pd.qcut(data_clean['number_of_reviews_ltm'], q=5, duplicates='drop')

# Calcular la media de reviews_per_month en cada rango de number_of_reviews_ltm
mean_reviews_ltm = data_clean.groupby('reviews_ltm_bin')['reviews_per_month'].mean().reset_index()

# Gráfico de barras del promedio de reviews_per_month por rango de number_of_reviews_ltm
plt.figure(figsize=(10, 6))
sns.barplot(data=mean_reviews_ltm, x='reviews_ltm_bin', y='reviews_per_month')

plt.xticks(rotation=45, ha='right') # Rotar etiquetas para mejor legibilidad
plt.ylabel('Promedio de reviews por mes')
plt.xlabel('Rango de Número de Reviews en el Último Año')
plt.title('Promedio de reviews por mes según rango de reviews en el último año')
plt.tight_layout()
plt.show()

"""Se observa que los alojamientos con más reviews por mes son a su vez los que tienen más reviews en el último año.

####score_reviews
"""

score_columns = [
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value"
]

# Iterar sobre cada columna de score y generar el gráfico de dispersión
for score_col in score_columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='reviews_per_month', y=score_col, data=data_clean, alpha=0.6)
    plt.title(f'Gráfico de Dispersión: {score_col} vs Reviews por Mes')
    plt.xlabel('Reviews por Mes')
    plt.ylabel(score_col)
    plt.tight_layout()
    plt.show()

# Definimos las columnas de interes (scores)
score_columns = [
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value"
]

# --- Análisis ignorando valores -1 ---

# Creamos un dataframe donde los valores -1 en las columnas de score se consideran NaN
data_no_neg1 = data_clean.copy()
for col in score_columns:
    data_no_neg1[col] = data_no_neg1[col].replace(-1, np.nan)

# Seleccionamos solo las columnas relevantes
correlation_data_no_neg1 = data_no_neg1[['reviews_per_month'] + score_columns].dropna()

# Calculamos la matriz de correlación
correlation_matrix_no_neg1 = correlation_data_no_neg1.corr()

# Extraemos correlaciones con reviews_per_month y ordenarlas
correlation_subset_no_neg1 = correlation_matrix_no_neg1.loc['reviews_per_month', score_columns]
correlation_subset_no_neg1_sorted = correlation_subset_no_neg1.sort_values(ascending=False) # decreciente


# Heatmap con orden por correlación decreciente
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_subset_no_neg1_sorted.to_frame(), annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
plt.title('Heatmap de Correlación: Reviews por mes vs Scores (orden decreciente)')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Gráfico de barras
plt.figure(figsize=(10, 6))
sns.barplot(x=correlation_subset_no_neg1_sorted.index, y=correlation_subset_no_neg1_sorted.values)
plt.title('Coeficientes de Correlación: Reviews por mes vs Scores (ignorando -1)')
plt.xlabel('Variables de Score')
plt.ylabel('Coeficiente de Correlación')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\nCoeficientes de Correlación (ignorando -1, orden decreciente):\n", correlation_subset_no_neg1_sorted)

"""Al analizar la correlación entre los distintos scores y el número de reviews mes, podemos observar que los scores sobre relación precio/calidad y en limpiza correlacionan mejor con el número de reviews mes que los scores relativos a la facilidad en el checkin, la ubicación y la comunicación con el host"""

# Columnas de scores a graficar
score_columns = [
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value"
]

# Creamos un dataframe filtrando las filas donde al menos una columna de score es -1
# Creamos una máscara booleana para identificar filas con -1 en *alguna* columna de score
mask_no_neg1_scores = (data_clean[score_columns] != -1).all(axis=1)

# Filtramos el dataframe
data_filtered_scores = data_clean[mask_no_neg1_scores].copy()

# Creamos 10 bines para reviews_per_month en el dataframe filtrado
# Usamos pd.qcut para crear bines con un número similar de observaciones
data_filtered_scores['reviews_per_month_bin'] = pd.qcut(
    data_filtered_scores['reviews_per_month'],
    q=10,
    labels=False,
    duplicates='drop'
)

# Calculamos el rango de cada bin antes de agrupar (minimo y maximo de reviews por mes por bin)
bin_ranges = data_filtered_scores.groupby('reviews_per_month_bin')['reviews_per_month'].agg(['min', 'max'])

# Creamos una lista de con los rangos
range_labels = [f'{row["min"]:.2f}-{row["max"]:.2f}' for index, row in bin_ranges.iterrows()]

# Calculamos la media de cada score para cada bin de reviews_per_month
mean_scores_by_reviews_bin = data_filtered_scores.groupby('reviews_per_month_bin')[score_columns].mean().reset_index()

# Preparamos los datos para el gráfico
df_plot = mean_scores_by_reviews_bin.melt(
    id_vars='reviews_per_month_bin',
    value_vars=score_columns,
    var_name='Score Type',
    value_name='Mean Score'
)

# Creamos el line plot
plt.figure(figsize=(14, 8))

# Usar seaborn.lineplot
sns.lineplot(
    data=df_plot,
    x='reviews_per_month_bin', # Use the bin number here
    y='Mean Score',
    hue='Score Type',
    style='Score Type',
    markers=True,
    dashes=False
)

plt.title('Promedio de Scores por Rango de Reviews por Mes (Ignorando -1)', fontsize=16)
plt.xlabel('Rango de Reviews por Mes', fontsize=12)
plt.ylabel('Promedio del Score', fontsize=12)

# Usamos la lista de rangos para el eje x
plt.xticks(ticks=bin_ranges.index, labels=range_labels, rotation=45, ha='right')

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Tipo de Score', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

"""##Conclusión

Caracteristicias de las propiedades con más reviews:
- Se encuentran en zona norte y el bajo (especialmente Palermo, San Nicolas, Palermo)
- Son propiedades completas de hasta 4 ambientes
- Disponibles para estadias de corto plazo
- Pertenecen a superhosts
- Tienen precio bajo/medio
- Son las de mayores scores de limpieza y relacion precio/calidad

##Bonus track: Ingresos y rentabilidad

Esta sección la agregamos al fianl como una aproximación a estimar rentabilidad (con muchos supuestos) pero decidimos no mostrarla en la exposicion.
"""

### Creamos una variable que estima la ocupación a la que llamaremos estimated_occupancy_ltm
# Esta variable busca estimar el porcentaje de días que un anuncio estuvo ocupado en los últimos 12 meses.

# Parámetros utilizados en la estimación:
estadia_promedio = 5.7 # Estimación de la duración promedio de la estadía en noches (fuente: https://airbtics.com/tourism-statistics-buenos-aires-ar)
review_rate = 0.5 # Estimación de la proporción de huéspedes que dejan una reseña (fuente: https://insideairbnb.com/data-assumptions/)

# La fórmula para estimar la ocupación (estimated_occupancy_ltm) se basa en el número de reseñas en los últimos 12 meses (number_of_reviews_ltm)
# y el número mínimo de noches requeridas (minimum_nights).

# La lógica es diferente para avisos con y sin reseñas en el último año.
# Para listings con reseñas, se estima el número de estancias a partir de las reseñas (ajustado por el review_rate)
# y se multiplica por la duración de la estadía (usando la estadía promedio o el minimum_nights, lo que sea mayor).
# Para listings sin reseñas, se asigna una ocupación estimada a una proporción de listings (igual al review_rate)
# para simular que algunos podrían haber tenido ocupación sin dejar reseña.

# Creamos un dataframe nuevo a partir de data_clean para realizar los cálculos sin modificar el original.
data_clean_ing = data_clean

# Recuperamos la variable 'minimum_nights' desde el dataset raw original (data_res_raw).
# Esto es necesario porque la columna 'minimum_nights' en data_clean fue transformada a categórica ('short-term'/'long-term').
# Necesitamos los valores numéricos originales para el cálculo de la ocupación.
original_min_nights = data_res_raw[['id', 'minimum_nights']].copy()

# Unimos los valores originales de minimum_nights al dataframe data_clean_ing usando el 'id' como clave.
# Se usa el sufijo '_original' para evitar conflictos con la columna 'minimum_nights' categórica existente.
data_clean_ing = data_clean_ing.merge(original_min_nights, on='id', how='left', suffixes=('', '_original'))

# Nos aseguramos de que la columna 'minimum_nights_original' sea de tipo numérico.
data_clean_ing['minimum_nights_original'] = pd.to_numeric(data_clean_ing['minimum_nights_original'], errors='coerce')

# Inicializamos la nueva columna 'estimated_occupancy_ltm' con valores de 0.0.
data_clean_ing['estimated_occupancy_ltm'] = 0.0

# --- Estimación para Listings con reseñas (>0 en number_of_reviews_ltm) ---

# Creamos una máscara booleana para identificar los listings que tienen reseñas en los últimos 12 meses.
mask_reviews = data_clean_ing['number_of_reviews_ltm'] > 0

# Creamos una máscara para identificar listings con minimum_nights original menor a la estadía promedio.
# Se utiliza la columna 'minimum_nights_original' recuperada del raw data.
mask_min_menor = data_clean_ing['minimum_nights_original'] < estadia_promedio

# Calculamos la ocupación estimada para listings con reseñas y minimum_nights < estadia_promedio.
# La fórmula usa la estadía promedio estimada.
data_clean_ing.loc[mask_reviews & mask_min_menor, 'estimated_occupancy_ltm'] = (
    estadia_promedio * data_clean_ing['number_of_reviews_ltm'] * 100 / (review_rate * 365)
)

# Calculamos la ocupación estimada para listings con reseñas y minimum_nights >= estadia_promedio.
# La fórmula usa el valor original de minimum_nights.
data_clean_ing.loc[mask_reviews & ~mask_min_menor, 'estimated_occupancy_ltm'] = (
    data_clean_ing['minimum_nights_original'] * data_clean_ing['number_of_reviews_ltm'] * 100 / (review_rate * 365)
)

# --- Estimación para Listings sin reseñas (number_of_reviews_ltm == 0) ---

# Filtramos los listings que no tienen reseñas en los últimos 12 meses.
sin_reviews = data_clean_ing[data_clean_ing['number_of_reviews_ltm'] == 0]

# Seleccionamos aleatoriamente un porcentaje de estos listings sin reseñas igual al review_rate.
# Esto simula que un porcentaje de listings sin reseñas podrían haber tenido ocupación.
# random_state=42 asegura que la selección sea la misma cada vez que se ejecuta el código.
seleccionados = sin_reviews.sample(frac=review_rate, random_state=42)

# Creamos una máscara para identificar los listings seleccionados sin reseñas con minimum_nights original < estadia_promedio.
# Se utiliza la columna 'minimum_nights_original'.
cond_menor = seleccionados['minimum_nights_original'] < estadia_promedio

# Calculamos la ocupación estimada para los listings seleccionados sin reseñas y minimum_nights < estadia_promedio.
# La fórmula usa la estadía promedio estimada.
data_clean_ing.loc[seleccionados[cond_menor].index, 'estimated_occupancy_ltm'] = (
    estadia_promedio * 100 / (review_rate * 365)
)

# Calculamos la ocupación estimada para los listings seleccionados sin reseñas y minimum_nights >= estadia_promedio.
# La fórmula usa el valor original de minimum_nights.
data_clean_ing.loc[seleccionados[~cond_menor].index, 'estimated_occupancy_ltm'] = (
    seleccionados.loc[~cond_menor, 'minimum_nights_original'] * 100 / (review_rate * 365)
)

# Limitamos el valor máximo de la ocupación estimada al 70% (fuente: https://insideairbnb.com/data-assumptions/)
data_clean_ing['estimated_occupancy_ltm'] = data_clean_ing['estimated_occupancy_ltm'].clip(upper=70)
### Creamos una variable que estima la ocupación a la que llamaremos estimated_occupancy_ltm
# Esta variable busca estimar el porcentaje de días que un anuncio estuvo ocupado en los últimos 12 meses.

# Parámetros utilizados en la estimación:
estadia_promedio = 5.7 # Estimación de la duración promedio de la estadía en noches (fuente: https://airbtics.com/tourism-statistics-buenos-aires-ar)
review_rate = 0.5 # Estimación de la proporción de huéspedes que dejan una reseña (fuente: https://insideairbnb.com/data-assumptions/)

# La fórmula para estimar la ocupación (estimated_occupancy_ltm) se basa en el número de reseñas en los últimos 12 meses (number_of_reviews_ltm)
# y el número mínimo de noches requeridas (minimum_nights).

# La lógica es diferente para avisos con y sin reseñas en el último año.
# Para listings con reseñas, se estima el número de estancias a partir de las reseñas (ajustado por el review_rate)
# y se multiplica por la duración de la estadía (usando la estadía promedio o el minimum_nights, lo que sea mayor).
# Para listings sin reseñas, se asigna una ocupación estimada a una proporción de listings (igual al review_rate)
# para simular que algunos podrían haber tenido ocupación sin dejar reseña.

# Creamos un dataframe nuevo a partir de data_clean para realizar los cálculos sin modificar el original.
data_clean_ing = data_clean

# Recuperamos la variable 'minimum_nights' desde el dataset raw original (data_res_raw).
# Esto es necesario porque la columna 'minimum_nights' en data_clean fue transformada a categórica ('short-term'/'long-term').
# Necesitamos los valores numéricos originales para el cálculo de la ocupación.
original_min_nights = data_res_raw[['id', 'minimum_nights']].copy()

# Unimos los valores originales de minimum_nights al dataframe data_clean_ing usando el 'id' como clave.
# Se usa el sufijo '_original' para evitar conflictos con la columna 'minimum_nights' categórica existente.
data_clean_ing = data_clean_ing.merge(original_min_nights, on='id', how='left', suffixes=('', '_original'))

# Nos aseguramos de que la columna 'minimum_nights_original' sea de tipo numérico.
data_clean_ing['minimum_nights_original'] = pd.to_numeric(data_clean_ing['minimum_nights_original'], errors='coerce')

# Inicializamos la nueva columna 'estimated_occupancy_ltm' con valores de 0.0.
data_clean_ing['estimated_occupancy_ltm'] = 0.0

# --- Estimación para Listings con reseñas (>0 en number_of_reviews_ltm) ---

# Creamos una máscara booleana para identificar los listings que tienen reseñas en los últimos 12 meses.
mask_reviews = data_clean_ing['number_of_reviews_ltm'] > 0

# Creamos una máscara para identificar listings con minimum_nights original menor a la estadía promedio.
# Se utiliza la columna 'minimum_nights_original' recuperada del raw data.
mask_min_menor = data_clean_ing['minimum_nights_original'] < estadia_promedio

# Calculamos la ocupación estimada para listings con reseñas y minimum_nights < estadia_promedio.
# La fórmula usa la estadía promedio estimada.
data_clean_ing.loc[mask_reviews & mask_min_menor, 'estimated_occupancy_ltm'] = (
    estadia_promedio * data_clean_ing['number_of_reviews_ltm'] * 100 / (review_rate * 365)
)

# Calculamos la ocupación estimada para listings con reseñas y minimum_nights >= estadia_promedio.
# La fórmula usa el valor original de minimum_nights.
data_clean_ing.loc[mask_reviews & ~mask_min_menor, 'estimated_occupancy_ltm'] = (
    data_clean_ing['minimum_nights_original'] * data_clean_ing['number_of_reviews_ltm'] * 100 / (review_rate * 365)
)

# --- Estimación para Listings sin reseñas (number_of_reviews_ltm == 0) ---

# Filtramos los listings que no tienen reseñas en los últimos 12 meses.
sin_reviews = data_clean_ing[data_clean_ing['number_of_reviews_ltm'] == 0]

# Seleccionamos aleatoriamente un porcentaje de estos listings sin reseñas igual al review_rate.
# Esto simula que un porcentaje de listings sin reseñas podrían haber tenido ocupación.
# random_state=42 asegura que la selección sea la misma cada vez que se ejecuta el código.
seleccionados = sin_reviews.sample(frac=review_rate, random_state=42)

# Creamos una máscara para identificar los listings seleccionados sin reseñas con minimum_nights original < estadia_promedio.
# Se utiliza la columna 'minimum_nights_original'.
cond_menor = seleccionados['minimum_nights_original'] < estadia_promedio

# Calculamos la ocupación estimada para los listings seleccionados sin reseñas y minimum_nights < estadia_promedio.
# La fórmula usa la estadía promedio estimada.
data_clean_ing.loc[seleccionados[cond_menor].index, 'estimated_occupancy_ltm'] = (
    estadia_promedio * 100 / (review_rate * 365)
)

# Calculamos la ocupación estimada para los listings seleccionados sin reseñas y minimum_nights >= estadia_promedio.
# La fórmula usa el valor original de minimum_nights.
data_clean_ing.loc[seleccionados[~cond_menor].index, 'estimated_occupancy_ltm'] = (
    seleccionados.loc[~cond_menor, 'minimum_nights_original'] * 100 / (review_rate * 365)
)

# Limitamos el valor máximo de la ocupación estimada al 70% (fuente: https://insideairbnb.com/data-assumptions/)
data_clean_ing['estimated_occupancy_ltm'] = data_clean_ing['estimated_occupancy_ltm'].clip(upper=70)

# Eliminamos la columna temporal 'minimum_nights_original' que ya no es necesaria.
data_clean_ing = data_clean_ing.drop(columns=['minimum_nights_original'])

data_clean_ing['estimated_occupancy_ltm'].describe()

# Histograma de estimated_occupancy_ltm
plt.figure(figsize=(10, 6))
sns.histplot(data_clean_ing['estimated_occupancy_ltm'], bins=30, kde=True)
plt.title('Distribución de la ocupación estimada en el último año')
plt.xlabel('% Ocupación estimada en el último año')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

# Box-plot de estimated_occupancy_ltm
plt.figure(figsize=(10, 6))
sns.boxplot(x=data_clean_ing['estimated_occupancy_ltm'])
plt.title('Box-plot de la ocupación estimada en el último año')
plt.xlabel('% Ocupación estimada en el último año')
plt.tight_layout()
plt.show()

# Descripción estadística de la columna estimated_occupancy_ltm
print("\nDescripción estadística de la columna 'estimated_occupancy_ltm':")
print(data_clean_ing['estimated_occupancy_ltm'].describe())

# Mediana y Moda de la columna estimated_occupancy_ltm
mediana_occupancy_ltm = data_clean_ing['estimated_occupancy_ltm'].median()
moda_occupancy_ltm = data_clean_ing['estimated_occupancy_ltm'].mode()

print(f'Moda de la ocupación estimada en el último año: {moda_occupancy_ltm.iloc[0]}')
print(f"Mediana de la ocupación estimada en el último año: {mediana_occupancy_ltm}")

### Creamos una variable llamada ingresos_ltm
# Es el producto de price y estimated_occupancy_ltm

data_clean_ing['ingresos_ltm'] = data_clean_ing['price'] * data_clean_ing['estimated_occupancy_ltm']*365/100

# Histograma de ingresos_ltm
plt.figure(figsize=(10, 6))
sns.histplot(data_clean_ing['ingresos_ltm'], bins=30, kde=True)
plt.title('Distribución de los ingresos estimados en el último año')
plt.xlabel('Ingresos estimados en el último año')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

# Box-plot de ingresos_ltm
plt.figure(figsize=(10, 6))
sns.boxplot(x=data_clean_ing['ingresos_ltm'])
plt.title('Box-plot de los ingresos estimados en el último año')
plt.xlabel('Ingresos estimados en el último año')
plt.tight_layout()
plt.show()

# Descripción estadística de la columna ingresos_ltm
print("\nDescripción estadística de la columna 'ingresos_ltm':")
print(data_clean_ing['ingresos_ltm'].describe())

# Mediana y Moda de la columna ingresos_ltm
mediana_ingresos_ltm = data_clean_ing['ingresos_ltm'].median()
moda_ingresos_ltm = data_clean_ing['ingresos_ltm'].mode()

print(f'Moda de los ingresos estimados en el último año: {moda_ingresos_ltm.iloc[0]}')
print(f"Mediana de los ingresos estimados en el último año: {mediana_ingresos_ltm}")

# Calcular métricas por barrio
ingresos_mean = data_clean_ing.groupby("neighbourhood")["ingresos_ltm"].mean()
listing_counts = data_clean_ing["neighbourhood"].value_counts()
listing_pct = listing_counts / listing_counts.sum() * 100

# Unir métricas
summary_df_ingresos = pd.DataFrame({
    "ingresos_ltm": ingresos_mean,
    "listings": listing_counts,
    "listing_pct": listing_pct
}).reset_index().rename(columns={"index": "neighbourhood"})

# Unir con datos geográficos
barrios_merge_ingresos = barrios.merge(summary_df_ingresos, on="neighbourhood", how="left")

# Crear figura
fig, ax = plt.subplots(figsize=(12, 10))
barrios_merge_ingresos.plot(column="ingresos_ltm", cmap=custom_cmap, legend=True,
                   edgecolor="black", linewidth=0.4, ax=ax)

# Agregar textos a cada barrio
for _, row in barrios_merge_ingresos.iterrows():
    if pd.notna(row["ingresos_ltm"]):
        x, y = row["geometry"].centroid.x, row["geometry"].centroid.y
        barrio = row["neighbourhood"]
        ingresos = row["ingresos_ltm"]
        count = int(row["listings"])
        pct = row["listing_pct"]

        ax.text(x, y + 0.002, f"{barrio}", fontsize=8, ha='center', weight='bold', color='black')
        ax.text(x, y, f"{ingresos:,.0f}", fontsize=8, ha='center', color='black') # Formato sin decimales
        ax.text(x, y - 0.002, f"{count} ({pct:.1f}%)", fontsize=7, ha='center', color='black')

# Título y referencia textual
plt.title("Promedio de ingresos estimados en el último año y anuncios por barrio", fontsize=14)

# Referencia textual (arriba derecha, texto centrado en el recuadro)
text = (
    "Referencia:\n"
    "Nombre del barrio\n"
    "Ingresos LTM (prom)\n"
    "Cantidad de listings (%)"
)
plt.gcf().text(
    0.755, 0.955, text,
    fontsize=9, va='top', ha='center',
    family='monospace',
    bbox=dict(facecolor='white', alpha=0.8)
)

plt.axis("off")
plt.tight_layout()
plt.show()

"""Puerto Madero es el barrio que genera mayores ingresos anuales

Para tener una idea aproximada de la rentabilidad buscamos datos de precio del metro cuadrado en los distintos barrios de Buenos Aires
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL de la página
url = 'https://mudafy.com.ar/d/valor-metro-cuadrado-en-caba-por-barrio'

# Solicitud HTTP con encoding explícito
response = requests.get(url)
response.encoding = 'utf-8'  # Fuerza la codificación UTF-8

# Analizar el contenido HTML
soup = BeautifulSoup(response.text, 'html.parser')

# Buscar la tabla
table = soup.find('table')
if not table:
    raise ValueError("No se encontró ninguna tabla en la página.")

# Extraer encabezados
headers = [th.get_text(strip=True) for th in table.find_all('th')]

# Extraer filas
rows = []
for tr in table.find_all('tr')[1:]:
    cells = [td.get_text(strip=True) for td in tr.find_all('td')]
    if cells:
        rows.append(cells)

# Crear dataframe
precios_m2 = pd.DataFrame(rows, columns=headers)

# Mostrar los datos
print(precios_m2.head())

# Comparamos los barrios de data_clean_ing y precios_m2

barrios_data_clean = data_clean_ing['neighbourhood'].unique()
barrios_precios_m2 = precios_m2['Barrio'].unique()

print("Barrios en data_clean_ing:", sorted(barrios_data_clean))
print("\nBarrios en precios_m2:", sorted(barrios_precios_m2))

# Encontrar barrios que están en data_clean_ing pero no en precios_m2
barrios_solo_en_data_clean = set(barrios_data_clean) - set(barrios_precios_m2)
print("\nBarrios solo en data_clean_ing:", sorted(list(barrios_solo_en_data_clean)))

# Encontrar barrios que están en precios_m2 pero no en data_clean_ing
barrios_solo_en_precios_m2 = set(barrios_precios_m2) - set(barrios_data_clean)
print("\nBarrios solo en precios_m2:", sorted(list(barrios_solo_en_precios_m2)))

# Encontrar barrios que están en ambos dataframes
barrios_en_ambos = set(barrios_data_clean).intersection(set(barrios_precios_m2))
print("\nBarrios en ambos dataframes:", sorted(list(barrios_en_ambos)))

print(f"\nNúmero de barrios en data_clean_ing: {len(barrios_data_clean)}")
print(f"Número de barrios en precios_m2: {len(barrios_precios_m2)}")
print(f"Número de barrios en ambos: {len(barrios_en_ambos)}")

# Renombramos los siguientes barrios de precios_m2 para que coincidan con los de data_clean_ing
# Agronomía => Agronomia
# Constitución => Constitucion
# La Boca => Boca

replacements = {
    'Agronomía': 'Agronomia',
    'Constitución': 'Constitucion',
    'La Boca': 'Boca'
}

precios_m2['Barrio'] = precios_m2['Barrio'].replace(replacements)

# Verificar los cambios en la columna 'Barrio'
print("Después de renombrar:")
print(precios_m2['Barrio'].unique())

#Opcional: volver a verificar la intersección si es necesario
barrios_data_clean_updated = data_clean_ing['neighbourhood'].unique()
barrios_precios_m2_updated = precios_m2['Barrio'].unique()
barrios_en_ambos_updated = set(barrios_data_clean_updated).intersection(set(barrios_precios_m2_updated))
print("\nBarrios en ambos dataframes (después de renombrar):", sorted(list(barrios_en_ambos_updated)))
print(f"Número de barrios en ambos (después de renombrar): {len(barrios_en_ambos_updated)}")

# Volvemos a comparar

barrios_data_clean = data_clean_ing['neighbourhood'].unique()
barrios_precios_m2 = precios_m2['Barrio'].unique()

print("Barrios en data_clean_ing:", sorted(barrios_data_clean))
print("\nBarrios en precios_m2:", sorted(barrios_precios_m2))

# Encontrar barrios que están en data_clean_ing pero no en precios_m2
barrios_solo_en_data_clean = set(barrios_data_clean) - set(barrios_precios_m2)
print("\nBarrios solo en data_clean_ing:", sorted(list(barrios_solo_en_data_clean)))

# Encontrar barrios que están en precios_m2 pero no en data_clean_ing
barrios_solo_en_precios_m2 = set(barrios_precios_m2) - set(barrios_data_clean)
print("\nBarrios solo en precios_m2:", sorted(list(barrios_solo_en_precios_m2)))

# Encontrar barrios que están en ambos dataframes
barrios_en_ambos = set(barrios_data_clean).intersection(set(barrios_precios_m2))
print("\nBarrios en ambos dataframes:", sorted(list(barrios_en_ambos)))

print(f"\nNúmero de barrios en data_clean_ing: {len(barrios_data_clean)}")
print(f"Número de barrios en precios_m2: {len(barrios_precios_m2)}")
print(f"Número de barrios en ambos: {len(barrios_en_ambos)}")

# Eliminamos los caracteres no numericos y pasamos a float la columna 'Valor m2 (USD)' en precios_m2

# Eliminar caracteres no numéricos
precios_m2['Valor m2 (USD)'] = precios_m2['Valor m2 (USD)'].astype(str).str.replace('[$,.]', '', regex=True)

# Convertimos a float
precios_m2['Valor m2 (USD)'] = pd.to_numeric(precios_m2['Valor m2 (USD)'], errors='coerce')

# Mostramos las primeras filas y los tipos de datos para verificar el cambio
print("\nprecios_m2 después de limpiar 'Valor m2 (USD)':")
print(precios_m2.head())
print("\nTipos de datos después de la limpieza:")
print(precios_m2.info())

# Agregamos una columna en data_clean_ing lamada precio_m2_usd y le asignamos su valor segun el barrio al que pertenece el aviso
# y tomando los valores de precios_m2

# Creamos un diccionario a partir de precios_m2
precio_m2_dict = precios_m2.set_index('Barrio')['Valor m2 (USD)'].to_dict()

# Mapear la columna 'neighbourhood' de data_clean_ing a los valores del diccionario
# Esto creará la nueva columna 'precio_m2_usd'
# Si un barrio en data_clean_ing no está en el diccionario, el valor en 'precio_m2_usd' será NaN (missing).
data_clean_ing['precio_m2_usd'] = data_clean_ing['neighbourhood'].map(precio_m2_dict)

# Mostramos los barrios en data_clean_ing que quedaron con missing values en precio_m2_usd
barrios_con_missing_precio_m2 = data_clean_ing[data_clean_ing['precio_m2_usd'].isnull()]['neighbourhood'].unique()

print("\nNeighbourhoods en data_clean_ing con missing values en 'precio_m2_usd':")
if len(barrios_con_missing_precio_m2) > 0:
    for barrio in sorted(barrios_con_missing_precio_m2):
        print(f"- {barrio}")
else:
    print("No hay neighbourhoods con missing values en 'precio_m2_usd'.")

data_clean_ing.head()

# En las publicaciones con missing values en precio_m2_usd, estimamos el precio del metro cuadrado
# para cada barrio utilizando el promedio de precio_m2_usd en los barrios limitrofes usando el geojson de insideairbnb

# Identificar los barrios con missing values en 'precio_m2_usd'
missing_barrios = data_clean_ing[data_clean_ing['precio_m2_usd'].isnull()]['neighbourhood'].unique()

# Cargar el archivo GeoJSON de barrios
url_geojson = "https://data.insideairbnb.com/argentina/ciudad-aut%C3%B3noma-de-buenos-aires/buenos-aires/2025-01-29/visualisations/neighbourhoods.geojson"
barrios_geo = gpd.read_file(url_geojson)

# Crear un diccionario para mapear nombres de barrios a sus geometrías
barrio_geometries = barrios_geo.set_index('neighbourhood')['geometry']

# Crear un diccionario para almacenar vecinos (barrios adyacentes)
# Esto puede ser computacionalmente costoso, podrías precalcularlo si el GeoJSON no cambia.
barrio_neighbors = {}
for index, row in barrios_geo.iterrows():
    barrio_name = row['neighbourhood']
    neighbors = []
    # Iterate through all other barrios to check for touching geometries
    for other_index, other_row in barrios_geo.iterrows():
        if index != other_index:
            other_barrio_name = other_row['neighbourhood']
            # Check if the geometries touch or intersect (excluding self-intersection)
            if row['geometry'].touches(other_row['geometry']) or row['geometry'].intersects(other_row['geometry']):
                 # Ensure they are not the same geometry (intersects can include self)
                 if not row['geometry'].equals(other_row['geometry']):
                    neighbors.append(other_barrio_name)
    barrio_neighbors[barrio_name] = neighbors


# Calcular el promedio de precio_m2_usd para cada barrio disponible (sin missing values)
barrio_avg_price = data_clean_ing.groupby('neighbourhood')['precio_m2_usd'].mean().dropna().to_dict()

# Función para calcular el promedio de precio_m2_usd de los barrios vecinos
def get_neighbor_avg_price(barrio_name, barrio_neighbors, barrio_avg_price):
    neighbors = barrio_neighbors.get(barrio_name, [])
    neighbor_prices = [barrio_avg_price.get(neighbor) for neighbor in neighbors if barrio_avg_price.get(neighbor) is not None]

    if neighbor_prices:
        return np.mean(neighbor_prices)
    else:
        return np.nan # Return NaN if no neighbors with valid prices are found

# Rellenar los missing values en 'precio_m2_usd' utilizando el promedio de los vecinos
for barrio in missing_barrios:
    # Get the average price of neighbors
    neighbor_avg = get_neighbor_avg_price(barrio, barrio_neighbors, barrio_avg_price)

    # Fill missing values for listings in this barrio
    if not np.isnan(neighbor_avg):
        data_clean_ing.loc[data_clean_ing['neighbourhood'] == barrio, 'precio_m2_usd'] = neighbor_avg
        print(f"Filled missing precio_m2_usd for '{barrio}' with neighbor average: {neighbor_avg:.2f}")
    else:
        print(f"Could not fill missing precio_m2_usd for '{barrio}': No neighbors with valid price data.")

# Verificar si todavía quedan missing values en 'precio_m2_usd'
remaining_missing = data_clean_ing['precio_m2_usd'].isnull().sum()
print(f"\nRemaining missing values in 'precio_m2_usd' after filling with neighbor average: {remaining_missing}")

# Opcional: Mostrar los barrios que aún tienen missing values (si los hay)
if remaining_missing > 0:
    still_missing_barrios = data_clean_ing[data_clean_ing['precio_m2_usd'].isnull()]['neighbourhood'].unique()
    print("\nNeighbourhoods that still have missing values in 'precio_m2_usd':")
    for barrio in sorted(still_missing_barrios):
        print(f"- {barrio}")

# Chequeamos que ya no hay missing values

print("\nMissing values en 'precio_m2_usd' column:")
print(data_clean_ing['precio_m2_usd'].isnull().sum())

# Convertimos el precio del metro cuadrado a pesos

tipo_de_cambio = 1052 # Usamos el valor de la fecha del scrapeo (feunte: https://www.bcra.gob.ar/PublicacionesEstadisticas/Cotizaciones_por_fecha_2.asp)
data_clean_ing['precio_m2'] = tipo_de_cambio * data_clean_ing['precio_m2_usd']

# Graficamos el promedio del cociente entre ingresos y precio del metro cuadrado para cada barrio

# Calcular el cociente entre ingresos_ltm y precio_m2
data_clean_ing['cociente_ingresos_precio'] = data_clean_ing['ingresos_ltm'] / data_clean_ing['precio_m2']

# Calcular el cociente promedio por barrio
cociente_promedio_barrio = data_clean_ing.groupby("neighbourhood")["cociente_ingresos_precio"].mean()

# Calcular métricas adicionales por barrio (listings, porcentaje)
listing_counts = data_clean_ing["neighbourhood"].value_counts()
listing_pct = listing_counts / listing_counts.sum() * 100

# Unir métricas
summary_df_cociente = pd.DataFrame({
    "cociente_ingresos_precio": cociente_promedio_barrio,
    "listings": listing_counts,
    "listing_pct": listing_pct
}).reset_index().rename(columns={"index": "neighbourhood"})

# Unir con datos geográficos de barrios de CABA
# Asumiendo que el GeoDataFrame 'barrios' ya está cargado desde la celda anterior
# url_shapefile = "https://data.insideairbnb.com/argentina/ciudad-aut%C3%B3noma-de-buenos-aires/buenos-aires/2025-01-29/visualisations/neighbourhoods.geojson"
# barrios = gpd.read_file(url_shapefile) # Asegúrate de que esta línea esté descomentada si no se ejecutó antes

barrios_merge_cociente = barrios.merge(summary_df_cociente, on="neighbourhood", how="left")

# Crear figura para el mapa
fig, ax = plt.subplots(figsize=(14, 12))
ax.set_title("Cociente Promedio Ingresos LTM / Precio m² por Barrio", fontsize=16)
ax.set_axis_off()

# Colorear por el cociente promedio
barrios_merge_cociente.plot(column="cociente_ingresos_precio", ax=ax, legend=True,
                            cmap='RdYlGn',  # Un colormap divergente (Rojo-Amarillo-Verde)
                            edgecolor="black", linewidth=0.5,
                            legend_kwds={'label': "Cociente Promedio Ingresos LTM / Precio m²",
                                         'orientation': "vertical"})

# Agregar textos a cada barrio
for idx, row in barrios_merge_cociente.iterrows():
    if pd.notna(row["cociente_ingresos_precio"]):
        # Evitar errores si la geometría no es un polígono simple
        try:
            x, y = row["geometry"].centroid.x, row["geometry"].centroid.y
        except Exception:
            continue # Saltar si el cálculo del centroide falla

        barrio = row["neighbourhood"]
        cociente = row["cociente_ingresos_precio"]
        count = int(row["listings"])
        pct = row["listing_pct"]

        # Texto a mostrar: Nombre del barrio, Cociente Promedio, Cantidad de Listings (%)
        text = f"{barrio}\nCociente: {cociente:.2f}\nListings: {count} ({pct:.1f}%)"

        # Ajustar posición y estilo del texto
        # Usar un color de texto que contraste bien con el colormap RdYlGn
        # Podríamos hacerlo condicionalmente (ej: blanco para valores bajos/medios, negro para valores altos)
        # O simplemente usar un color como negro con un fondo semitransparente.
        text_color = 'black' # Color por defecto
        bbox_props = dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6) # Fondo blanco semitransparente

        # Ejemplo de color condicional (ajustar umbral según la distribución del cociente)
        # if cociente < barrios_merge_cociente['cociente_ingresos_precio'].median():
        #     text_color = 'white'
        #     bbox_props = dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5)


        ax.text(x, y, text, fontsize=7, ha='center', va='center', color=text_color, bbox=bbox_props)

plt.tight_layout()
plt.show()

# Opcional: Graficar el cociente promedio por barrio como un gráfico de barras
plt.figure(figsize=(15, 8))
cociente_promedio_barrio_sorted = cociente_promedio_barrio.sort_values(ascending=False)
sns.barplot(x=cociente_promedio_barrio_sorted.index, y=cociente_promedio_barrio_sorted.values, palette='viridis')
plt.xticks(rotation=90)
plt.title('Cociente Promedio Ingresos LTM / Precio m² por Barrio', fontsize=14)
plt.xlabel('Barrio', fontsize=12)
plt.ylabel('Cociente Promedio', fontsize=12)
plt.tight_layout()
plt.show()

"""Monte Castro y Barracas parecen ser los barrios de mayor rentabilidad. Sin embargo son barrios con muy pocas publicaciones. ENtre los barrios con más avisos se destaca San Nicolas como más rentable por sobre Palermo y Puerto Madero.

Sin embargo estamos incluyendo todo tipo de propiedades de distintas dimensiones y relativizando al valor del metro cuadrado. Además distintos barrios pueden mostar distinta distribución de tamaños de alojamientos.

Por lo tango graficamos el msimo tipo de mapa pero para cada valor posible de la variable bedrooms (para independizarnos aunque sea de manera aproximada de los tamaños de las propiedades)
"""

# Define the desired order for categorical variables
order_mapping = {
    'bedrooms': ['1 amb', '2 amb', '3 amb', '4 amb', '5 o más amb']
}

# Get the unique values of 'bedrooms' to create a map for each
bedroom_categories = data_clean_ing['bedrooms'].unique()

# Sort the bedroom categories using the defined order_mapping
bedroom_categories = sorted(bedroom_categories, key=lambda x: order_mapping.get('bedrooms', []).index(x) if x in order_mapping.get('bedrooms', []) else len(order_mapping.get('bedrooms', []))) # Order them logically

# Calcular el average cociente entre ingresos y precio del metro cuadrado para cada barrio y cada tipo de habitación
cociente_promedio_barrio_bedrooms = data_clean_ing.groupby(['neighbourhood', 'bedrooms'])["cociente_ingresos_precio"].mean().reset_index()

# Cargar geometría de barrios de CABA (Asegúrate de que esta celda ya se haya ejecutado)
# url_shapefile = "https://data.insideairbnb.com/argentina/ciudad-aut%C3%B3noma-de-buenos-aires/buenos-aires/2025-01-29/visualisations/neighbourhoods.geojson"
# barrios = gpd.read_file(url_shapefile)

# Determine the global min and max of the cociente for consistent color scaling across maps
global_min_cociente = cociente_promedio_barrio_bedrooms['cociente_ingresos_precio'].min()
global_max_cociente = cociente_promedio_barrio_bedrooms['cociente_ingresos_precio'].max()

# Create a map for each bedroom category
for bedroom_cat in bedroom_categories:
    # Filter data for the current bedroom category
    cociente_filtered = cociente_promedio_barrio_bedrooms[cociente_promedio_barrio_bedrooms['bedrooms'] == bedroom_cat].copy()

    # Merge with barrio geometries
    barrios_merge_cociente_bedroom = barrios.merge(cociente_filtered, on="neighbourhood", how="left")

    # Create figure for the map
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_title(f"Cociente Promedio Ingresos LTM / Precio m² por Barrio ({bedroom_cat})", fontsize=16)
    ax.set_axis_off()

    # Plot the map, colored by the average cociente for this bedroom category
    # Use vmin and vmax from global min/max for consistent color scale
    barrios_merge_cociente_bedroom.plot(column="cociente_ingresos_precio", ax=ax, legend=True,
                                        cmap='RdYlGn',  # Un colormap divergente (Rojo-Amarillo-Verde)
                                        edgecolor="black", linewidth=0.5,
                                        vmin=global_min_cociente, vmax=global_max_cociente, # Consistent color scale
                                        legend_kwds={'label': f"Cociente Promedio Ingresos LTM / Precio m² ({bedroom_cat})",
                                                     'orientation': "vertical"})

    # Add texts to each barrio
    # Optional: Only add text for barrios that actually have data for this bedroom category
    # Corrected the column name from 'cociente_promedio_precio' to 'cociente_ingresos_precio'
    barrios_with_data = barrios_merge_cociente_bedroom[pd.notna(barrios_merge_cociente_bedroom['cociente_ingresos_precio'])]
    for idx, row in barrios_with_data.iterrows():
        try:
            x, y = row["geometry"].centroid.x, row["geometry"].centroid.y
        except Exception:
            continue

        barrio = row["neighbourhood"]
        cociente = row["cociente_ingresos_precio"]

        text = f"{barrio}\n{cociente:.2f}"

        text_color = 'black'
        bbox_props = dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6)

        ax.text(x, y, text, fontsize=7, ha='center', va='center', color=text_color, bbox=bbox_props)

    plt.tight_layout()
    plt.show()

# Calcular el cociente promedio de ingresos sobre precio del metro cuadrado para cada barrio y valor de bedrooms
cociente_promedio_barrio_bedrooms = data_clean_ing.groupby(['neighbourhood', 'bedrooms'])["cociente_ingresos_precio"].mean().reset_index()

# Renombrar la columna del cociente para mayor claridad
cociente_promedio_barrio_bedrooms = cociente_promedio_barrio_bedrooms.rename(columns={"cociente_ingresos_precio": "cociente_promedio_img_m2"})

# Ordenar la tabla por el cociente promedio de forma decreciente
tabla_ordenada = cociente_promedio_barrio_bedrooms.sort_values(by="cociente_promedio_img_m2", ascending=False)

# Mostrar la tabla
print("Tabla de cociente promedio de ingresos sobre precio del metro cuadrado por barrio y número de dormitorios (ordenada decrecientemente por cociente):")
tabla_ordenada

# Seleccionar los 10 barrios con más listings en `data_clean_ing`
top_10_neighbourhoods_ing = data_clean_ing['neighbourhood'].value_counts().nlargest(10).index.tolist()

# Filtrar el dataframe `data_clean_ing` para incluir solo los listings de estos 10 barrios
data_top10_ing = data_clean_ing[data_clean_ing['neighbourhood'].isin(top_10_neighbourhoods_ing)].copy()

# Calcular el cociente promedio de ingresos sobre precio del metro cuadrado para los 10 barrios con más anuncios y cada valor de bedrooms
cociente_promedio_top10_bedrooms = data_top10_ing.groupby(['neighbourhood', 'bedrooms'])["cociente_ingresos_precio"].mean().reset_index()

# Renombrar la columna del cociente para mayor claridad
cociente_promedio_top10_bedrooms = cociente_promedio_top10_bedrooms.rename(columns={"cociente_ingresos_precio": "cociente_promedio_img_m2"})

# Ordenar la tabla por el cociente promedio de forma decreciente
tabla_ordenada_top10 = cociente_promedio_top10_bedrooms.sort_values(by="cociente_promedio_img_m2", ascending=False)

# Mostrar la tabla solo para los 10 barrios con más listings
print("Tabla de cociente promedio de ingresos sobre precio del metro cuadrado por barrio y número de dormitorios (solo para los 10 barrios con más listings, ordenada decrecientemente por cociente):")
tabla_ordenada_top10