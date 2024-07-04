
# LIBRERIAS DE LECTURA Y CONEXION
import pandas as pd
import numpy as np
from db_connection import MySQLDatabase
import config

# LIBRERIAS DE PRE PROCESAMIENTO, PROCESAMIENTO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def main():

    # Configuración de la conexión
    db = MySQLDatabase(
        host=config.host,
        user=config.user,
        password=config.password,
        database=config.database
    )

    # Definicion de Cabeceras
    Atributos = ['Id', 'Usuario', 'Tristeza', 'Pesimismo', 'Fracaso', 'Perdida_Placer', 'Sentimiento_Culpa', 'Sentimiento_Castigo', 'Disconformidad', 'Autocritica', 'Pensamiento_Suicidio', 'Llanto', 'Agitacion', 'Perdida_Interes', 'Indecision', 'Desvalorizacion', 'Perdida_Energia', 'Habitos_Sueno', 'Irritabilidad', 'Cambios_Apetito', 'Dificultad_Concentracion', 'Casancio', 'Perdida_Interes_S', 'Totales']
    # Conectar a la base de datos
    db.connect()

    if db.connection is not None and db.connection.is_connected():
    
       # Leer la tabla con nuevas cabeceras
        try:
            # Llamada a la Funcion de lectura de registros de la BD.
            dataFrame = db.read_table_with_headers('respuestas', Atributos)
            
            # ESCALAMIENTO
            #================================== ANTES DEL ESCALAMIENTO
            #HISTOGRAMAS UNIVARIABLE
            import matplotlib.pyplot as ptl
            import seaborn as sns
            f, ax =ptl.subplots(4,6,figsize=(7,21))
            sns.histplot(dataFrame["Tristeza"], ax=ax[0, 0], kde=True, stat="density")
            sns.histplot(dataFrame["Pesimismo"], ax=ax[0, 1], kde=True, stat="density")
            sns.histplot(dataFrame["Fracaso"], ax=ax[0, 2], kde=True, stat="density")
            sns.histplot(dataFrame["Perdida_Placer"], ax=ax[0, 3], kde=True, stat="density")
            sns.histplot(dataFrame["Sentimiento_Culpa"], ax=ax[0, 4], kde=True, stat="density")
            sns.histplot(dataFrame["Sentimiento_Castigo"], ax=ax[0, 5], kde=True, stat="density")
            sns.histplot(dataFrame["Disconformidad"], ax=ax[1, 0], kde=True, stat="density")
            sns.histplot(dataFrame["Autocritica"], ax=ax[1, 1], kde=True, stat="density")
            sns.histplot(dataFrame["Pensamiento_Suicidio"], ax=ax[1, 2], kde=True, stat="density")
            sns.histplot(dataFrame["Llanto"], ax=ax[1, 3], kde=True, stat="density")
            sns.histplot(dataFrame["Agitacion"], ax=ax[1, 4], kde=True, stat="density")
            sns.histplot(dataFrame["Perdida_Interes"], ax=ax[1, 5], kde=True, stat="density")
            sns.histplot(dataFrame["Indecision"], ax=ax[2, 0], kde=True, stat="density")
            sns.histplot(dataFrame["Desvalorizacion"], ax=ax[2, 1], kde=True, stat="density")
            sns.histplot(dataFrame["Perdida_Energia"], ax=ax[2, 2], kde=True, stat="density")
            sns.histplot(dataFrame["Habitos_Sueno"], ax=ax[2, 3], kde=True, stat="density")
            sns.histplot(dataFrame["Irritabilidad"], ax=ax[2, 4], kde=True, stat="density")
            sns.histplot(dataFrame["Cambios_Apetito"], ax=ax[2, 5], kde=True, stat="density")
            sns.histplot(dataFrame["Dificultad_Concentracion"], ax=ax[3, 0], kde=True, stat="density")
            sns.histplot(dataFrame["Casancio"], ax=ax[3, 1], kde=True, stat="density")
            sns.histplot(dataFrame["Perdida_Interes_S"], ax=ax[3, 2], kde=True, stat="density")
            ptl.show()

            
            # ESCALAMIENTO
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0,1))
            rescaledX = scaler.fit_transform(dataFrame)
            # Summarize transformed data
            np.set_printoptions(precision=2)
            print(Atributos)
            # :,: Son los limites de muestreo de datos en la matriz, X e Y. (0:5,:)-> mostrar columnas desde la 0 hasta la 5.
            print(rescaledX[:,:])
            print(type(rescaledX))

            #================================== DESPUES DEL ESCALAMIENTO
            #HISTOGRAMAS UNIVARIABLE
            import matplotlib.pyplot as ptl2
            import seaborn as sns2
            f, ax =ptl2.subplots(4,6,figsize=(7,21))
            sns2.histplot(rescaledX[:,0:1], ax=ax[0, 0], kde=True, stat="density")
            sns2.histplot(rescaledX[:,1:2], ax=ax[0, 1], kde=True, stat="density")
            sns2.histplot(rescaledX[:,2:3], ax=ax[0, 2], kde=True, stat="density")
            sns2.histplot(rescaledX[:,3:4], ax=ax[0, 3], kde=True, stat="density")
            sns2.histplot(rescaledX[:,4:5], ax=ax[0, 4], kde=True, stat="density")
            sns2.histplot(rescaledX[:,5:6], ax=ax[0, 5], kde=True, stat="density")
            sns2.histplot(rescaledX[:,6:7], ax=ax[1, 0], kde=True, stat="density")
            sns2.histplot(rescaledX[:,7:8], ax=ax[1, 1], kde=True, stat="density")
            sns2.histplot(rescaledX[:,8:9], ax=ax[1, 2], kde=True, stat="density")
            sns2.histplot(rescaledX[:,9:10], ax=ax[1, 3], kde=True, stat="density")
            sns2.histplot(rescaledX[:,10:11], ax=ax[1, 4], kde=True, stat="density")
            sns2.histplot(rescaledX[:,11:12], ax=ax[1, 5], kde=True, stat="density")
            sns2.histplot(rescaledX[:,12:13], ax=ax[2, 0], kde=True, stat="density")
            sns2.histplot(rescaledX[:,13:14], ax=ax[2, 1], kde=True, stat="density")
            sns2.histplot(rescaledX[:,14:15], ax=ax[2, 2], kde=True, stat="density")
            sns2.histplot(rescaledX[:,15:16], ax=ax[2, 3], kde=True, stat="density")
            sns2.histplot(rescaledX[:,16:17], ax=ax[2, 4], kde=True, stat="density")
            sns2.histplot(rescaledX[:,17:18], ax=ax[2, 5], kde=True, stat="density")
            sns2.histplot(rescaledX[:,18:19], ax=ax[3, 0], kde=True, stat="density")
            sns2.histplot(rescaledX[:,19:20], ax=ax[3, 1], kde=True, stat="density")
            sns2.histplot(rescaledX[:,21:22], ax=ax[3, 2], kde=True, stat="density")
            ptl2.show()

            # Mostrar las primeras filas del DataFrame
            print(dataFrame.head())
            print(f'Cantidad de filas y Columnas: {dataFrame.shape}')
            print(f'Nombre de Atributos Procesados: {dataFrame.columns}')
            # SUMA DE NULOS
            print('Cantidad de Nulos por Atributo.')
            print(dataFrame.isnull().sum())
            # TIPOS DE DATOS POR CATEGORIA
            print('Tipos de dato por Categoria.')
            print(dataFrame.info())
            # MIN-MAX-MEDIA-DESV.ESTANDAR
            print('Datos Pre Proceso')
            print(dataFrame.describe())

            # Columnas con las respuestas y la última columna es la sumatoria
            X = dataFrame.iloc[:, 2:23]  # Características (respuestas)
            y = dataFrame.iloc[:, 23]    # Variable objetivo (sumatoria de respuestas)

            print("===========DATOS DE ENTRENAMIENTO==========")
            print(X.head())
            print("===========VARIABLE OBJETIVO==========")
            print(y.head())

            # Dividir los datos en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Crear el modelo
            model = RandomForestRegressor(n_estimators=100, random_state=42)

            # Entrenar el modelo
            model.fit(X_train, y_train)

            # Realizar predicciones
            y_pred = model.predict(X_test)

            # Evaluar el modelo
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print(f"Error cuadrático medio (MSE): {mse}")
            print(f"Coeficiente de determinación (R²): {r2}")

            # Supongamos que tenemos un nuevo conjunto de respuestas
            new_data = pd.DataFrame({
                'Tristeza':[0],
                'Pesimismo':[1],
                'Fracaso':[0],
                'Perdida_Placer':[3],
                'Sentimiento_Culpa':[1],
                'Sentimiento_Castigo':[1],
                'Disconformidad':[0],
                'Autocritica':[1],
                'Pensamiento_Suicidio':[2],
                'Llanto':[0],
                'Agitacion':[1],
                'Perdida_Interes':[1],
                'Indecision':[0],
                'Desvalorizacion':[0],
                'Perdida_Energia':[0],
                'Habitos_Sueno':[1],
                'Irritabilidad':[1],
                'Cambios_Apetito':[3],
                'Dificultad_Concentracion':[3],
                'Casancio':[3],
                'Perdida_Interes_S':[2]
            })

            # Predecir la sumatoria de respuestas para los nuevos datos
            prediccion = model.predict(new_data)
            print(f"Predicción de la sumatoria de respuestas: {prediccion[0]}")

        except ValueError as e:
            print(e)

    else:
        print("No se pudo conectar a la base de datos")

if __name__ == "__main__":
    main()
