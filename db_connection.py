# pip install mysql-connector-python
import pandas as pd
import mysql.connector
from mysql.connector import Error

class MySQLDatabase:

    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            if self.connection.is_connected():
                print("Conexión exitosa a la base de datos")
        except Error as e:
            print(f"Error al conectar a la base de datos: {e}")
            self.connection = None

    def disconnect(self):
        if self.connection is not None and self.connection.is_connected():
            self.connection.close()
            print("Desconexión exitosa de la base de datos")

    def execute_query(self, query, params=None):
        if self.connection is not None and self.connection.is_connected():
            cursor = self.connection.cursor()
            try:
                cursor.execute(query, params)
                self.connection.commit()
                print("Consulta ejecutada exitosamente")
            except Error as e:
                print(f"Error al ejecutar la consulta: {e}")
            finally:
                cursor.close()

    def read_table_to_dataframe(self, table_name):
        if self.connection is None or not self.connection.is_connected():
            self.connect()
        
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, self.connection)
        return df

    def read_table_with_headers(self, table_name, headers):
            df = self.read_table_to_dataframe(table_name)
            if len(headers) == len(df.columns):
                df.columns = headers
            else:
                raise ValueError("El número de nuevas cabeceras no coincide con el número de columnas en el DataFrame.")
            return df

    def fetch_query(self, query, params=None):
        result = None
        if self.connection is not None and self.connection.is_connected():
            cursor = self.connection.cursor()
            try:
                cursor.execute(query, params)
                result = cursor.fetchall()
            except Error as e:
                print(f"Error al ejecutar la consulta: {e}")
            finally:
                cursor.close()
        return result
