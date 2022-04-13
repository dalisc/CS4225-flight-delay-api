#app.py
import numpy as np
from flask import Flask, request, jsonify, render_template
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler

app = Flask(__name__)

spark = SparkSession.builder.appName("FlightDelayPrediction").getOrCreate()
sc = spark.sparkContext
    # .config("spark.some.config.option", "some-value") \
    
model = RandomForestClassificationModel.load("/model")

@app.route('/')
def home():
    return "<p>Hello!</p>"

@app.route('/predict/<value_list>', methods=['GET'])
def predict(value_list):
    '''
    For rendering results on HTML GUI
    '''
    arr_value = []
    values = value_list.split(',')
    for value in values:
        arr_value.append(float(value))
    inputCols =  ['WeatherScore', 'Reporting_Airline_Index', 'Origin_Index', 'Dest_Index', 'MFR_Index', "sum_passengers_airport_month"]
    assembler = VectorAssembler(inputCols=inputCols, outputCol="features")
    df = spark.createDataFrame([tuple(arr_value)], inputCols)
    featuresMLData = assembler.transform(df)
    prediction_result = model.predict(featuresMLData.head().features)
    print(prediction_result)
    response = jsonify(
        value_input=value_list,
        result=prediction_result
    )
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')