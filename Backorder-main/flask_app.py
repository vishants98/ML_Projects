from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
#import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in = open("model2.pkl", "rb")
classifier = pickle.load(pickle_in)


@app.route('/')
def welcome():
    return "Welcome!!!"


@app.route('/predict', methods=["Get"])
def predict_back_order():
    
    """Let's predict Back order or not.
    
    ---
    parameters:
        
        - name: pieces_past_due
          in: query
          type: number
          required: true
        - name: sales_3_months
          in: query
          type: number
          required: true
        - name: national_inv
          in: query
          type: number
          required: true
        - name: forecast_6_months
          in: query
          type: number
          required: true
        - name: in_transit_quantity
          in: query
          type: number
          required: true
        - name: deck_risk
          in: query
          type: number
          required: false
        - name: lead_time
          in: query
          type: number
          required: false
        - name: forecast_3_month
          in: query
          type: number
          required: true
        - name: forecast_9_month
          in: query
          type: number
          required: true
        - name: sales_1_month
          in: query
          type: number
          required: true
        - name: sales_6_month
          in: query
          type: number
          required: false
        - name: sales_9_month
          in: query
          type: number
          required: true
        - name: min_bank
          in: query
          type: number
          required: false
        - name: perf_6_month_avg
          in: query
          type: number
          required: false
        - name: perf_12_month_avg
          in: query
          type: number
          required: false
        - name: local_bo_qty
          in: query
          type: number
          required: false
        - name: ppap_risk
          in: query
          type: number
          required: false
        - name: stop_auto_buy
          in: query
          type: number
          required: false
    responses:
        200:
            
            description: The output values

    """
    pieces_past_due = request.args.get("pieces_past_due")
    sales_3_months = request.args.get("sales_3_months")
    national_inv = request.args.get("national_inv")
    forecast_6_months = request.args.get("forecast_6_months")
    in_transit_quantity = request.args.get("in_transit_quantity")
    deck_risk = request.args.get("deck_risk")
    lead_time = request.args.get("lead_time")
    forecast_3_month = request.args.get("forecast_3_month")
    forecast_9_month = request.args.get("forecast_9_month")
    sales_1_month = request.args.get("sales_1_month")
    sales_6_month = request.args.get("sales_6_month")
    sales_9_month = request.args.get("sales_9_month")
    min_bank = request.args.get("min_bank")
    perf_6_month_avg = request.args.get("perf_6_month_avg")
    perf_12_month_avg = request.args.get("perf_12_month_avg")
    local_bo_qty = request.args.get("local_bo_qty")
    ppap_risk = request.args.get("ppap_risk")
    stop_auto_buy = request.args.get("stop_auto_buy")
    prediction = classifier.predict(np.array([[national_inv, lead_time, in_transit_quantity, forecast_3_month,
       forecast_6_months, forecast_9_month, sales_1_month,
       sales_3_months, sales_6_month, sales_9_month, min_bank,
       pieces_past_due, perf_6_month_avg, perf_12_month_avg,
       local_bo_qty, deck_risk, ppap_risk, stop_auto_buy]]))
    print(prediction)
    if (prediction[0]) == 1:
        return "Went on back order"
    else:
        return "Didn't go on back order"
            
    #return "went on back order = " + str(prediction[0])


'''@app.route('/predict_order', methods=["POST"])
def predict_back_order():
    """Let's predict Back order or not.
    
    ---
    parameters:
        
        - name: file
          in: formData
          type: file
          required: true

    responses:
        200:
            description: The output values

    """
    df_test = pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction = classifier.predict(df_test)

    return str(list(prediction))
'''

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
