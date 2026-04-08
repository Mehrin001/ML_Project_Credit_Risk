from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask.templating import render_template
import numpy as np

app = Flask(__name__, template_folder='templates')

# Load trained model
with open('lending_club_pipeline.pkl', 'rb') as f:
    artifacts = pickle.load(f)

clf = artifacts['classifier']
amt_reg = artifacts['regressor']
scaler = artifacts['scaler']

# print('Number of features: ', model.n_features_in_)

state_mappimg = artifacts['state_mapping']
default_state = artifacts['default_state']  



def get_state(state):
    state = state.upper().strip()
    return state_mappimg.get(state, default_state)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Extract inputs
        amount = float(data.get("loan_amnt", 0))
        risk_score = float(data.get("fico", 0))
        dti = float(data.get("dti", 0))
        state_raw = data.get("state", "Other")
        state = int(get_state(state_raw))
        emp_length = float(data.get("emp_length", 0))
        
        input_data = pd.DataFrame([[amount, risk_score, dti, state, emp_length]],
                                  columns=['amount', 'risk_score', 'dti', 'state', 'emp_length'])

        input_scaled = scaler.transform(input_data)

        # Predict loan status
        pred_status = int(clf.predict(input_scaled)[0])
        print("pred_status: ", pred_status)
        prob = float(clf.predict_proba(input_scaled)[0][1])
        print("probability of default: ", prob)
        # Predict amount only if approved
        
        reg_input = pd.DataFrame([[risk_score, dti, state, emp_length]],
                                     columns=['risk_score', 'dti', 'state', 'emp_length'])

        pred_amount = float(amt_reg.predict(reg_input)[0])
        
        # Risk Label based on probability
        
        # Loan Product Recommendation
        
        return jsonify({
            "prediction": pred_status,
            "probability": round(prob, 3),
            "recommended_amount": round(pred_amount, 2)
        })
      
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)