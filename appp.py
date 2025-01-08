import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('PassiveAggressiveRegressor_model_sc.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('insta.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extracting input values from the form
        Likes = float(request.form['Likes'])
        Saves = float(request.form['Saves'])
        Comments = float(request.form['Comments'])
        Shares = float(request.form['Shares'])
        Profile_Visits = float(request.form.get('Profile_Visits', 0.0))
        Follows = float(request.form['Follows'])

        # Creating a DataFrame with the input values
        final_features = pd.DataFrame([[Likes, Saves, Comments, Shares, Profile_Visits, Follows]])

        # Making predictions using the model
        prediction = model.predict(final_features)
        output = prediction[0]

        # Returning the result to the HTML template
        return render_template('insta.html',REACH='INSTAGRAM REACH IS {}'.format(output))

    except Exception as e:
        # Handle any potential errors and display a message
        error_message = f"An error occurred: {str(e)}"
        return render_template('insta.html', prediction_text=error_message)

if __name__ == "__main__":
    app.run(debug=True)
