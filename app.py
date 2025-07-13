from flask import Flask, render_template, request, send_file
import os
import io
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib
import matplotlib.pyplot as plt

from app.utils import load_trained_model, forecast_next_steps, forecast_from_manual_input, get_latest_input_data
from src.data_preprocessing import load_and_clean_data
from src.evaluation import get_evaluation_metrics

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

@app.route('/', methods=['GET', 'POST'])
def index():
    model = load_trained_model()
    latest_inputs = get_latest_input_data()

    if request.method == 'POST':
        steps = int(request.form.get('steps', 1))
        use_manual = request.form.get('input_type') == 'manual'

        if use_manual:
            try:
                manual_data = {
                    'P_IN': float(request.form.get('P_IN')),
                    'T_IN': float(request.form.get('T_IN')),
                    'P_OUT': float(request.form.get('P_OUT')),
                    'T_OUT': float(request.form.get('T_OUT')),
                }

                # Add dummy LOAD values (needed for lag features)
                dummy_loads = list(range(13, 0, -1))  # or [100.0] * 13
                manual_df = pd.DataFrame([
                    {**manual_data, 'LOAD': val} for val in dummy_loads
                ])

                forecast_values = forecast_from_manual_input(model, manual_df, steps)

                return render_template(
                    'forecast_result.html',
                    forecast_values=forecast_values,
                    steps=steps,
                    latest_inputs=manual_data,
                    manual_used=True
                )

            except Exception as e:
                return f"<h3>Error in manual input: {e}</h3><br><a href='/'>Go back</a>"



        else:

            try:

                print("Using latest dataset row for forecasting...")
                forecast_values = forecast_next_steps(model, steps)
                print("Forecast values:", forecast_values)

                return render_template(

                    'forecast_result.html',

                    forecast_values=forecast_values,

                    steps=steps,

                    latest_inputs=latest_inputs,

                    manual_used=False

                )

            except Exception as e:

                print("Forecasting with latest data failed:", e)  # <--- Add this line

                return f"<h3>Error: {e}</h3><br><a href='/'>Go back</a>"
    return render_template('index.html', latest_inputs=latest_inputs)


@app.route('/metrics')
def metrics():
    try:
        metrics = get_evaluation_metrics()
        return render_template(
            'metrics.html',
            rmse=metrics['rmse'],
            mae=metrics['mae'],
            mape=metrics['mape'],
            smape=metrics['smape'],
            y_test=metrics['y_test'],
            y_pred=metrics['y_pred']
        )
    except Exception as e:
        return f"<h3>Error loading evaluation metrics: {e}</h3><br><a href='/'>Go back</a>"


@app.route('/load_plot.png')
def load_plot():
    try:
        df = load_and_clean_data('data/combinedddddd_dataset.xlsx')

        # Clean and convert date column
        df['DATE'] = df['DATE'].astype(str).str.strip()
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        df = df.dropna(subset=['DATE', 'LOAD'])
        df = df[df['LOAD'] > 0].tail(500)

        plt.figure(figsize=(12, 4))
        plt.plot(df['DATE'], df['LOAD'], color='blue')
        plt.title('Time vs Load')
        plt.xlabel('Time')
        plt.ylabel('Load (kW)')
        plt.xticks(rotation=45)
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        return send_file(img, mimetype='image/png')

    except Exception as e:
        return f"<h3>Error generating plot: {e}</h3><br><a href='/'>Go back</a>"
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    message = None
    if request.method == 'POST':
        file = request.files.get('dataset')
        if file:
            filename = secure_filename(file.filename)
            save_path = os.path.join('data', filename)  # Save to /data folder
            file.save(save_path)
            message = f"Dataset uploaded successfully as '{filename}'"
        else:
            message = "No file selected."

    return render_template('upload.html', message=message)


from src.model_training import train_and_save_model

@app.route('/train')
def train():
    try:
        model, rmse, mae, mape, smape = train_and_save_model('data/combinedddddd_dataset.xlsx')
        message = f"New model trained successfully! RMSE: {rmse}, MAE: {mae}"
        return render_template('train_success.html', message=message)
    except Exception as e:
        return f"<h3>Error training model: {e}</h3><br><a href='/'>Go back</a>"

"""
if __name__ == '__main__':
    app.run(debug=True)
"""
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
