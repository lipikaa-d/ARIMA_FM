from flask import Flask, render_template, request
from app.utils import (
    load_trained_model,
    forecast_next_steps,
    forecast_from_manual_input,
    get_latest_input_data
)
from src.evaluation import get_evaluation_metrics
from src.data_preprocessing import load_and_clean_data
from src.model_training import train_and_save_model

import os
from werkzeug.utils import secure_filename

app = Flask(
    __name__,
    template_folder='app/templates',
    static_folder='app/static'
)
app.secret_key = 'your-secret-key'

@app.route('/', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def index():
    model = load_trained_model()
    latest_input = get_latest_input_data()

    if request.method == 'POST':
        try:
            steps = int(request.form.get('steps', 1))
            input_type = request.form.get('input_type')

            if input_type == 'manual':
                manual_values = {
                    'P_IN': float(request.form.get('P_IN')),
                    'T_IN': float(request.form.get('T_IN')),
                    'P_OUT': float(request.form.get('P_OUT')),
                    'T_OUT': float(request.form.get('T_OUT')),
                    'LOAD': float(request.form.get('LOAD'))
                }

                forecast_values = forecast_from_manual_input(model, manual_values, steps)

                return render_template(
                    'forecast_result.html',
                    forecast_values=forecast_values,
                    steps=steps,
                    latest_inputs=manual_values,
                    manual_used=True
                )

            else:  # Use latest dataset row
                forecast_values = forecast_next_steps(model, steps)

                return render_template(
                    'forecast_result.html',
                    forecast_values=forecast_values,
                    steps=steps,
                    latest_inputs=latest_input,
                    manual_used=False
                )

        except Exception as e:
            return f"<h3>Error during forecasting: {e}</h3><br><a href='/'>Go back</a>"

    return render_template('index.html', latest_inputs=latest_input)



@app.route('/metrics')
def metrics():
    try:
        metrics = get_evaluation_metrics()

        return render_template(
            'metrics.html',
            r2=None,  # Not used for ARIMA
            rmse=metrics['rmse'],
            mae=metrics['mae'],
            y_test=metrics['y_test'],
            y_pred=metrics['y_pred']
        )
    except Exception as e:
        return f"<h3>Error loading evaluation metrics: {e}</h3><br><a href='/'>Go back</a>"


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    message = None
    if request.method == 'POST':
        file = request.files.get('dataset')
        if file:
            filename = secure_filename(file.filename)
            save_path = os.path.join('data', filename)
            file.save(save_path)
            message = f"Dataset uploaded successfully as '{filename}'"
        else:
            message = "No file selected."

    return render_template('upload.html', message=message)


@app.route('/train', methods=['GET', 'POST'])
def train():
    try:
        model, rmse, mae, mape, smape = train_and_save_model('data/combinedddddd_dataset.xlsx')
        return render_template(
            'train_result.html',
            rmse=rmse,
            mae=mae,
            mape=mape,
            smape=smape
        )
    except Exception as e:
        return f"<h3>Error during model training: {e}</h3><br><a href='/'>Go back</a>"



if __name__ == '__main__':
    app.run(debug=True)
