from flask import Blueprint, render_template, request, redirect, flash
import os
from app.utils import process_forecast

main = Blueprint('main', __name__)

@main.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        steps = int(request.form.get("steps"))

        if file:
            filepath = os.path.join("data", file.filename)
            file.save(filepath)
            forecast_df, fig = process_forecast(filepath, steps)

            return render_template("forecast_result.html", table=forecast_df.to_html(classes='table'), plot=fig.to_html())

    return render_template("index.html")
