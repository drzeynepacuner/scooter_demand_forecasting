# Use a lightweight Python base
FROM python:3.9-slim

WORKDIR /app

# Copy requirements first, then install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code (including final_model.joblib)
COPY . .

# EXPOSE your Flask port (e.g., 5000)
EXPOSE 5000

# By default, run the Flask app (in app.py) with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
