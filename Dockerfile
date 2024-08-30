FROM python:3.10


WORKDIR /app/WebPredict

COPY requirements.txt /app/WebPredict

COPY . /app/WebPredict

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python", "app.py"]
