FROM python:3.9-slim as builder

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
 
COPY . /app 

EXPOSE 8000

# Default command to keep the container running
#CMD ["tail", "-f", "/dev/null"]

#CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "api:app"]
CMD ["python", "api.py"]