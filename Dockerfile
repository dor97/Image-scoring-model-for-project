FROM python:3.9

# 
WORKDIR /app

# 
COPY api.py .
COPY requirements.txt .

# 
# 
#RUN pip install flask==2.1.3
#RUN pip install requests
#RUN pip install pymongo
RUN pip install -r requirements.txt

# 
COPY . /app 

EXPOSE 8000

# 
CMD ["python", "api.py"]