FROM python:3.12.1
WORKDIR /app
COPY  main.py .
RUN pip install -U scikit-learn
EXPOSE 5000 
CMD ["python","main.py"]
