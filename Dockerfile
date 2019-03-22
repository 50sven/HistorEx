FROM python:3

# copy
COPY . /app

# install requirements
RUN pip install -r requirements.txt

WORKDIR /app

EXPOSE 1880

CMD ["python", ""./dashboard/run_app.py"]