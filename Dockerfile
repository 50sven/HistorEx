FROM python:3

WORKDIR /app

# copy current directory to /app
COPY /dashboard/assets/ /app/assets/
COPY . /app

# install requirements
RUN pip install -r requirements.txt

EXPOSE 1880

CMD ["python", "./dashboard/run_app.py"]