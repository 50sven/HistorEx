FROM python:3

WORKDIR /app

# copy current directory to /app
COPY /dashboard/assets/ /app/assets/
COPY . /app

# install requirements
RUN pip install -r requirements.txt
# add user admin - does not start app as root user
RUN useradd -ms /bin/bash admin

USER admin

EXPOSE 1880

CMD ["python", "./dashboard/run_app.py"]