FROM python:3.8.2

WORKDIR /usr/app

ENV API_URL=http://localhost:4000

EXPOSE 4000

COPY requirements.txt ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "server.py" ]

#1. Build the image:
#   $ docker image build [OPTIONS] PATH
#Ex:
#   $ docker build -t name-of-docker-image .
#2. Run the container:
#   $ docker container run [OPTIONS] IMAGE
#Ex:
#   $ docker run -it -p 4000:4000 name-of-docker-image
#3. Acess:
#   http://localhost:4000