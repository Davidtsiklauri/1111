FROM python:3.9.6

WORKDIR /src
# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY . .
RUN ls
# command to run on container start
CMD  python ./house-model.py && uvicorn server:app --reload --port=3000 --host=0.0.0.0