FROM python:3.12.9

COPY requirements.txt requirements.txt
COPY package_folder package_folder
COPY setup.py setup.py
COPY models models

RUN pip install --upgrade pip
RUN pip install -e .


#Run cotainer locally
CMD uvicorn package_folder.vibe_api:app --reload --host 0.0.0.0

#Run container deployed -> GCP
# CMD uvicorn package_folder.api_file:app --reload --host 0.0.0.0 --port $PORT


#Then:
# docker build --tag=$IMAGE:dev .
