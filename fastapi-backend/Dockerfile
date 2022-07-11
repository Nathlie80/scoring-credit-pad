FROM python:3.8

RUN mkdir /fastapi-backend

ADD . /fastapi-backend

WORKDIR /fastapi-backend

RUN pip install -r requirements.txt
RUN pip install shap

COPY . /fastapi-backend

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000" , "--reload"]