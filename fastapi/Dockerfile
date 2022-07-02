FROM python3.8

RUN mkdir /fastapi


COPY requirements.txt /fastapi

WORKDIR /fastapi

RUN pip install -r requirements.txt
RUN pip install shap

COPY . /fastapi

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]