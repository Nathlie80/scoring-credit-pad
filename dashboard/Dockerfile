FROM python:3.8

RUN mkdir /dashboard

ADD . /dashboard

RUN rm ./dashboard/sample.pkl

COPY requirements.txt /dashboard

WORKDIR /dashboard

RUN pip install -r requirements.txt
RUN pip install shap

COPY . /dashboard

EXPOSE 8501

CMD ["streamlit", "run", "app_dashboard.py"]