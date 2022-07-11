FROM python:3.8

RUN mkdir /streamlit-frontend

ADD . /streamlit-frontend

WORKDIR /streamlit-frontend

RUN pip install -r requirements.txt
RUN pip install shap


EXPOSE 8501

CMD ["streamlit", "run", "app_dashboard.py"]