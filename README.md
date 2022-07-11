# scoring-credit-pad

Implementation of a credit scoring tool (API + Dashboard)

Files : 

	P7_notebookAnalyseExploratoire: 
		Contains all the pre-analyses, analyses, pre-processing and processing of the data before modeling. 
		Contains the different models tested, the calculation of the score and the first analysis of the global and individual data interpretability.
		Contains the creation of data processing functions to automate and facilitate the use of the model with new data.

	function:	contains only all the functions to automate the data processing

  	data_pretraitement: Contains all the data preparation work for its use by the API and the Dashboard 
		Using the functions in the function file. 
		The input data to be filled in are
			the path to the folder containing the 7 files + the file of column descriptions 
			the path to the file lgbm_model.joblib.

	docker-compose: specifies the commands that are executed by the app (with Docker)

Folders : 

	fastapi-backend

		Files:

		app.py: contains the code to give: the probility for the client to be classified in risky or not, the classification prediction.

		lgbm_model.joblib: contains the .best_estimator_ to be applied in the app

		requirements: contains all library needed in the app code

		sample_norm: sample with 10% of the data prepared to be applied in the model

		Procfile: specifies the commands that are executed by the app (with Heroku)
		
		Dockerfile: specifies the commands that are executed by the app (with Docker)


	streamlit-frontend

		Files:

		app_dashboard.py: contains the code to request in the API: the probility for the client to be classified in risky or not, the classification prediction. contains the code to explain these predictions for each client

		explainer.bz2: contains the the shap explainer to explain variable influences

		shap_values: contains the array of variable influences by client

		requirements: contains all library needed in the app code

		sample: sample with 10% of the data before applying normalisation

		sample_norm: sample normalized to be applied in the model

		logo-pad: the logo of the project

		col_shap_most_importance_dic: dictionnary with client and his 20 most influent variables

		columns_descrition_dic: dictionnary with variables and there descriptions

		Dockerfile: specifies the commands that are executed by the app (with Docker)