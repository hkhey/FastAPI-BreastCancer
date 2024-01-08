# FastAPI-BreastCancer
## HTML Template
The HTML template, located in the templates folder, is designed for predicting cancer diagnoses. It incorporates a form with input fields for relevant features and displays the prediction result.

## Jupyter Notebook
The Jupyter Notebook handles data loading, cleaning, and model training for predicting cancer diagnoses. It utilizes various machine learning algorithms, including Logistic Regression, SVM, Decision Trees, and KNN. The best-performing model (SVM) is saved as model.pkl.

## FastAPI Application
The FastAPI application serves as the backend for the web interface. It loads the pre-trained SVM model (model.pkl) and provides an endpoint for making predictions.

## Running the Application
To run the FastAPI application, use the command uvicorn app:app --reload. Access the application at http://localhost:8080 in your web browser.
