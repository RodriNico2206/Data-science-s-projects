# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib, numpy as np, uvicorn


# class Prediction(BaseModel):
#     feature1: float
#     feature2: float
#     feature3: float

# app = FastAPI()

# model = joblib.load("model.pkl")


# @app.post('/predict')

# def predict(request: Prediction):
#     features = np.array([[request.feature1, request.feature2, request.feature3]])

#     prediction = model.predict(features)
    
#     prediction = prediction[0]

#     return {"prediction": prediction}


# if __name__ == "__main__":
#     uvicorn.run('app:app', host="localhost", port=8080, reload= True)

# execute file

# open file app.py in terminal and then it must be executed with command
# py app.py



from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, numpy as np, uvicorn, os


# Define the input model with descriptions
class PredictionInput(BaseModel):
    feature1: float = 0.0
    feature2: float = 0.0
    feature3: float = 0.0
    
# Define the output model
class PredictionOutput(BaseModel):
    prediction: float
    probability: float = None
    input_features: dict
    model_info: dict

app = FastAPI(
    title="ML Prediction API",
    description="API for making predictions using a machine learning model",
    version="1.0.0"
)

# Load model with error handling
try:
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = joblib.load(model_path)
    model_info = {
        "type": type(model).__name__,
        "features_expected": 3
    }
except Exception as e:
    model = None
    model_info = {"error": str(e)}
    print(f"Error loading model: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Prediction API", "status": "active"}

@app.post('/predict', response_model=PredictionOutput)
def predict(request: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available")
    
    try:
        # Extract features
        features = np.array([[request.feature1, request.feature2, request.feature3]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get probability if available (for classification models)
        probability = None
        try:
            if hasattr(model, 'predict_proba'):
                probability = float(np.max(model.predict_proba(features)[0]))
        except:
            pass
        
        # Return detailed response
        return {
            "prediction": prediction,
            "probability": probability,
            "input_features": {
                "feature1": request.feature1,
                "feature2": request.feature2,
                "feature3": request.feature3
            },
            "model_info": model_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }

if __name__ == "__main__":
    print("Starting API server at http://localhost:8080\n")
    print("Documentation available at http://localhost:8080/docs\n")
    uvicorn.run('app:app', host="localhost", port=8080, reload=True)