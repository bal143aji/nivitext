from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import os
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Emotion API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
MODEL_PATH = 'models/baseline_model.pkl'
TFIDF_PATH = 'models/tfidf_vectorizer.pkl'

if os.path.exists(MODEL_PATH) and os.path.exists(TFIDF_PATH):
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
else:
    model = None
    tfidf = None

label_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: TextRequest):
    if model is None or tfidf is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        # Prediction
        text_vec = tfidf.transform([request.text])
        prediction = int(model.predict(text_vec)[0])
        probs = model.predict_proba(text_vec)[0].tolist()
        
        response = {
            "emotion": label_map[prediction],
            "confidence": probs[prediction],
            "probabilities": {label_map[i]: prob for i, prob in enumerate(probs)}
        }
        return response
    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files from 'web' directory
if os.path.exists('web'):
    app.mount("/", StaticFiles(directory="web", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
