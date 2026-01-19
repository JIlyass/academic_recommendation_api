# ========== 1. IMPORTS ==========
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from typing import List


# ========== 2. INITIALISATION ==========
app = FastAPI(
    title="Academic Recommendation API",
    description="API pour recommandation de filières académiques",
    version="1.0.0"
)


# ========== 3. CONFIGURATION CORS ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://academicrec.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== 4. VARIABLES GLOBALES ==========
model = None
label_encoder = None


# ========== 5. SCHÉMAS PYDANTIC ==========
class PredictionInput(BaseModel):
    mathematiques: float
    physique: float
    chimie: float
    biologie: float
    informatique: float
    francais: float
    arabe: float
    economie: float
    histoire_geographie: float
    philosophie: float

class PredictionOutput(BaseModel):
    filiere_recommandee: str
    probability: float = None
    all_probabilities: dict = None


# ========== 6. CHARGEMENT DU MODÈLE AU DÉMARRAGE ==========
@app.on_event("startup")
async def load_model():
    global model, label_encoder
    
    print("=" * 50)
    print("Demarrage de l'application...")
    
    # Charger le modèle
    try:
        with open("models/random_forest_model.pkl", "rb") as f:
            model = pickle.load(f)
        print("OK - Modele charge avec succes")
    except Exception as e:
        print(f"ERREUR - Impossible de charger le modele: {e}")
        model = None
    
    # Charger le label encoder
    try:
        with open("models/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        print("OK - Label Encoder charge avec succes")
        print(f"Filieres disponibles: {len(label_encoder.classes_)}")
    except Exception as e:
        print(f"ERREUR - Impossible de charger le label encoder: {e}")
        label_encoder = None
    
    print("=" * 50)
# ========== 7. ROUTES ==========

@app.get("/")
async def root():
    return {
        "message": "Bienvenue sur l'API Academic Recommendation",
        "status": "running",
        "description": "API pour recommander des filières académiques basées sur les notes",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model/info",
            "filieres": "/filieres"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "label_encoder_loaded": label_encoder is not None
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    if model is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Modèle ou Label Encoder non chargé")
    
    try:
        # Créer un DataFrame avec les noms de colonnes exacts
        features_dict = {
            'Mathématiques': [input_data.mathematiques],
            'Physique': [input_data.physique],
            'Chimie': [input_data.chimie],
            'Biologie': [input_data.biologie],
            'Informatique': [input_data.informatique],
            'Français': [input_data.francais],
            'Arabe': [input_data.arabe],
            'Économie': [input_data.economie],
            'Histoire-Géographie': [input_data.histoire_geographie],
            'Philosophie': [input_data.philosophie]
        }
        
        X_input = pd.DataFrame(features_dict)
        
        # Faire la prédiction (retourne l'index encodé)
        prediction_encoded = model.predict(X_input)[0]
        
        # Décoder la prédiction pour obtenir le nom de la filière
        filiere_recommandee = label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Obtenir les probabilités si disponible
        probability = None
        all_probabilities = None
        
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X_input)[0]
            probability = float(max(probas))
            
            # Créer un dictionnaire avec toutes les filières et leurs probabilités
            all_probabilities = {
                label_encoder.inverse_transform([i])[0]: float(probas[i])
                for i in range(len(probas))
            }
        
        return {
            "filiere_recommandee": filiere_recommandee,
            "probability": probability,
            "all_probabilities": all_probabilities
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")


@app.get("/filieres")
async def get_filieres():
    """Retourne la liste de toutes les filières disponibles"""
    if label_encoder is None:
        raise HTTPException(status_code=503, detail="Label Encoder non chargé")
    
    try:
        filieres = label_encoder.classes_.tolist()
        return {
            "filieres": filieres,
            "count": len(filieres)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@app.get("/model/info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    features = ['Mathématiques', 'Physique', 'Chimie', 'Biologie',
                'Informatique', 'Français', 'Arabe', 'Économie',
                'Histoire-Géographie', 'Philosophie']
    
    return {
        "model_type": type(model).__name__,
        "n_features": len(features),
        "features": features,
        "n_classes": len(label_encoder.classes_) if label_encoder else "Unknown"
    }


# ========== 8. DÉMARRAGE (OPTIONNEL POUR LOCAL) ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)