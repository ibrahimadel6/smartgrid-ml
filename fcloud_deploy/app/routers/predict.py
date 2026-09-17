from fastapi import APIRouter, HTTPException

from ..schemas import PredictRequest, PredictResponse
from ..service import service
from ..validation import clean_features, validate_features

router = APIRouter(prefix="/api/predict", tags=["predict"])


@router.post("", response_model=PredictResponse)
def predict(req: PredictRequest):
    proj = service.project()
    if req.model not in proj.models:
        raise HTTPException(404, f"Unknown model '{req.model}'")

    # "model" is a routing control key, never a feature.
    features = clean_features(req.features)

    errors = validate_features(features)
    if errors:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Invalid input for model '{req.model}'.",
                "errors": errors,
            },
        )

    result = service.predict(req.model, features)
    return PredictResponse(**result)
