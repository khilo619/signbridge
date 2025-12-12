"""Shared Pydantic schemas for sign prediction APIs."""

from typing import List

from pydantic import BaseModel


class TopKPrediction(BaseModel):
    """A single prediction in the top-k list."""
    gloss: str
    label: int
    probability: float


class SignPredictionResponse(BaseModel):
    """Response model for sign prediction endpoints."""
    gloss: str
    label: int
    probability: float
    topk: List[TopKPrediction]
    num_frames_used: int
    processing_ms: float
