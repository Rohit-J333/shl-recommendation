from pydantic import BaseModel, field_validator
from typing import Optional


class RecommendRequest(BaseModel):
    query: Optional[str] = None
    jd_text: Optional[str] = None
    jd_url: Optional[str] = None

    @field_validator("query", "jd_text", "jd_url", mode="before")
    @classmethod
    def strip_whitespace(cls, v):
        if isinstance(v, str):
            return v.strip() or None
        return v

    def get_input_text(self) -> Optional[str]:
        return self.query or self.jd_text or self.jd_url


class AssessmentRecommendation(BaseModel):
    assessment_name: str
    assessment_url: str
    score: float = 0.0
    test_type: str
    duration_minutes: Optional[int] = None
    remote_testing: bool = False
    adaptive_irt: bool = False
    explanation: str = ""


class RecommendResponse(BaseModel):
    recommendations: list[AssessmentRecommendation]


class HealthResponse(BaseModel):
    status: str = "ok"
