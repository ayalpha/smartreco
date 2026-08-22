"""AI-integrated Learning Paths."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Form, Request, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user_optional, render_page
from app.models.product import Product
from app.models.user import User
from app.agent.mesh_client import call_llm_json

logger = logging.getLogger(__name__)

router = APIRouter(tags=["paths"])
api_router = APIRouter(prefix="/api/paths", tags=["paths"])

@router.get("/paths", include_in_schema=False)
def paths_page(
    request: Request,
    user: Optional[User] = Depends(get_current_user_optional),
) -> Response:
    """Render the learning paths interface."""
    return render_page(request, "paths.html", user)

@router.post("/paths", include_in_schema=False)
def generate_path_form(
    request: Request,
    goal: str = Form(...),
    db: Session = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional),
) -> Response:
    """Handle form submission and generate a learning path."""
    products = list(db.scalars(select(Product).where(Product.is_active.is_(True))))
    
    catalog_context = "\n".join([f"ID: {p.id} | Title: {p.title} | Category: {p.category} | Level: {p.skill_level}" for p in products])
    
    prompt = [
        {"role": "system", "content": "You are a career counselor AI. Given a user's learning goal and a catalog of available courses, construct a logical, chronological learning path of 3 to 5 courses. Return a JSON object with 'title' (string), 'description' (string), 'courses' (list of integers matching the chosen course IDs), and 'rationale' (list of strings explaining why each course was chosen in order). DO NOT invent course IDs, only use the ones provided."},
        {"role": "user", "content": f"User Goal: {goal}\n\nAvailable Courses:\n{catalog_context}"}
    ]
    
    try:
        payload = call_llm_json(prompt, purpose="structured", max_tokens=1500)
        
        path_courses = []
        for cid in payload.get("courses", []):
            course = next((p for p in products if p.id == cid), None)
            if course:
                path_courses.append(course)
                
        payload["course_objects"] = path_courses
        
    except Exception as e:
        logger.error(f"Failed to generate path: {e}")
        payload = {"error": "Failed to generate path. Please try again or check your LLM configuration."}
        
    return render_page(request, "paths.html", user, goal=goal, path=payload)
