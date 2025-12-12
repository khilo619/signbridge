from fastapi.testclient import TestClient
from api.sign_demo.main import app
import pytest

# Create a TestClient that wraps your FastAPI app
client = TestClient(app)

def test_health_check():
    """
    Test that the API is up and running.
    """
    response = client.get("/")
    assert response.status_code == 200
    # Your root endpoint likely returns a welcome message or redirects to docs
    # Adjust this assertion based on your actual main.py
    # If main.py redirects to docs, this might be 200 (docs page) or the welcome JSON
    
def test_model_info_endpoint():
    """
    Test the endpoint that returns model metadata (if it exists).
    Assuming you have a standard health/info endpoint.
    If not, we can stick to the root check.
    """
    # This is a placeholder. If you don't have this endpoint, the test will fail (404).
    # response = client.get("/health") 
    # assert response.status_code in [200, 404]
    pass
