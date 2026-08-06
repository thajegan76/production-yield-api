from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd

app = FastAPI()

# Load the trained model
model = joblib.load("production_yield_model.pkl")

# HTML templates
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": None
        }
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    Solder_Temperature: float = Form(...),
    Conveyor_Speed: float = Form(...),
    Humidity: float = Form(...),
    Machine_Age: int = Form(...),
    Operator_Experience: int = Form(...),
    Material_Quality: float = Form(...),
    Component_Density: int = Form(...),
    Inspection_Time: float = Form(...),
    Rework_Count: int = Form(...),
    Shift: int = Form(...)
):

    new_batch = pd.DataFrame({
        "Solder_Temperature": [Solder_Temperature],
        "Conveyor_Speed": [Conveyor_Speed],
        "Humidity": [Humidity],
        "Machine_Age": [Machine_Age],
        "Operator_Experience": [Operator_Experience],
        "Material_Quality": [Material_Quality],
        "Component_Density": [Component_Density],
        "Inspection_Time": [Inspection_Time],
        "Rework_Count": [Rework_Count],
        "Shift": [Shift]
    })

    prediction = float(model.predict(new_batch)[0])

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": round(prediction,2)
        }
    )