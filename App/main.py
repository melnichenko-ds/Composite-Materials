from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras

#Инициализация приложения
app = FastAPI(
    title="Прогнозирование соотношения матрица-наполнитель",
    description="ВКР: Data Science PRO, МГТУ им. Н.Э. Баумана",
    version="1.0.0"
)

#Загрузка модели и scaler
model = keras.models.load_model('nn_matrix_filler.keras')
scaler = joblib.load('scaler_minmax.pkl')
feature_order = joblib.load('feature_order.pkl')


#Схема входных данных
class CompositeInput(BaseModel):
    density: float = Field(..., description="Плотность, кг/м3")
    elastic_modulus: float = Field(..., description="Модуль упругости связующего, ГПа")
    hardener: float = Field(..., description="Количество отвердителя, м.%")
    epoxy_groups: float = Field(..., description="Содержание эпоксидных групп, %")
    flash_temp: float = Field(..., description="Температура вспышки, °C")
    surface_density: float = Field(..., description="Поверхностная плотность, г/м2")
    tensile_modulus: float = Field(..., description="Модуль упругости при растяжении, ГПа")
    tensile_strength: float = Field(..., description="Прочность при растяжении, МПа")
    resin_consumption: float = Field(..., description="Потребление смолы, г/м2")
    angle: float = Field(..., description="Угол нашивки, град")
    step: float = Field(..., description="Шаг нашивки")
    density_nup: float = Field(..., description="Плотность нашивки")

    class Config:
        json_schema_extra = {
            "example": {
                "density": 1975.0,
                "elastic_modulus": 739.0,
                "hardener": 110.0,
                "epoxy_groups": 22.2,
                "flash_temp": 285.0,
                "surface_density": 482.0,
                "tensile_modulus": 73.3,
                "tensile_strength": 2466.0,
                "resin_consumption": 220.0,
                "angle": 0.0,
                "step": 5.0,
                "density_nup": 57.0
            }
        }


#Схема выходных данных
class PredictionOutput(BaseModel):
    matrix_filler_ratio: float = Field(..., description="Рекомендованное соотношение матрица-наполнитель")


@app.get("/")
def root():
    return {
        "message": "API для прогнозирования соотношения матрица-наполнитель",
        "status": "OK",
        "docs": "/docs"
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(data: CompositeInput):
    try:
        # Собираем признаки в порядке из feature_order.pkl
        features = np.array([[
            data.density,
            data.elastic_modulus,
            data.hardener,
            data.epoxy_groups,
            data.flash_temp,
            data.surface_density,
            data.tensile_modulus,
            data.tensile_strength,
            data.resin_consumption,
            data.angle,
            data.step,
            data.density_nup
        ]])

        # Нормализуем ТОЛЬКО 9 признаков (индексы 0,1,2,3,4,5,8,10,11)
        scaled_indices = [0, 1, 2, 3, 4, 5, 8, 10, 11]
        features_scaled = features.copy().astype(float)
        features_scaled[:, scaled_indices] = scaler.transform(features[:, scaled_indices])

        # Предсказываем
        prediction = model.predict(features_scaled, verbose=0).flatten()[0]

        return PredictionOutput(matrix_filler_ratio=float(prediction))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка прогноза: {str(e)}")

from fastapi.responses import HTMLResponse

from fastapi.responses import HTMLResponse

@app.get("/app", response_class=HTMLResponse)
def app_page():
    return """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>Прогноз соотношения матрица-наполнитель</title>
        <style>
            * {
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            body {
                max-width: 700px;
                margin: 40px auto;
                padding: 30px;
                background: linear-gradient(135deg, #f9f9c5 0%, #e8f5b8 100%);
                color: #000000;
                font-size: 18px;
            }
            h1 {
                color: #000000;
                font-size: 28px;
                font-weight: bold;
                text-align: center;
                padding: 20px;
                background: linear-gradient(90deg, #d4e157 0%, #aed581 100%);
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                margin-bottom: 30px;
            }
            label {
                display: block;
                margin-top: 15px;
                font-weight: bold;
                font-size: 18px;
                color: #000000;
            }
            input {
                width: 100%;
                padding: 12px;
                margin-top: 6px;
                box-sizing: border-box;
                border: 2px solid #aed581;
                border-radius: 8px;
                font-size: 18px;
                color: #000000;
                background: #ffffff;
                transition: border-color 0.3s, box-shadow 0.3s;
            }
            input:focus {
                outline: none;
                border-color: #7cb342;
                box-shadow: 0 0 8px rgba(124, 179, 66, 0.5);
            }
            button {
                display: block;
                width: 100%;
                margin-top: 30px;
                padding: 16px;
                background: linear-gradient(90deg, #9ccc65 0%, #7cb342 100%);
                color: #000000;
                border: none;
                cursor: pointer;
                font-size: 20px;
                font-weight: bold;
                border-radius: 10px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
                transition: transform 0.2s, box-shadow 0.2s;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 14px rgba(0, 0, 0, 0.25);
            }
            button:active {
                transform: translateY(0);
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
            }
            #result {
                display: none;
                margin-top: 30px;
                padding: 25px;
                background: #ffffff;
                border-left: 8px solid #7cb342;
                border-radius: 10px;
                font-size: 24px;
                font-weight: bold;
                color: #000000;
                text-align: center;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            }
            #result-value {
                display: block;
                font-size: 36px;
                color: #33691e;
                margin-top: 10px;
            }
        </style>
    </head>
    <body>
        <h1>Прогноз соотношения матрица-наполнитель</h1>

        <label>Плотность, кг/м³: <input id="density" value="1975"></label>
        <label>Модуль упругости связующего, ГПа: <input id="elastic_modulus" value="739"></label>
        <label>Количество отвердителя, м.%: <input id="hardener" value="110"></label>
        <label>Содержание эпоксидных групп, %: <input id="epoxy_groups" value="22.2"></label>
        <label>Температура вспышки, °C: <input id="flash_temp" value="285"></label>
        <label>Поверхностная плотность, г/м²: <input id="surface_density" value="482"></label>
        <label>Модуль упругости при растяжении, ГПа: <input id="tensile_modulus" value="73.3"></label>
        <label>Прочность при растяжении, МПа: <input id="tensile_strength" value="2466"></label>
        <label>Потребление смолы, г/м²: <input id="resin_consumption" value="220"></label>
        <label>Угол нашивки, град: <input id="angle" value="0"></label>
        <label>Шаг нашивки: <input id="step" value="5"></label>
        <label>Плотность нашивки: <input id="density_nup" value="57"></label>

        <button onclick="predict()">Рассчитать</button>

        <div id="result">
            Рекомендованное соотношение:
            <span id="result-value"></span>
        </div>

        <script>
        async function predict() {
            const data = {
                density: parseFloat(document.getElementById('density').value),
                elastic_modulus: parseFloat(document.getElementById('elastic_modulus').value),
                hardener: parseFloat(document.getElementById('hardener').value),
                epoxy_groups: parseFloat(document.getElementById('epoxy_groups').value),
                flash_temp: parseFloat(document.getElementById('flash_temp').value),
                surface_density: parseFloat(document.getElementById('surface_density').value),
                tensile_modulus: parseFloat(document.getElementById('tensile_modulus').value),
                tensile_strength: parseFloat(document.getElementById('tensile_strength').value),
                resin_consumption: parseFloat(document.getElementById('resin_consumption').value),
                angle: parseFloat(document.getElementById('angle').value),
                step: parseFloat(document.getElementById('step').value),
                density_nup: parseFloat(document.getElementById('density_nup').value)
            };
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await response.json();
            document.getElementById('result').style.display = 'block';
            document.getElementById('result-value').textContent = result.matrix_filler_ratio.toFixed(4);
        }
        </script>
    </body>
    </html>
    """
