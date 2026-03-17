from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
import openai
import os
import base64

app = FastAPI()

# Берём API ключ из переменной окружения
openai.api_key = os.environ.get("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", "")

# Конфиг моделей
MODELS = {
    "мини": "bebraAI-mini-1.0",
    "стандарт": "bebraAI-standard-1.0"
}

# История генераций
history = []

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
        <body>
            <h2>bebraAI Супер-Генератор</h2>
            <select id="model">
                <option value="мини">мини 1.0</option>
                <option value="стандарт">стандарт 1.0</option>
            </select><br><br>
            <textarea id="prompt" placeholder="Введите запрос..." rows="5" cols="50"></textarea><br><br>
            <button onclick="send()">Сгенерировать текст</button>
            <button onclick="generateImage()">Сгенерировать картинку</button>
            <pre id="result"></pre>
            <img id="image" style="max-width:400px; display:block; margin-top:10px;" />
            <script>
                async function send() {
                    const model = document.getElementById('model').value;
                    const prompt = document.getElementById('prompt').value;
                    const res = await fetch('/generate', {
                        method:'POST',
                        headers: {'Content-Type':'application/json'},
                        body: JSON.stringify({model,prompt})
                    });
                    const data = await res.json();
                    document.getElementById('result').innerText = data.result;
                    document.getElementById('image').style.display='none';
                }
                async function generateImage() {
                    const prompt = document.getElementById('prompt').value;
                    const res = await fetch('/generate_image', {
                        method:'POST',
                        headers: {'Content-Type':'application/json'},
                        body: JSON.stringify({prompt})
                    });
                    const data = await res.json();
                    document.getElementById('image').src = 'data:image/png;base64,' + data.image;
                    document.getElementById('image').style.display='block';
                }
            </script>
        </body>
    </html>
    """

@app.post("/generate")
async def generate(req: Request):
    data = await req.json()
    model = MODELS.get(data.get('model'), MODELS['мини'])
    prompt = data.get('prompt','')

    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        max_tokens=500
    )

    result = response.choices[0].message.content
    history.append({"model":model, "prompt":prompt, "result":result})

    return JSONResponse({"result":result, "history_len":len(history)})

@app.post("/generate_image")
async def generate_image(req: Request):
    data = await req.json()
    prompt = data.get('prompt','')

    image_resp = openai.Image.create(
        prompt=prompt,
        n=1,
        size="512x512"
    )

    image_base64 = image_resp.data[0].b64_json
    return JSONResponse({"image": image_base64})

@app.get("/history")
async def get_history():
    return JSONResponse(history)