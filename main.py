from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uuid

app = FastAPI()

# Загружаем модель
MODEL_NAME = "mosaicml/mpt-7b-instruct"  # пример
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

# Чаты
chats = {}

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<html>
<body>
<h2>bebraAI Local</h2>
<textarea id="prompt" rows="5" cols="50"></textarea><br>
<button onclick="send()">Сгенерировать</button>
<pre id="result"></pre>
<script>
async function send(){
  const prompt = document.getElementById("prompt").value;
  const res = await fetch("/generate",{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({prompt})
  });
  const data = await res.json();
  document.getElementById("result").innerText = data.result;
}
</script>
</body>
</html>
"""

@app.post("/generate")
async def generate(req: Request):
    data = await req.json()
    prompt = data.get("prompt", "")

    # Токенизация и генерация
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=200)
    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return JSONResponse({"result": result})
