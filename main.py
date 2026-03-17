from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
import asyncio
import uuid
import torch

app = FastAPI()

# Загружаем локальную GPT-2 (полностью без API)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Очередь генераций
queue = asyncio.Queue()
MAX_CONCURRENT = 2  # одновременно обрабатываем 2 запроса

async def worker():
    while True:
        task = await queue.get()
        chat_id, prompt, fut = task
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            output_ids = model.generate(**inputs, max_new_tokens=100)
            result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            fut.set_result(result)
        except Exception as e:
            fut.set_result(f"Ошибка генерации: {e}")
        finally:
            queue.task_done()

# Запускаем воркеры
for _ in range(MAX_CONCURRENT):
    asyncio.create_task(worker())

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<html>
<body>
<h2>bebraAI Local (без API, с очередью)</h2>
<textarea id="prompt" rows="5" cols="50" placeholder="Введите запрос..."></textarea><br>
<button onclick="send()">Сгенерировать</button>
<pre id="status"></pre>
<pre id="result"></pre>

<script>
async function send(){
  const prompt = document.getElementById("prompt").value;
  const statusDiv = document.getElementById("status");
  statusDiv.innerText = "🤔 Вы в очереди...";
  
  const res = await fetch("/generate", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({prompt})
  });
  
  const data = await res.json();
  statusDiv.innerText = "✅ Готово!";
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
    chat_id = str(uuid.uuid4())
    
    fut = asyncio.get_event_loop().create_future()
    await queue.put((chat_id, prompt, fut))
    
    position_in_queue = queue.qsize()  # можно показывать пользователю
    result = await fut
    return JSONResponse({"result": result, "queue_position": position_in_queue})
