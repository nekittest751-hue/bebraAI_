from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
import asyncio
import uuid
import torch

app = FastAPI()

# Доступные модели
MODEL_NAMES = {
    "distilgpt2": "distilgpt2",
    "gpt2": "gpt2"
}

# Загружаем модель по умолчанию
DEFAULT_MODEL = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[DEFAULT_MODEL])
model = AutoModelForCausalLM.from_pretrained(MODEL_NAMES[DEFAULT_MODEL])

# История чатов
history = []

# Очередь генераций
queue = asyncio.Queue()
MAX_CONCURRENT = 1  # один запрос одновременно

async def worker():
    while True:
        task = await queue.get()
        chat_id, prompt, chosen_model, fut = task
        try:
            # Переключаем модель, если нужно
            global tokenizer, model
            if chosen_model != DEFAULT_MODEL:
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[chosen_model])
                model = AutoModelForCausalLM.from_pretrained(MODEL_NAMES[chosen_model])

            inputs = tokenizer(prompt, return_tensors="pt")
            output_ids = model.generate(**inputs, max_new_tokens=50)
            result = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Сохраняем в историю
            history.append({"model": chosen_model, "prompt": prompt, "result": result})

            fut.set_result(result)
        except Exception as e:
            fut.set_result(f"Ошибка генерации: {e}")
        finally:
            queue.task_done()

asyncio.create_task(worker())

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<html>
<body>
<h2>bebraAI Local (с историей и выбором модели)</h2>

<select id="model">
    <option value="distilgpt2">distilgpt2 (легкая)</option>
    <option value="gpt2">gpt2 (полная)</option>
</select><br><br>

<textarea id="prompt" rows="5" cols="50" placeholder="Введите запрос..."></textarea><br>
<button onclick="send()">Сгенерировать</button>

<pre id="status"></pre>
<pre id="result"></pre>
<pre id="history"></pre>

<script>
async function send(){
  const prompt = document.getElementById("prompt").value;
  const model = document.getElementById("model").value;
  const statusDiv = document.getElementById("status");
  statusDiv.innerText = "🤔 Вы в очереди...";

  const res = await fetch("/generate", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({prompt, model})
  });

  const data = await res.json();
  statusDiv.innerText = "✅ Готово!";
  document.getElementById("result").innerText = data.result;

  // Показываем историю
  let histText = "История:\n";
  data.history.forEach((item, idx)=>{
      histText += `${idx+1}. [${item.model}] ${item.prompt} → ${item.result}\n`;
  });
  document.getElementById("history").innerText = histText;
}
</script>
</body>
</html>
"""

@app.post("/generate")
async def generate(req: Request):
    data = await req.json()
    prompt = data.get("prompt", "")
    chosen_model = data.get("model", DEFAULT_MODEL)
    chat_id = str(uuid.uuid4())

    fut = asyncio.get_event_loop().create_future()
    await queue.put((chat_id, prompt, chosen_model, fut))
    
    result = await fut
    return JSONResponse({"result": result, "queue_position": queue.qsize(), "history": history})
