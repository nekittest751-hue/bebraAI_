from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import asyncio
import uuid
import requests

# bebraAI PRO - lightweight local system (no API keys required)
app = FastAPI()

# memory per user (in-memory, persists while process runs)
memory = {}

# async queue for generation tasks
queue = asyncio.Queue()

# lazy-loaded tokenizer/model to avoid OOM at startup
tokenizer = None
model = None
MODEL_NAME = "sshleifer/tiny-gpt2"  # ultra-lite model suitable for small hosts

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# local knowledge base loader
def load_knowledge():
    try:
        with open("knowledge.txt", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

# very simple web search using DuckDuckGo HTML (small, best-effort)
def web_search(query):
    try:
        r = requests.get(f"https://duckduckgo.com/html/?q={query}", timeout=3)
        return r.text[:800]
    except Exception:
        return ""

async def worker():
    while True:
        user_id, prompt, fut = await queue.get()
        try:
            load_model()

            knowledge = load_knowledge()
            user_memory = memory.get(user_id, "")
            web = web_search(prompt)

            # system personality + RAG-style prompt
            full_prompt = f"""
Ты bebraAI PRO.
Ты независимая нейросеть и не являешься ChatGPT.

Контекст:
{knowledge}

Интернет (срез):
{web}

История:
{user_memory}

Пользователь: {prompt}
Ответ:
"""

            inputs = tokenizer(full_prompt, return_tensors="pt")
            output = model.generate(**inputs, max_new_tokens=40)
            result = tokenizer.decode(output[0], skip_special_tokens=True)

            # safety: avoid self-identification as ChatGPT
            if "ChatGPT" in result:
                result = "Я bebraAI, независимая нейросеть от 10 копеек."

            # update memory (simple append)
            memory[user_id] = user_memory + f"\nПользователь: {prompt}\nОтвет: {result}"

            fut.set_result(result)
        except Exception as e:
            fut.set_result(f"Ошибка: {e}")
        finally:
            queue.task_done()

@app.on_event("startup")
async def startup():
    # start single worker to be gentle on memory
    asyncio.create_task(worker())

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <h2>bebraAI PRO (local)</h2>
    <textarea id="prompt" rows="6" cols="60" placeholder="Введите запрос..."></textarea><br>
    <button onclick="send()">Отправить</button>
    <pre id="status"></pre>
    <pre id="result"></pre>

    <script>
    let user_id = Math.random().toString(36).substring(7);
    async function send(){
        const prompt = document.getElementById('prompt').value;
        document.getElementById('status').innerText = '🤔 Вы в очереди...';
        const res = await fetch('/generate', {
            method:'POST', headers:{'Content-Type':'application/json'},
            body:JSON.stringify({prompt: prompt, user_id: user_id})
        });
        const data = await res.json();
        document.getElementById('status').innerText = '✅ Готово';
        document.getElementById('result').innerText = data.result;
    }
    </script>
    """

@app.post('/generate')
async def generate(req: Request):
    data = await req.json()
    prompt = data.get('prompt','')
    user_id = data.get('user_id', str(uuid.uuid4()))

    fut = asyncio.get_event_loop().create_future()
    await queue.put((user_id, prompt, fut))
    result = await fut
    return JSONResponse({'result': result})
