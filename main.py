from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import openai
import os
import uuid

app = FastAPI()

# Берём API ключ из переменной окружения
openai.api_key = os.environ.get("OPENAI_API_KEY", "")

# Модели
MODELS = {
    "мини": "bebraAI-mini-1.0",
    "стандарт": "bebraAI-standard-1.0"
}

# Хранение чатов и их истории
chats = {}

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
    <head>
        <style>
            body {font-family:sans-serif; display:flex; height:100vh; margin:0;}
            #chatList {width:200px; border-right:1px solid #ccc; padding:10px; overflow-y:auto;}
            #chatWindow {flex:1; display:flex; flex-direction:column;}
            #messages {flex:1; padding:10px; overflow-y:auto; border-bottom:1px solid #ccc;}
            .message {margin:5px 0;}
            .user {color:blue;}
            .assistant {color:green;}
            #inputArea {display:flex;}
            #inputArea textarea {flex:1;}
        </style>
    </head>
    <body>
        <div id="chatList">
            <button onclick="createChat()">+ Новый чат</button>
            <ul id="chats"></ul>
        </div>
        <div id="chatWindow">
            <div id="messages"></div>
            <div id="inputArea">
                <select id="modelSelect">
                    <option value="мини">мини 1.0</option>
                    <option value="стандарт">стандарт 1.0</option>
                </select>
                <textarea id="userInput" rows="2"></textarea>
                <button onclick="sendMessage()">Отправить</button>
            </div>
            <div id="thinking" style="display:none;">🤔 Нейросеть думает...</div>
        </div>

        <script>
            let currentChat = null;

            async function loadChats(){
                const res = await fetch("/chats");
                const data = await res.json();
                const ul = document.getElementById("chats");
                ul.innerHTML="";
                for(let c of data){
                    const li = document.createElement("li");
                    li.innerText = c.name;
                    li.onclick = ()=>selectChat(c.id);
                    ul.appendChild(li);
                }
            }

            async function createChat(){
                const name = prompt("Имя чата:");
                if(!name) return;
                const res = await fetch("/chat/create", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({name})});
                await loadChats();
            }

            async function selectChat(id){
                currentChat = id;
                const res = await fetch(`/chat/${id}/history`);
                const data = await res.json();
                const messagesDiv = document.getElementById("messages");
                messagesDiv.innerHTML="";
                for(let m of data){
                    const div = document.createElement("div");
                    div.className="message "+(m.role=="user"?"user":"assistant");
                    div.innerText=m.content;
                    messagesDiv.appendChild(div);
                }
                const modelRes = await fetch(`/chat/${id}/model`);
                const modelData = await modelRes.json();
                document.getElementById("modelSelect").value = modelData.model;
            }

            async function sendMessage(){
                if(!currentChat) return alert("Выберите чат!");
                const text = document.getElementById("userInput").value;
                if(!text) return;
                const messagesDiv = document.getElementById("messages");
                const div = document.createElement("div");
                div.className="message user";
                div.innerText = text;
                messagesDiv.appendChild(div);
                document.getElementById("userInput").value="";
                document.getElementById("thinking").style.display="block";
                const model = document.getElementById("modelSelect").value;
                const res = await fetch(`/chat/${currentChat}/send`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({content:text, model})});
                const data = await res.json();
                const div2 = document.createElement("div");
                div2.className="message assistant";
                div2.innerText = data.response;
                messagesDiv.appendChild(div2);
                document.getElementById("thinking").style.display="none";
            }

            loadChats();
        </script>
    </body>
    </html>
    """

@app.get("/chats")
async def get_chats():
    return [{"id":cid, "name":c["name"]} for cid,c in chats.items()]

@app.post("/chat/create")
async def create_chat(req: Request):
    data = await req.json()
    chat_id = str(uuid.uuid4())
    chats[chat_id] = {"name":data.get("name","Без имени"), "model":"мини", "messages":[]}
    return {"id":chat_id}

@app.get("/chat/{chat_id}/history")
async def chat_history(chat_id: str):
    if chat_id not in chats:
        return JSONResponse({"error":"Чат не найден"}, status_code=404)
    return chats[chat_id]["messages"]

@app.get("/chat/{chat_id}/model")
async def chat_model(chat_id: str):
    if chat_id not in chats:
        return JSONResponse({"error":"Чат не найден"}, status_code=404)
    return {"model":chats[chat_id]["model"]}

@app.post("/chat/{chat_id}/send")
async def chat_send(chat_id: str, req: Request):
    if chat_id not in chats:
        return JSONResponse({"error":"Чат не найден"}, status_code=404)
    data = await req.json()
    model = data.get("model", "мини")
    content = data.get("content", "")
    chats[chat_id]["messages"].append({"role":"user","content":content})
    response = openai.ChatCompletion.create(
        model=MODELS.get(model,"bebraAI-mini-1.0"),
        messages=[{"role":m["role"],"content":m["content"]} for m in chats[chat_id]["messages"]],
        max_tokens=500
    )
    reply = response.choices[0].message.content
    chats[chat_id]["messages"].append({"role":"assistant","content":reply})
    chats[chat_id]["model"] = model
    return {"response": reply}
