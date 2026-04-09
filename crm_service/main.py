from fastapi import FastAPI, HTTPException, BackgroundTasks
import httpx
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="CRM Service")

class StageUpdate(BaseModel):
    stage: str

opportunities = {
    "opp_1": {
        "client": "TechCorp", 
        "salesperson": "Carlos", 
        "stage": "lead", 
        "price": 50000.0, 
        "cost": 30000.0
    }
}

KNOWLEDGE_WEBHOOK_URL = "http://knowledge_service:8003/webhook/crm_update"

def fire_webhook(data: dict):
    try:
        httpx.post(KNOWLEDGE_WEBHOOK_URL, json=data)
    except Exception as e:
        print(f"Erro ao disparar webhook: {e}")

@app.put("/opportunities/{opp_id}/stage")
def update_stage(opp_id: str, data: StageUpdate, bg_tasks: BackgroundTasks):
    valid_stages = ["lead", "proposal", "won", "loss"]
    if opp_id not in opportunities:
        raise HTTPException(status_code=404, detail="Não encontrada")
    if data.stage not in valid_stages:
        raise HTTPException(status_code=400, detail="Estágio inválido")
    
    opportunities[opp_id]["stage"] = data.stage
    
    # Prepara o payload combinando o ID com os dados da oportunidade
    payload = {"opp_id": opp_id, **opportunities[opp_id]}
    
    # Dispara o evento em tempo real sem bloquear a resposta da API
    bg_tasks.add_task(fire_webhook, payload)
    
    return {"status": "success", "data": opportunities[opp_id]}

@app.get("/opportunities")
def get_opportunities():
    return opportunities

@app.put("/opportunities/{opp_id}/stage")
def update_stage(opp_id: str, data: StageUpdate):
    valid_stages = ["lead", "proposal", "won", "loss"]
    if opp_id not in opportunities:
        raise HTTPException(status_code=404, detail="Não encontrada")
    if data.stage not in valid_stages:
        raise HTTPException(status_code=400, detail="Estágio inválido")
    
    opportunities[opp_id]["stage"] = data.stage
    return {"status": "success", "data": opportunities[opp_id]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)