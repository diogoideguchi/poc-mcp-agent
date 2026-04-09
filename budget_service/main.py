from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Budget Service")

class WorkLog(BaseModel):
    seniority: str
    hours: float

rates = {"junior": 50.0, "pleno": 100.0, "senior": 150.0}
projects = {"opp_1": {"budget": 30000.0, "actual_cost": 0.0, "logs": []}}

@app.post("/projects/{project_id}/log")
def log_hours(project_id: str, log: WorkLog):
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Não encontrado")
    if log.seniority not in rates:
        raise HTTPException(status_code=400, detail="Senioridade inválida")

    cost = rates[log.seniority] * log.hours
    projects[project_id]["actual_cost"] += cost
    projects[project_id]["logs"].append(log.model_dump())
    
    return {"status": "success", "added_cost": cost, "total_cost": projects[project_id]["actual_cost"]}

@app.get("/projects/{project_id}/cpi")
def get_cpi(project_id: str):
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Não encontrado")

    p = projects[project_id]
    ev = p["budget"]
    ac = p["actual_cost"]
    cpi = (ev / ac) if ac > 0 else 1.0 
    
    return {"project_id": project_id, "CPI": round(cpi, 2)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)