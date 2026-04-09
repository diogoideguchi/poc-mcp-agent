from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from neo4j import GraphDatabase
import chromadb
import uvicorn

app = FastAPI(title="Knowledge & Webhook Service")

# Setup Neo4j (Graph)
NEO4J_URI = "neo4j://neo4j_db:7687"
driver = GraphDatabase.driver(NEO4J_URI, auth=("neo4j", "password"))

# Setup ChromaDB (Vector) localmente na pasta /app/chroma_data
chroma_client = chromadb.PersistentClient(path="./chroma_data")
notes_collection = chroma_client.get_or_create_collection(name="crm_notes")

# Injeta um dado não-estruturado mockado no VectorDB para nosso teste
notes_collection.upsert(
    documents=[
        "Reunião de Kickoff TechCorp: O cliente enfatizou que o foco absoluto do projeto deve ser segurança em nuvem. Eles exigem criptografia AES-256 e auditoria semanal."
    ],
    metadatas=[{"opp_id": "opp_1", "type": "meeting_note"}],
    ids=["note_1"]
)

class CRMEvent(BaseModel):
    opp_id: str
    client: str
    salesperson: str
    stage: str
    price: float
    cost: float

def update_graph(event: CRMEvent):
    """Lógica de atualização do Neo4j baseada no evento recebido."""
    query = """
    MERGE (c:Client {name: $client})
    MERGE (s:Salesperson {name: $salesperson})
    MERGE (o:Opportunity {id: $opp_id})
    SET o.stage = $stage, o.price = $price, o.cost = $cost
    MERGE (c)-[:HAS_OPPORTUNITY]->(o)
    MERGE (s)-[:MANAGES]->(o)
    FOREACH (ignore IN CASE WHEN $stage = 'won' THEN [1] ELSE [] END |
        MERGE (p:Project {id: $opp_id})
        MERGE (o)-[:EVOLVED_TO]->(p)
    )
    """
    with driver.session() as session:
        session.run(query, **event.model_dump())
        print(f"[Graph Event] Oportunidade {event.opp_id} atualizada no grafo.")

@app.post("/webhook/crm_update")
def crm_webhook(event: CRMEvent, bg_tasks: BackgroundTasks):
    # Processa a atualização do grafo em background para não travar o CRM
    bg_tasks.add_task(update_graph, event)
    return {"status": "Event received"}

@app.get("/graph/context/{entity}")
def get_graph_context(entity: str):
    query = """
    MATCH (n)-[r]-(m) WHERE n.id = $id OR n.name = $id
    RETURN labels(n)[0] AS EntityType, coalesce(n.id, n.name) AS Entity,
           type(r) AS Relationship, labels(m)[0] AS ConnectedType, coalesce(m.id, m.name) AS ConnectedEntity
    """
    with driver.session() as session:
        result = session.run(query, id=entity)
        return {"context": [record.data() for record in result]}

@app.get("/vector/search")
def search_notes(query: str, opp_id: str = None):
    # Busca semântica usando os embeddings automáticos do ChromaDB
    where_filter = {"opp_id": opp_id} if opp_id else None
    results = notes_collection.query(
        query_texts=[query],
        n_results=2,
        where=where_filter
    )
    return {"results": results["documents"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)