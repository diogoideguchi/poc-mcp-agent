import httpx
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from fastembed import TextEmbedding

app = FastAPI(title="Knowledge & Webhook Service (ArcadeDB)")

# ==========================================
# 1. Configuração do Embedding Local
# ==========================================
print("Carregando modelo de embeddings (FastEmbed/ONNX)...")
# Usaremos um modelo pequeno, em inglês/multilíngue, altamente eficiente
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
print("Modelo carregado e pronto na CPU!")

# ==========================================
# 2. Configuração do ArcadeDB
# ==========================================
ARCADE_URL = "http://arcadedb:2480/api/v1"
AUTH = ("root", "playwithdata")
DB_NAME = "SalesDB"
DB_COMMAND_URL = f"{ARCADE_URL}/command/{DB_NAME}"
DB_QUERY_URL = f"{ARCADE_URL}/query/{DB_NAME}"

def run_arcade_cmd(command: str, language: str = "sql", params: dict = None):
    """Executa um comando de escrita no ArcadeDB"""
    payload = {"language": language, "command": command}
    if params:
        payload["parameters"] = params
    resp = httpx.post(DB_COMMAND_URL, json=payload, auth=AUTH, timeout=10.0)
    resp.raise_for_status()
    return resp.json()

def run_arcade_query(query: str, language: str = "sql", params: dict = None):
    """Executa uma query de leitura no ArcadeDB"""
    payload = {"language": language, "command": query}
    if params:
        payload["parameters"] = params
    resp = httpx.post(DB_QUERY_URL, json=payload, auth=AUTH, timeout=10.0)
    resp.raise_for_status()
    return resp.json().get("result", [])

@app.on_event("startup")
def init_database():
    """Inicializa o banco de dados e as estruturas no ArcadeDB"""
    try:
        # Verifica se o DB existe, se não, cria
        resp = httpx.get(f"{ARCADE_URL}/exists/{DB_NAME}", auth=AUTH)
        if not resp.json().get("result", False):
            httpx.post(f"{ARCADE_URL}/server", json={"command": f"create database {DB_NAME}"}, auth=AUTH)
            print(f"Banco {DB_NAME} criado no ArcadeDB.")

        # Cria a estrutura Vetorial (Document/Vertex) usando a linguagem SQL do ArcadeDB
        schema_cmds = [
            "CREATE VERTEX TYPE Note IF NOT EXISTS",
            "CREATE PROPERTY Note.id IF NOT EXISTS STRING",
            "CREATE PROPERTY Note.text IF NOT EXISTS STRING",
            "CREATE PROPERTY Note.opp_id IF NOT EXISTS STRING",
            "CREATE PROPERTY Note.embedding IF NOT EXISTS LIST OF FLOAT",
            # Cria o índice HNSW para busca vetorial rápida
            "CREATE INDEX Note_embedding_idx IF NOT EXISTS ON Note (embedding) TYPE HNSW"
        ]
        for cmd in schema_cmds:
            run_arcade_cmd(cmd)

        # Insere a nota mockada vetorizando com o FastEmbed
        mock_text = "Reunião de Kickoff TechCorp: O cliente enfatizou que o foco absoluto do projeto deve ser segurança em nuvem. Eles exigem criptografia AES-256 e auditoria semanal."
        # FastEmbed retorna um generator, pegamos o primeiro item e convertemos pra lista de floats
        mock_vector = list(embedding_model.embed([mock_text]))[0].tolist()
        
        run_arcade_cmd(
            "UPDATE Note SET id = 'note_1', text = :text, opp_id = 'opp_1', embedding = :vec UPSERT WHERE id = 'note_1'",
            params={"text": mock_text, "vec": mock_vector}
        )
        print("Mock Vetorial inserido com sucesso!")

    except Exception as e:
        print(f"Aviso na inicialização do ArcadeDB: {e}")

# ==========================================
# 3. Endpoints (Graph & Vector)
# ==========================================
class CRMEvent(BaseModel):
    opp_id: str
    client: str
    salesperson: str
    stage: str
    price: float
    cost: float

def update_graph_in_arcade(event: CRMEvent):
    """Atualiza a topologia de grafos usando a linguagem Cypher suportada pelo ArcadeDB"""
    cypher_query = """
    MERGE (c:Client {name: $client})
    MERGE (s:Salesperson {name: $salesperson})
    MERGE (o:Opportunity {id: $opp_id})
    SET o.stage = $stage, o.price = $price, o.cost = $cost
    MERGE (c)-[:HAS_OPPORTUNITY]->(o)
    MERGE (s)-[:MANAGES]->(o)
    """
    try:
        run_arcade_cmd(cypher_query, language="cypher", params=event.model_dump())
        
        # Como o OpenCypher às vezes possui sintaxes restritivas para loops FOREACH com MERGE
        # lidamos com a evolução para "Project" de forma separada por segurança
        if event.stage == "won":
            project_cypher = """
            MATCH (o:Opportunity {id: $opp_id})
            MERGE (p:Project {id: $opp_id})
            MERGE (o)-[:EVOLVED_TO]->(p)
            """
            run_arcade_cmd(project_cypher, language="cypher", params={"opp_id": event.opp_id})
            
        print(f"[Graph] Oportunidade {event.opp_id} atualizada no ArcadeDB.")
    except Exception as e:
        print(f"Erro ao processar Cypher no ArcadeDB: {e}")

@app.post("/webhook/crm_update")
def crm_webhook(event: CRMEvent, bg_tasks: BackgroundTasks):
    bg_tasks.add_task(update_graph_in_arcade, event)
    return {"status": "Event received"}

@app.get("/graph/context/{entity}")
def get_graph_context(entity: str):
    cypher_query = """
    MATCH (n)-[r]-(m) 
    WHERE n.id = $id OR n.name = $id
    RETURN labels(n)[0] AS EntityType, coalesce(n.id, n.name) AS Entity,
           type(r) AS Relationship, labels(m)[0] AS ConnectedType, coalesce(m.id, m.name) AS ConnectedEntity
    """
    result = run_arcade_query(cypher_query, language="cypher", params={"id": entity})
    return {"context": result}

@app.get("/vector/search")
def search_notes(query: str, opp_id: str = None):
    # 1. Gera o embedding da pergunta localmente
    query_vector = list(embedding_model.embed([query]))[0].tolist()
    
    # 2. Busca no ArcadeDB comparando vetores (usamos SQL para tirar vantagem da função vectorDistance)
    sql_query = "SELECT text, opp_id, vectorDistance(embedding, :vec) as distance FROM Note"
    
    params = {"vec": query_vector}
    if opp_id:
        sql_query += " WHERE opp_id = :opp_id"
        params["opp_id"] = opp_id
        
    sql_query += " ORDER BY distance ASC LIMIT 2"

    results = run_arcade_query(sql_query, language="sql", params=params)
    
    # Limpa a resposta para retornar apenas o texto relevante ao Agente
    documents = [r["text"] for r in results]
    return {"results": documents}