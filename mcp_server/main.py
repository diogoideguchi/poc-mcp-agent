from mcp.server.fastmcp import FastMCP
import httpx
from neo4j import GraphDatabase

mcp = FastMCP("SalesBudgetBridge",host="0.0.0.0", port=8000)

# Usando os nomes dos containers como hostname
CRM_URL = "http://crm_service:8001"
BUDGET_URL = "http://budget_service:8002"

# Conexão com o Neo4j
NEO4J_URI = "neo4j://neo4j_db:7687"
NEO4J_AUTH = ("neo4j", "password")
driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

KNOWLEDGE_URL = "http://knowledge_service:8003"

@mcp.tool()
def get_crm_opportunities() -> dict:
    return httpx.get(f"{CRM_URL}/opportunities").json()

@mcp.tool()
def move_opportunity_stage(opp_id: str, stage: str) -> dict:
    return httpx.put(f"{CRM_URL}/opportunities/{opp_id}/stage", json={"stage": stage}).json()

@mcp.tool()
def log_project_hours(project_id: str, seniority: str, hours: float) -> dict:
    return httpx.post(f"{BUDGET_URL}/projects/{project_id}/log", json={"seniority": seniority, "hours": hours}).json()

@mcp.tool()
def get_project_cpi(project_id: str) -> dict:
    return httpx.get(f"{BUDGET_URL}/projects/{project_id}/cpi").json()

@mcp.tool()
def sync_knowledge_graph() -> str:
    """Extrai os dados do CRM e Budget e constrói as relações no banco de dados de grafos."""
    crm_data = httpx.get(f"{CRM_URL}/opportunities").json()
    
    # Query Cypher para criar/atualizar os nós e relacionamentos
    query = """
    MERGE (c:Client {name: $client})
    MERGE (s:Salesperson {name: $salesperson})
    MERGE (o:Opportunity {id: $opp_id})
    SET o.stage = $stage, o.price = $price, o.cost = $cost
    
    MERGE (c)-[:HAS_OPPORTUNITY]->(o)
    MERGE (s)-[:MANAGES]->(o)
    
    // Se a oportunidade virou projeto (está no budget)
    FOREACH (ignoreMe IN CASE WHEN $stage = 'won' THEN [1] ELSE [] END |
        MERGE (p:Project {id: $opp_id})
        MERGE (o)-[:EVOLVED_TO]->(p)
    )
    """
    
    with driver.session() as session:
        for opp_id, data in crm_data.items():
            session.run(query, opp_id=opp_id, client=data['client'], 
                        salesperson=data['salesperson'], stage=data['stage'], 
                        price=data['price'], cost=data['cost'])
            
    return "Knowledge Graph sincronizado com sucesso!"

@mcp.tool()
def get_entity_graph_context(entity_id_or_name: str) -> dict:
    """Usa GraphRAG para buscar as conexões diretas de um projeto, oportunidade, cliente ou vendedor no grafo."""
    query = """
    MATCH (n)-[r]-(m)
    WHERE n.id = $identifier OR n.name = $identifier
    RETURN labels(n)[0] AS EntityType, coalesce(n.id, n.name) AS Entity,
           type(r) AS Relationship,
           labels(m)[0] AS ConnectedType, coalesce(m.id, m.name) AS ConnectedEntity
    """
    with driver.session() as session:
        result = session.run(query, identifier=entity_id_or_name)
        connections = [record.data() for record in result]
        
    return {"entity_queried": entity_id_or_name, "graph_context": connections}

@mcp.tool()
def get_entity_graph_context(entity_id_or_name: str) -> dict:
    """Busca as conexões diretas de um projeto, cliente ou vendedor no grafo estruturado."""
    return httpx.get(f"{KNOWLEDGE_URL}/graph/context/{entity_id_or_name}").json()

@mcp.tool()
def search_unstructured_notes(query: str, opp_id: str = "") -> dict:
    """Busca informações textuais não-estruturadas (atas de reunião, e-mails, requisitos) usando semântica."""
    params = {"query": query}
    if opp_id:
        params["opp_id"] = opp_id
    return httpx.get(f"{KNOWLEDGE_URL}/vector/search", params=params).json()

if __name__ == "__main__":
    # Rodando como servidor SSE para permitir chamadas via rede do Docker
    mcp.run(transport='sse')