import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client
from ollama import AsyncClient

async def main():
    print("Iniciando conexão MCP (SSE) e Ollama...")
    
    # Cliente do Ollama apontando para o container "ollama"
    llm_client = AsyncClient(host='http://ollama:11434')
    
    # Conecta no servidor MCP usando SSE
    async with sse_client("http://mcp_server:8000/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools_response = await session.list_tools()
            ollama_tools = []
            
            for tool in mcp_tools_response.tools:
                properties = tool.inputSchema.get("properties", {}) if tool.inputSchema else {}
                required = tool.inputSchema.get("required", []) if tool.inputSchema else []
                ollama_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {"type": "object", "properties": properties, "required": required}
                    }
                })

            messages = [{
                "role": "system", 
                "content": (
                    "Você é um agente autônomo de gestão de vendas e operações. "
                    "Você tem acesso a um Knowledge Graph (GraphRAG). "
                    "Sempre que precisar entender o panorama geral (ex: 'Quem é o vendedor do projeto X?' ou "
                    "'Quais oportunidades o cliente Y possui?'), use a ferramenta sync_knowledge_graph primeiro para "
                    "garantir que os dados estejam atualizados, e depois use get_entity_graph_context para deduzir as respostas. "
                    "Seja direto em suas ações."
                )            
            }] 
            print("\n=== Agente Pronto! (Digite 'sair' para encerrar) ===")
            
            while True:
                user_input = await asyncio.to_thread(input, "\nVocê: ")
                if user_input.lower() in ['sair', 'exit', 'quit']:
                    break

                messages.append({"role": "user", "content": user_input})

                response = await llm_client.chat(model='llama3.1', messages=messages, tools=ollama_tools)
                messages.append(response['message'])

                if response['message'].get('tool_calls'):
                    for tool_call in response['message']['tool_calls']:
                        func_name = tool_call['function']['name']
                        func_args = tool_call['function']['arguments']
                        print(f"\n[Ação] Executando: {func_name}({func_args})...")

                        result = await session.call_tool(func_name, arguments=func_args)
                        messages.append({
                            "role": "tool",
                            "content": result.content[0].text if result.content else "{}",
                        })

                    final_response = await llm_client.chat(model='llama3.1', messages=messages)
                    messages.append(final_response['message'])
                    print(f"\nAgente: {final_response['message']['content']}")
                else:
                    print(f"\nAgente: {response['message']['content']}")

if __name__ == "__main__":
    asyncio.run(main())