import asyncio
import os
import json
from mcp import ClientSession
from mcp.client.sse import sse_client
from ollama import AsyncClient as OllamaClient
from openai import AsyncOpenAI

# Configurações de Ambiente
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
USE_OPENROUTER = bool(OPENROUTER_API_KEY)

async def main():
    print("Iniciando conexão MCP (SSE)...")
    
    # Inicializa o cliente LLM correspondente
    if USE_OPENROUTER:
        print(f"Modo Nuvem: Usando OpenRouter (Modelo: {OPENROUTER_MODEL})")
        llm_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
    else:
        print("Modo Local: Usando Ollama (Modelo: llama3.1)")
        llm_client = OllamaClient(host='http://ollama:11434')
    
    # Conecta no servidor MCP usando SSE
    async with sse_client("http://mcp_server:8000/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools_response = await session.list_tools()
            llm_tools = []
            
            for tool in mcp_tools_response.tools:
                properties = tool.inputSchema.get("properties", {}) if tool.inputSchema else {}
                required = tool.inputSchema.get("required", []) if tool.inputSchema else []
                llm_tools.append({
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
                    "Use o get_entity_graph_context para entender relações estruturais de negócio "
                    "e o search_unstructured_notes para descobrir detalhes de conversas. "
                    "Seja direto e use as ferramentas disponíveis."
                )
            }]
            
            print("\n=== Agente Pronto! (Digite 'sair' para encerrar) ===")
            
            while True:
                user_input = await asyncio.to_thread(input, "\nVocê: ")
                if user_input.lower() in ['sair', 'exit', 'quit']:
                    break

                messages.append({"role": "user", "content": user_input})

                # ==========================================
                # FLUXO OPENROUTER
                # ==========================================
                if USE_OPENROUTER:
                    response = await llm_client.chat.completions.create(
                        model=OPENROUTER_MODEL, 
                        messages=messages, 
                        tools=llm_tools
                    )
                    msg = response.choices[0].message
                    
                    # Salva a mensagem do assistente no histórico (incluindo as tool_calls)
                    messages.append(msg.model_dump(exclude_none=True))

                    if msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            func_name = tool_call.function.name
                            # OpenRouter retorna argumentos como String JSON
                            func_args = json.loads(tool_call.function.arguments) 
                            print(f"\n[Ação] Executando: {func_name}({func_args})...")

                            result = await session.call_tool(func_name, arguments=func_args)
                            
                            # OpenAI/OpenRouter exige o tool_call_id na resposta
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result.content[0].text if result.content else "{}"
                            })

                        # Segunda chamada para gerar a resposta baseada nos dados da ferramenta
                        final_response = await llm_client.chat.completions.create(
                            model=OPENROUTER_MODEL, 
                            messages=messages
                        )
                        final_msg = final_response.choices[0].message
                        messages.append({"role": "assistant", "content": final_msg.content})
                        print(f"\nAgente: {final_msg.content}")
                    else:
                        print(f"\nAgente: {msg.content}")

                # ==========================================
                # FLUXO OLLAMA LOCAL
                # ==========================================
                else:
                    response = await llm_client.chat(model='llama3.1', messages=messages, tools=llm_tools)
                    messages.append(response['message'])

                    if response['message'].get('tool_calls'):
                        for tool_call in response['message']['tool_calls']:
                            func_name = tool_call['function']['name']
                            # Ollama Python SDK já retorna um dicionário
                            func_args = tool_call['function']['arguments'] 
                            print(f"\n[Ação] Executando: {func_name}({func_args})...")

                            result = await session.call_tool(func_name, arguments=func_args)
                            
                            messages.append({
                                "role": "tool",
                                "content": result.content[0].text if result.content else "{}"
                            })

                        final_response = await llm_client.chat(model='llama3.1', messages=messages)
                        messages.append(final_response['message'])
                        print(f"\nAgente: {final_response['message']['content']}")
                    else:
                        print(f"\nAgente: {response['message']['content']}")

if __name__ == "__main__":
    asyncio.run(main())