import streamlit as st
import asyncio
import os
import json
from mcp import ClientSession
from mcp.client.sse import sse_client
from ollama import AsyncClient as OllamaClient
from openai import AsyncOpenAI

# ==========================================
# Configurações Iniciais
# ==========================================
st.set_page_config(page_title="Agente de Vendas & Operações", page_icon="🤖", layout="centered")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
USE_OPENROUTER = bool(OPENROUTER_API_KEY)

# ==========================================
# Gerenciamento de Estado do Streamlit
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "system",
        "content": (
            "Você é um agente autônomo de gestão de vendas e operações. "
            "Sempre que precisar entender o panorama geral, use o get_entity_graph_context "
            "e o search_unstructured_notes para descobrir detalhes de conversas não-estruturadas. "
            "Seja direto e use as ferramentas disponíveis."
        )
    }]

# ==========================================
# Lógica Assíncrona Principal (Agente)
# ==========================================
async def process_message(user_input: str, status_container):
    # Inicializa o cliente LLM
    if USE_OPENROUTER:
        llm_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    else:
        llm_client = OllamaClient(host='http://ollama:11434')

    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Conecta no MCP Server temporariamente para resolver a interação
    status_container.write("Conectando ao Servidor MCP...")
    async with sse_client("http://mcp_server:8000/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Mapeia Ferramentas
            mcp_tools_response = await session.list_tools()
            llm_tools = []
            for tool in mcp_tools_response.tools:
                props = tool.inputSchema.get("properties", {}) if tool.inputSchema else {}
                reqs = tool.inputSchema.get("required", []) if tool.inputSchema else []
                llm_tools.append({
                    "type": "function",
                    "function": {"name": tool.name, "description": tool.description, "parameters": {"type": "object", "properties": props, "required": reqs}}
                })

            status_container.write("Analisando intenção...")
            
            # --- FLUXO OPENROUTER ---
            if USE_OPENROUTER:
                response = await llm_client.chat.completions.create(
                    model=OPENROUTER_MODEL, messages=st.session_state.messages, tools=llm_tools
                )
                msg = response.choices[0].message
                st.session_state.messages.append(msg.model_dump(exclude_none=True))

                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        func_name = tool_call.function.name
                        func_args = json.loads(tool_call.function.arguments)
                        status_container.write(f"🔧 Executando ferramenta: **{func_name}**...")
                        
                        result = await session.call_tool(func_name, arguments=func_args)
                        
                        st.session_state.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result.content[0].text if result.content else "{}"
                        })

                    status_container.write("Gerando resposta final...")
                    final_response = await llm_client.chat.completions.create(
                        model=OPENROUTER_MODEL, messages=st.session_state.messages
                    )
                    final_msg = final_response.choices[0].message
                    st.session_state.messages.append({"role": "assistant", "content": final_msg.content})
                    return final_msg.content
                else:
                    return msg.content

            # --- FLUXO OLLAMA LOCAL ---
            else:
                response = await llm_client.chat(model='llama3.1', messages=st.session_state.messages, tools=llm_tools)
                st.session_state.messages.append(response['message'])

                if response['message'].get('tool_calls'):
                    for tool_call in response['message']['tool_calls']:
                        func_name = tool_call['function']['name']
                        func_args = tool_call['function']['arguments']
                        status_container.write(f"🔧 Executando ferramenta: **{func_name}**...")
                        
                        result = await session.call_tool(func_name, arguments=func_args)
                        
                        st.session_state.messages.append({
                            "role": "tool",
                            "content": result.content[0].text if result.content else "{}"
                        })

                    status_container.write("Gerando resposta final...")
                    final_response = await llm_client.chat(model='llama3.1', messages=st.session_state.messages)
                    st.session_state.messages.append(final_response['message'])
                    return final_response['message']['content']
                else:
                    return response['message']['content']

# ==========================================
# Interface de Usuário (Streamlit)
# ==========================================
st.title("🤖 Agente Autônomo POC")
st.caption(f"Processamento: {'Nuvem (OpenRouter)' if USE_OPENROUTER else 'Local (Ollama)'}")

# Renderiza o histórico (ocultando prompts de sistema e metadados de tool_calls)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant" and msg.get("content"):
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# Caixa de input do usuário
if prompt := st.chat_input("Ex: Avance a oportunidade opp_1 e lance 40 horas..."):
    # Exibe imediatamente a mensagem do usuário
    with st.chat_message("user"):
        st.markdown(prompt)

    # Exibe a resposta do assistente (com um loader dinâmico para as ferramentas)
    with st.chat_message("assistant"):
        with st.status("Processando requisição...", expanded=True) as status:
            # Chama a função assíncrona dentro do Streamlit (que é síncrono)
            final_text = asyncio.run(process_message(prompt, status))
            status.update(label="Finalizado!", state="complete", expanded=False)
        
        # Renderiza o texto final
        st.markdown(final_text)