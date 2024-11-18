import json
from datetime import datetime
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load JSON file
with open("/Users/tarikhadi/Desktop/contratos_json.json", 'r') as file:
    json_data = json.load(file)

data_de_hoje = datetime.today().strftime("%d-%m-%Y")

api_key = st.secrets["OPENAI_API_KEY"]

def create_chunks_from_json(json_data):
    chunks = []

    for key, value in json_data.items():
        empresa_name = key

        chunk = f"Empresa: {empresa_name}\n"
        
        for field, field_value in value.items():
            if isinstance(field_value, list):  
                field_value_str = "\n- ".join([str(item) for item in field_value])
                chunk += f"{field}: \n- {field_value_str}\n"
            elif isinstance(field_value, dict):  
                chunk += f"{field}: \n"
                for sub_key, sub_value in field_value.items():
                    chunk += f"  {sub_key}: {sub_value}\n"
            else:  
                chunk += f"{field}: {field_value}\n"
        
        chunks.append(chunk)
    
    return chunks

def store_in_vector_db(chunks):
    documents = [Document(page_content=chunk, metadata={}) for chunk in chunks]
    
    embedding_model = OpenAIEmbeddings(model='text-embedding-ada-002', disallowed_special=(), api_key=api_key)
    
    vectorstore = Chroma.from_documents(documents, embedding=embedding_model)
    return vectorstore

def process_pdf_for_rag():
    chunks = create_chunks_from_json(json_data)
    vectorstore = store_in_vector_db(chunks)
    return vectorstore, chunks  # Return both vectorstore and chunks

st.title("Assistente de Contratos - Shopping Center Iguatemi")

@st.cache_resource
def get_vectorstore_and_chunks():
    return process_pdf_for_rag()

vectorstore, chunks = get_vectorstore_and_chunks()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# store history
def build_conversation_history(messages):
    history = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            history += f"User: {content}\n"
        else:
            history += f"Assistant: {content}\n"
    return history

# Create CHATBOT_INSTRUCTIONS after chunks are created
CHATBOT_INSTRUCTIONS = f"""
### INSTRUÇÕES:
Você é um assistente especializado em responder perguntas sobre contratos de aluguel entre o Shopping Iguatemi e as lojas presentes nos shoppings.
O usuário do chatbot vai fazer perguntas super específicas sobre cada contrato, assim como também pode fazer perguntas gerais para que contemplaram diversos contratos ao mesmo tempo.

Você tem acesso a todos os contratos aqui: {chr(10).join(chunks)}

As chaves de cada elemento do JSON são:
    nome_da_loja_querendo_alugar:
    reajuste_previsto: 
    data_inicio_contrato: 
    escopos_previstos_contrato_XXXX: 
    multa_resolucao_imotivada:
    condicoes_comerciais_previstas: 
    renovacao_automatica: 
    escopos_nao_previstos_contrato_XXXX: 
    cnpj: 
    vencimento_contrato: 
    endereco: 
    representantes_legais: 
    garantia_prevista: 
    fundo_promocao: 
    limite_carga_forca_luz: 
    raio_exclusividade: 
    prazo_locacao: 
    aluguel_escalonado: 
    penalidades_por_inadimplencia: 

Os contratos estão no formato JSON, como chaves e valores sobre vários pontos de interesse do usuário em relação ao contrato.

Você receberá perguntas do tipo (são apenas exemplos):

- Perguntas fechadas (exemplos de perguntas):
    Com relação a loja X, qual é o índice de reajuste previsto em contrato?
    Qual é a data de inicio referente o contrato com o fornecedor X?
    Qual(is) é(são) o(s) escopo(s) previsto em contrato com o(a) X?
    Existe multa em caso de resolução imotivada com o fornecedor X?
    Quais as condições comerciais previstas em contrato com a loja X?
    Existe clausula de renovação automática prevista em contrato com a loja X?
    Qual(is) é(são) o(s) escopo não previsto(s) em contrato com o(a) a loja X?
    Qual é o cnpj da loja X?
    Quando vence o contrato da loja X?

Loja, neste caso, são com quem o shopping Iguatemi está firmando o contrato.

- Perguntas abertas
    Quais contratos possuem índice de reajuste IGP-M e quais contratos possuem índice de reajuste IPCA?
    Quais contratos vencem nos próximos 12 meses?
    Quais contratos possuem multa por rescisão/resolução imotivada?

Hoje é {data_de_hoje}.

### NOTAS:
1) SEMPRE que tiver que listar diversas lojas para responder uma pergunta (Ex: "Quais contratos vencem nos proximos 6 meses?") LISTE TODAS AS EMPRESAS QUE SE ENCAIXAM NA RESPOSTA. Você JAMAIS deve mencionar apenas alguns exemplos, e sim todas as empresas que se enquadram na pergunta do usuário. Se for preciso, use todo seu output window para responder, mas nunca deixa de responder de forma COMPLETA.
2) Responda de forma objetiva. Não de explicações desnecessárias. Informa de forma CONCISA, OBJETIVA porém COMPLETA todas as perguntas do usuário. 
3) Considere TODOS os contratos que você tem acesso para responder a pergunta.
4) Se, por exemplo, o usuário perguntar "Quais contratos se encerram nos proximos 2 meses?" você deve considerar a {data_de_hoje} e olhar para a chave 'vencimento_contrato' e listar TODOS os contratos (com os dados necessarios) que fazem sentido em relação à pergunta do usuário.

"""

# handles the user prompt
if prompt := st.chat_input("O que quer saber sobre os contratos?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # full prompt
    conversation_history = build_conversation_history(st.session_state.messages)
    full_prompt = CHATBOT_INSTRUCTIONS + "\n" + conversation_history + f"\nUser question: {prompt}"

    # generate the response
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
    )

    # what it needs to use to generate the response
    with st.chat_message("assistant"):
        response = qa_chain({"query": full_prompt})

        # extract the answer
        result = response['result']

        st.markdown(result)
        st.session_state.messages.append({"role": "assistant", "content": result})