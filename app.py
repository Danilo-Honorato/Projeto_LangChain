import os
import tempfile #Salvar os arquivos em disco

import streamlit as st

from decouple import config

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')
persist_directory = 'db'

#FUNÇÃO PARA CRIAR OS CHUNKs DO ARQUIVO
def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file.read()) #Salvar o arquivo binário em disco
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    os.remove(temp_file_path) #Remover o arquivo do disco

    #Configuração das quebras dos chunks
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400,
    )
    chunks = text_spliter.split_documents(documents=docs)
    return chunks

#FUNÇÃO PARA CRIAR EMBADDING (VETORES) COM OPENAI
def load_existing_vector_store():
    if os.path.exists(os.path.join(persist_directory)):
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(),
        )
        return vector_store
    return None

#FUNÇÃO PARA VERIFICAR SE EXISTE EMBADDING (VETORES)
def add_to_vector_store(chunks, vector_store=None):
    if vector_store:
        vector_store.add_documents(chunks)
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(),
            persist_directory=persist_directory,
        )
    return vector_store

#FUNÇÃO PARA CRIAR A RESPOSTA COM LLM
def ask_question(model, query, vector_store):
    llm = ChatOpenAI(model=model)
    retriever = vector_store.as_retriever() #Retriever para buscar no banco de dados

    system_prompt = '''
    Use o contexto para responder as perguntas.
    Se não encontrar uma resposta no contexto,
    explique que não há informações disponíveis.
    Responda em formato de markdown e com visualizações
    elaboradas e interativas.
    Contexto: {context}
    '''
    messages = [('system', system_prompt)]
    for message in st.session_state.messages:
        messages.append((message.get('role'), message.get('content'))) #Remontar o chat de mensagens 
    messages.append(('human', '{input}')) #Ultima pergunta feita

    prompt = ChatPromptTemplate.from_messages(messages) #Criando um prompt com o historio do chat

    question_answer_chain = create_stuff_documents_chain( #Ferramenta para perguntas e respostas
        llm=llm,
        prompt=prompt,
    )
    chain = create_retrieval_chain( #Ferramenta para buscar a pergunta com base no arquivo
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
    )
    response = chain.invoke({'input': query})
    return response.get('answer')


vector_store = load_existing_vector_store()

#CONFIGURAÇÕES DOS TITULOS E ICONES
st.set_page_config(
    page_title='Chat PyGPT',
    page_icon='📄',
)
st.header('🤖 Chat com seus documentos (RAG)')

#PÁGINA LATERAL COM AS OPÇÕES DE LLM E DOC
with st.sidebar:
    st.sidebar.markdown('### Sobre')
    st.sidebar.markdown('''Essa ferramenta foi desenvolvida para buscar respostas utilizando uma LLM com base no arquivo .PDF fornecido.
                        Carregue o arquivo no campo abaixo, em seguida, escolha qual LLM para elaborar sua resposta.
                        ''')
    #SOLICITAR OS ARQUIVOS
    st.header('Upload de arquivos 📄')
    uploaded_files = st.file_uploader(
        label='Faça o upload de arquivos PDF',
        type=['pdf'],
        accept_multiple_files=True,
    )

    #PROCESSAO PARA EXECUTAR A CRIAÇÃO DOS CHUNCKS E VETORES
    if uploaded_files:
        with st.spinner('Processando documentos...'):
            all_chunks = []
            for uploaded_file in uploaded_files:
                chunks = process_pdf(file=uploaded_file)
                all_chunks.extend(chunks)
            vector_store = add_to_vector_store( #Criar e calcular os vetores
                chunks=all_chunks,
                vector_store=vector_store,
            )

    #OPÇÕES PARA CARREGAR AS LLMs
    model_options = [
        'gpt-3.5-turbo',
        'gpt-4o',
        'llama-3.3-70b-versatile',
        'gemma2-9b-it',
        'llama3-70b-8192',
        'whisper-large-v3-turbo',
        'mixtral-8x7b-32768',
        ]
    selected_model = st.sidebar.selectbox(
        label='Selecione o modelo LLM',
        options=model_options,
    )
    st.sidebar.markdown('Nota: A persistência dos dados foi criado no SQLite')

#Lista para salvar as mensagens
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

#LINHA PARA O USUÁRIO DIGITAR A MENSAGEM
question = st.chat_input('Como posso ajudar?')

#ETAPA PARA CRIAR A CONEXÃO COM O USUÁRIO
if vector_store and question:
    for message in st.session_state.messages:
        st.chat_message(message.get('role')).write(message.get('content'))

    st.chat_message('user').write(question)
    st.session_state.messages.append({'role': 'user', 'content': question}) #Montar o histório de mensagem

    #Elaborar a resposta do usuario com LLM
    with st.spinner('Buscando resposta...'):
        response = ask_question(
            model=selected_model,
            query=question,
            vector_store=vector_store,
        )

        st.chat_message('ai').write(response)#FUNÇÃO PARA REGISTRAR MENSAGENS
        st.session_state.messages.append({'role': 'ai', 'content': response})
