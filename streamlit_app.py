import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.indexes.vectorstore import RetrievalQA
from dotenv import load_dotenv
from retriever import MistralLangChainLLM  # remplace par ta classe
from Retriever import Neo4jRetriever


# ----------------------------------------
# 1. Charger les variables d’environnement
# ----------------------------------------
load_dotenv()

# ----------------------------------------
# 2. Instancier le LLM Albert
# ----------------------------------------
llm = MistralLangChainLLM(temperature=0.2)

# ----------------------------------------
# 3. Créer le prompt template pour la QA
# ----------------------------------------
# {context} = textes du graphe Neo4j
# {question} = question utilisateur
prompt_template = """
Tu es un assistant expert en véhicules autonomes.
Voici le contexte extrait d'un graphe de connaissances :
{context}

Question : {question}

Réponds de façon précise, en t’appuyant seulement sur ces faits. Si ce n’est pas dans le contexte, répond « Désolé, je n’ai pas cette information. ».
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# ----------------------------------------
# 4. Instancier le retriever Neo4j personnalisé
# ----------------------------------------
graph_retriever = Neo4jRetriever()

# ----------------------------------------
# 5. Construire la chaîne RetrievalQA
# ----------------------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",           # on bourre tout le contexte d’un coup
    retriever=graph_retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt}
)

# ----------------------------------------
# 6. Boucle interactive
# ----------------------------------------
if __name__ == "__main__":
    print("=== Chat GraphRAG (Neo4j + Albert via LangChain) ===")
    while True:
        question = input("\nPose ta question (ou « exit » pour quitter) : ")
        if question.lower().strip() in ("exit", "quit"):
            break

        # LangChain :
        #  1) graph_retriever.get_relevant_documents(question) → liste de Docs
        #  2) Concatène “context” = sommaire des docs, plus “question” dans le prompt
        #  3) Envoie tout à llm._call(prompt_final) → Albert → génération
        result = qa_chain.invoke({"query": question})
        print("\n📝 Réponse :")
        print(result["result"])