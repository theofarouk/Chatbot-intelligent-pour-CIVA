from mistralai.client import MistralClient
from langchain_core.language_models import LLM
from langchain.schema import BaseRetriever, Document
from typing import Optional, List, Mapping, Any
from pydantic import BaseModel, PrivateAttr
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase






load_dotenv()


class MistralLangChainLLM(LLM, BaseModel):
    """
    Wrapper LangChain-compatible pour le SDK mistralai.
    """

    temperature: float = 0.2
    model_name: str = "mistral-small"
    
    # ✅ Ajout d'attributs privés compatibles Pydantic
    _api_key: str = PrivateAttr()
    _client: MistralClient = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._api_key = os.getenv("MISTRAL_API_KEY")
        if not self._api_key:
            raise ValueError("⚠️ La clé API MISTRAL_API_KEY est manquante dans l’environnement.")
        self._client = MistralClient(api_key=self._api_key)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self._client.chat(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "mistral"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
        }






class Neo4jRetriever(BaseRetriever):
    """
    Retriever LangChain qui interroge Neo4j pour ramener un sous-graphe pertinent.
    """

    # Déclare driver comme attribut privé afin que Pydantic ne l'exige pas comme champ
    _driver: Any = PrivateAttr()

    def __init__(self):
        # Appelle le constructeur de BaseModel
        super().__init__()

        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        pwd  = os.getenv("NEO4J_PWD")
        self._driver = GraphDatabase.driver(uri, auth=(user, pwd))

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        1) On extrait des tokens (mots) de la question,
        2) On interroge Neo4j pour chaque token correspondant
           à un nœud Entity.name,
        3) On construit un Document par relation trouvée.
        """
        # Tokenisation basique ; dans la vraie vie, on ferait un NER ou des lowercase+strip
        tokens = [tok.strip() for tok in query.split() if len(tok) > 1]
        seen_relations = set()
        docs: List[Document] = []

        with self._driver.session() as session:
            for tok in tokens:
                cypher = """
                MATCH (n:Entity {name: $name})-[r]-(m:Entity)
                RETURN n.name AS source, type(r) AS rel, m.name AS target
                """
                result = session.run(cypher, name=tok)
                for record in result:
                    src = record["source"]
                    rel = record["rel"]
                    tgt = record["target"]
                    triple_text = f"{src} {rel} {tgt}."
                    if triple_text not in seen_relations:
                        seen_relations.add(triple_text)
                        docs.append(Document(page_content=triple_text))

        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        # Pour la plupart des usages on peut renvoyer synchrone
        return self.get_relevant_documents(query)

    def __del__(self):
        try:
            self._driver.close()
        except:
            pass