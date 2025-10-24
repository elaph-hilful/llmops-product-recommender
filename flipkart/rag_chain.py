from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Import from langchain-core instead of langchain.chains
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from flipkart.config import Config


class RagChainBuilder:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.model = ChatGroq(model=Config.RAG_MODEL, temperature=0.5)
        self.history_store = {}

    def __get_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]

    def build_chain(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        # Contextualize question prompt
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # QA prompt
        qa_system_prompt = """You're an e-commerce bot answering product-related queries using reviews and titles.
        Stick to context. Be concise and helpful.

        {context}"""
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # Create the contextualize chain
        contextualize_q_chain = contextualize_q_prompt | self.model | StrOutputParser()

        def contextualized_question(input_dict):
            if input_dict.get("chat_history"):
                return contextualize_q_chain
            else:
                return input_dict["input"]

        # Create RAG chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: format_docs(retriever.invoke(contextualized_question(x)))
            )
            | qa_prompt
            | self.model
            | StrOutputParser()
        )

        # Wrap with message history
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.__get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        # Wrap to return dict with 'answer' key
        def chain_with_answer_key(input_dict, config):
            response = conversational_rag_chain.invoke(input_dict, config=config)
            return {"answer": response}

        return lambda input_dict, config: chain_with_answer_key(input_dict, config)