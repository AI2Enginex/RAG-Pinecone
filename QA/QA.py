from LLMUtils.LLMConfigs import ChatGoogleGENAI, GeminiConfig, QAState, api_key
from langgraph.graph import StateGraph
from LLMUtils.PromptClass import PromptManager
from LLMUtils.TextProcessing import PrepareText



# ========================== QASYSTEM ============================

# class to decide the behaviour of the agent
class QASystem(PrepareText, ChatGoogleGENAI):
    """
    Agentic RAG system:
    - Agent decides what to do
    - Tools perform retrieval / answering / verification
    """

    def __init__(self, file_path: str, config=None, separator=None, chunk_size=None, overlap=None):

        try:

            # Load document and embeddings
            PrepareText.__init__(self, file_paths=file_path, config=config)

            # initialize ChatGoogleGENAI 
            ChatGoogleGENAI.__init__(self, config=config)
            
            # calling the preprocessed vectors
            self.vector_store = self.create_text_vectors(
                separator=separator,
                chunksize=chunk_size,
                overlap=overlap
            )
            
            # create FAISS as Retriever
            self.retriever = None
            if self.vector_store:
                self.retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": 40}
                )

            print("Agentic QA System initialized")

        except Exception as e:
            print(f"Error initializing QASystem: {e}")
            self.vector_store = None

    def retrieve_chunks(self, state: QAState):
        try:
            if not self.vector_store:
                print("Vector store not initialized!")
                return state

            docs = self.retriever.invoke(state["question"])
            state["retrieved_chunks"] = [doc.page_content for doc in docs]
            return state

        except Exception as e:
            print(f"Error retrieving chunks: {e}")
        return state
    

    # method to normalize
    # the LLM output
    def normalize_llm_output(self, content):
        """
        Converts Gemini response content into plain text.
        Handles str, list[str], list[dict], and mixed cases.
        """
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    
                    if "text" in item:
                        texts.append(item["text"])
                    else:
                        texts.append(str(item))
                else:
                    texts.append(str(item))
            return " ".join(texts)

        return str(content)
    

    def answer_questions(self, state: QAState):
        try:
            if not self.llm:
                print("LLM not initialized!")
                return state

            # ensure prompt_type exists
            if not state.get("prompt_type"):
                state["prompt_type"] = "key_word_extraction"

            prompt_manager = PromptManager()
            prompt_template = prompt_manager.get_prompt(state["prompt_type"])
            if not prompt_template:
                print("Prompt template not found!")
                return state

            context = "\n\n".join(state["retrieved_chunks"])
            prompt = prompt_template.format(
                context=context,
                question=state["question"]
            )

            response = self.llm.invoke(prompt)
            
            # normalizing the LLM response
            if response:
                state["answer"] = self.normalize_llm_output(response.content)
            else:
                state["answer"] = ""


            return state

        except Exception as e:
            print(f"Error answering question: {e}")
            return state
    
    
    def verify_answer(self, state: QAState):
        try:
            prompt_manager = PromptManager()
            prompt_template = prompt_manager.get_prompt("verification_prompt")

            context = "\n\n".join(state["retrieved_chunks"])
            prompt = prompt_template.format(
                context=context,
                question=state["question"]
            )

            response = self.llm.invoke(prompt)

            # normalization
            verification_text = self.normalize_llm_output(
                response.content
            ).lower()

            if "not supported" in verification_text or "not found" in verification_text:
                state["answer"] = (
                    "The document does not provide sufficient evidence "
                    "to answer this question."
                )

            return state

        except Exception as e:
            print(f"Error verifying answer: {e}")
            return state

            
    def agent_think(self, state: QAState):
        """
        Agent decides:
        - whether to retrieve
        - which prompt to use for answering
        - whether to verify
        - when to stop
        """

        try:
            question = state["question"].lower()

            # Need retrieval
            if not state["retrieved_chunks"]:
                state["next_action"] = "retrieve"
                return state

            # Need to answer
            if not state["answer"]:
                # PROMPT SELECTION LOGIC
                if any(word in question for word in [
                    "what is", "who is", "define", "list", "when", "where"
                ]):
                    state["prompt_type"] = "key_word_extraction"
                else:
                    state["prompt_type"] = "chain_of_thoughts"

                state["next_action"] = "answer"
                return state
            print(f"[AGENT] Selected prompt type: {state['prompt_type']}")

            # Verify once
            if not state["verified"]:
                state["next_action"] = "verify"
                state["verified"] = True
                return state

            # Done
            state["next_action"] = "end"
            return state
        except Exception as e:
            return e




# ========================== GRAPH EXECUTION ============================

class QASystemGraphExecution(QASystem):
    """Builds a LangGraph execution graph to automate QA workflow."""

    def __init__(self, file_path: str, config=None,
                 separator=None, chunk_size=None, overlap=None):
        try:
            super().__init__(file_path=file_path, config=config,
                             separator=separator, chunk_size=chunk_size, overlap=overlap)
        except Exception as e:
            print(f"Error initializing GraphExecution: {e}")

    def build_graph(self):
        try:
           
            graph = StateGraph(QAState)

            graph.add_node("agent", self.agent_think)
            graph.add_node("retrieve", self.retrieve_chunks)
            graph.add_node("answer_node", self.answer_questions)
            graph.add_node("verify", self.verify_answer)

            graph.set_entry_point("agent")

            graph.add_conditional_edges(
                "agent",
                lambda state: state["next_action"],
                {
                    "retrieve": "retrieve",
                    "answer": "answer_node",
                    "verify": "verify",
                    "end": "__end__"
                }
            )

            graph.add_edge("retrieve", "agent")
            graph.add_edge("answer_node", "agent")
            graph.add_edge("verify", "agent")

            return graph
        except Exception as e:
            print(f"Error building graph: {e}")
            return None

    def answer(self, question: str):
        """Executes the graph to answer the question."""
        try:
            graph_executor = self.build_graph()
            if not graph_executor:
                print("Graph not built correctly!")
                return None

            executor = graph_executor.compile()
            initial_state: QAState = {
            "question": question,
            "retrieved_chunks": [],
            "answer": "",
            "prompt_type": None,
            "next_action": None,
             "verified": False }
            result = executor.invoke(initial_state)
            return result.get("answer", "")
        except Exception as e:
            print(f"Error executing graph: {e}")
            return None


# ========================== MAIN EXECUTION ============================

if __name__ == "__main__":
    try:
        config = GeminiConfig(
            chat_model_name="gemini-3-flash-preview",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            temperature=0,
            top_p=0.8,
            top_k=32,
            max_output_tokens=15000,
            generation_max_tokens=20000,
            api_key=api_key  # Set your key here or via environment variable
        )

        file_path = [
        "E:/Tender Project/MCA.pdf",
        "E:/Tender Project/MCA1.pdf",
        "E:/Tender Project/MCA2.pdf"]

        question = input("Ask your question here: ")

        qa_system = QASystemGraphExecution(
            file_path=file_path,
            config=config,
            separator=["\n\n", "\n", " ", ""],
            chunk_size=1500,
            overlap=250
        )

        answer = qa_system.answer(question=question)
        print("\nQuestion:", question)
        print("Answer:", answer)

    except Exception as e:
        print(f"Error in main execution: {e}")