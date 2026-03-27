from LLMUtils.LLMConfigs import ChatGoogleGENAI, GeminiConfig, QAState, api_key
from langgraph.graph import StateGraph
from LLMUtils.PromptClass import PromptManager
from LLMUtils.TextProcessing import PrepareText
from LLMUtils.TextProcessing import RetrieverService


# ========================== QASYSTEM ============================

class QASystem(PrepareText, ChatGoogleGENAI):

    def __init__(self, file_path: str, user_id: int,config=None, separator=None, chunk_size=None, overlap=None):

        try:
            # Initialize LLM
            ChatGoogleGENAI.__init__(self, config=config)

           
            self.service = RetrieverService(
                file_paths=file_path,
                user_id=user_id,
                config=config,
                api_key=api_key
            )

            self.retriever = self.service.get_retriever(
                chunk=chunk_size,
                overlap=overlap,
                sep=separator,
                batch_size=10
            )

            print("Agentic QA System initialized with RetrieverService")

        except Exception as e:
            print(f"Error initializing QASystem: {e}")
            self.retriever = None


    def retrieve_chunks(self, state: QAState):
        try:
            if not self.retriever:
                print("Retriever not initialized!")
                return state

            
            docs = self.retriever.invoke(state["question"])

            state["retrieved_chunks"] = [
                doc.page_content if hasattr(doc, "page_content") else str(doc)
                for doc in docs
            ]

            return state

        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return state


    def normalize_llm_output(self, content):
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
        try:
            question = state["question"].lower()

            if not state["retrieved_chunks"]:
                state["next_action"] = "retrieve"
                return state

            if not state["answer"]:
                if any(word in question for word in [
                    "what is", "who is", "define", "list", "when", "where"
                ]):
                    state["prompt_type"] = "key_word_extraction"
                else:
                    state["prompt_type"] = "chain_of_thoughts"

                state["next_action"] = "answer"
                return state

            print(f"[AGENT] Selected prompt type: {state['prompt_type']}")

            if not state["verified"]:
                state["next_action"] = "verify"
                state["verified"] = True
                return state

            state["next_action"] = "end"
            return state

        except Exception as e:
            return e


# ========================== GRAPH EXECUTION ============================

class QASystemGraphExecution(QASystem):

    def __init__(self, file_path: str, userid: int,config=None,
                 separator=None, chunk_size=None, overlap=None):
        try:
            super().__init__(
                file_path=file_path,
                user_id=userid,
                config=config,
                separator=separator,
                chunk_size=chunk_size,
                overlap=overlap
            )
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
                "verified": False
            }

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
            temperature=0.4,
            top_p=0.8,
            top_k=32,
            max_output_tokens=15000,
            generation_max_tokens=20000,
            api_key=api_key
        )

        file_path = [
            "E:/RAG-QA/MCA2.pdf",
            "E:/RAG-QA/MCA.pdf",
            "E:/RAG-QA/MCA1.pdf",
            "E:/RAG-QA/data.pdf"]

        qa_system = QASystemGraphExecution(
            file_path=file_path,
            userid=1,
            config=config,
            separator=["\n\n", "\n", " ", ""],
            chunk_size=1500,
            overlap=250
        )
        
        question = input("Ask your question here: ")
        answer = qa_system.answer(question=question)

        print("\nQuestion:", question)
        print("Answer:", answer)

    except Exception as e:
        print(f"Error in main execution: {e}")