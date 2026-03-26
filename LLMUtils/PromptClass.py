from langchain_core.prompts import PromptTemplate

# ========================== PROMPT TEMPLATES ============================

class PromptTemplates:
    @classmethod
    def key_word_extraction(cls):
        prompt = """
        You are an intelligent assistant.
        Below is the information retrieved from the document:
        {context}
        Answer strictly based on above.
        Question: {question}
        """
        return PromptTemplate(template=prompt.strip(), input_variables=["context", "question"])

    @classmethod
    def chain_of_thoughts(cls):
        prompt = """
            You are a thoughtful assistant.

            Here is the document content:
            {context}

            Question: {question}

            Carefully analyze the content and provide a clear, well-reasoned answer based ONLY on the provided information.
            Make sure you cover all the points and no information is missed/

            Do NOT speculate or add information that is not supported by the text.
            {context}
            Question: {question}
            Think step by step based ONLY on the provided content.
            """
        return PromptTemplate(template=prompt.strip(), input_variables=["context", "question"])

    @classmethod
    def verification_prompt(cls):
        prompt = """
        You are a careful assistant.
        Here is the document content:
        {context}
        Question: {question}
        Verify if the answer is supported by the content.
        """
        return PromptTemplate(template=prompt.strip(), input_variables=["context", "question"])
    
    
class PromptManager:
    def __init__(self):
        self.prompt_dict = {
            "key_word_extraction": PromptTemplates.key_word_extraction,
            "chain_of_thoughts": PromptTemplates.chain_of_thoughts,
            "verification_prompt": PromptTemplates.verification_prompt
        }

    def get_prompt(self, name):
        try:
            func = self.prompt_dict.get(name)
            if not func:
                raise ValueError(f"Prompt '{name}' not found!")
            return func()
        except Exception as e:
            print(f"Error retrieving prompt: {e}")
            return None