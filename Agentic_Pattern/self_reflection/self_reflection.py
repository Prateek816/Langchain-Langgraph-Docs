import os
from colorama import Fore
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# --- Prompts (Preserved exactly as requested) ---
BASE_GENERATION_SYSTEM_PROMPT = """
Your task is to Generate the best content possible for the user's request.
If the user provides critique, respond with a revised version of your previous attempt.
You must always output the revised content.
"""

BASE_REFLECTION_SYSTEM_PROMPT = """
You are tasked with generating critique and recommendations to the user's generated content.
If the user content has something wrong or something to be improved, output a list of recommendations
and critiques. If the user content is ok and there's nothing to change, output this: <OK>
"""

class ReflectionAgent:
    """
    A class that implements a Reflection Agent using LangChain and Groq.
    It iteratively improves content through a generation-reflection loop.
    """

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        # Initializing the LangChain ChatGroq model
        self.llm = ChatGroq(
            model=model,
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.model_name = model

    def _request_completion(
        self,
        messages: list,
        verbose: int = 0,
        log_title: str = "COMPLETION",
        log_color: str = "",
    ):
        """Standardizes the LLM call using LangChain's invoke method."""
        response = self.llm.invoke(messages)
        output = response.content

        if verbose > 0:
            print(log_color, f"\n\n{log_title}\n\n", output)

        return output

    def generate(self, generation_history: list, verbose: int = 0) -> str:
        """Generates content based on history."""
        return self._request_completion(
            generation_history, verbose, log_title="GENERATION", log_color=Fore.BLUE
        )

    def reflect(self, reflection_history: list, verbose: int = 0) -> str:
        """Generates critique based on the last generation."""
        return self._request_completion(
            reflection_history, verbose, log_title="REFLECTION", log_color=Fore.GREEN
        )

    def _manage_history(self, history: list, new_message: any, max_length: int = 3):
        """
        Mimics FixedFirstChatHistory logic: 
        Keeps the first message (System Prompt) and limits total history.
        """
        history.append(new_message)
        if len(history) > max_length:
            # Keep index 0 (System), remove index 1 (the oldest non-system message)
            history.pop(1)
        return history

    def run(
        self,
        user_msg: str,
        generation_system_prompt: str = "",
        reflection_system_prompt: str = "",
        n_steps: int = 10,
        verbose: int = 0,
    ) -> str:
        """Runs the iterative Generate -> Reflect loop."""
        
        # Combine provided prompts with base prompts
        gen_sys_content = generation_system_prompt + BASE_GENERATION_SYSTEM_PROMPT
        ref_sys_content = reflection_system_prompt + BASE_REFLECTION_SYSTEM_PROMPT

        # Initialize Histories as lists of LangChain Message objects
        generation_history = [
            SystemMessage(content=gen_sys_content),
            HumanMessage(content=user_msg)
        ]

        reflection_history = [
            SystemMessage(content=ref_sys_content)
        ]

        last_generation = ""

        for step in range(n_steps):
            if verbose > 0:
                print(Fore.WHITE + f"\n--- STEP {step + 1}/{n_steps} ---")

            # 1. GENERATION STEP
            last_generation = self.generate(generation_history, verbose=verbose)
            
            # Update histories with the new generation
            self._manage_history(generation_history, AIMessage(content=last_generation))
            self._manage_history(reflection_history, HumanMessage(content=last_generation))

            # 2. REFLECTION STEP
            critique = self.reflect(reflection_history, verbose=verbose)

            # Check for Stop Sequence
            if "<OK>" in critique:
                if verbose > 0:
                    print(Fore.RED, "\n\nStop Sequence found. Stopping the reflection loop ... \n\n")
                break

            # Update histories with the critique for the next iteration
            self._manage_history(generation_history, HumanMessage(content=critique))
            self._manage_history(reflection_history, AIMessage(content=critique))

        return last_generation

# Example Usage:
if __name__ == "__main__":
    agent = ReflectionAgent()
    final_output = agent.run(
        user_msg="Write a quicksort implementation in Python.",
        verbose=1,
        n_steps=3
    )
    print(Fore.CYAN + "\nFINAL OUTPUT:\n", final_output)