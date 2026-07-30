from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from openmle_gym.llm_config import eval_llm_config

class OpenAILLMProvider:
    def __init__(self, api_key=None, base_url=None, model=None):

        load_dotenv()
        config = eval_llm_config()

        self.api_key = api_key or config.api_key
        self.base_url = base_url or config.base_url
        self.model = model or config.model
        
        self.llm = ChatOpenAI(
            model = self.model,
            api_key = self.api_key,
            base_url = self.base_url,
            max_tokens = 2048
        )
        
    def query(self, prompts: list[tuple]) -> str:
        try:
            response = self.llm.invoke(prompts)
            message = response.content
            print(f"\n[OK] {self.model} response received")
            print(message[:100]+"..." if len(message)>100 else message)
            return response
        except Exception as e:
            print(f"\n[ERROR] {self.model} call failed: {e}")
            
            # Raise clearer high-level API errors for common failure modes.
            if "rate limit" in str(e).lower() or "429" in str(e):
                raise RuntimeError("LLM API error: rate limit exceeded") from e
            elif "authentication" in str(e).lower() or "401" in str(e):
                raise RuntimeError("LLM API error: invalid API Key") from e
            elif "not found" in str(e).lower() or "404" in str(e):
                raise RuntimeError(f"LLM API error: model '{self.model}' not found or unavailable") from e
            elif "timeout" in str(e).lower():
                raise RuntimeError("LLM API error: request timed out") from e
            elif "connection" in str(e).lower():
                raise RuntimeError("LLM API error: connection timed out") from e
            elif "403" in str(e):
                raise RuntimeError("LLM API error: insufficient balance") from e
            else:
                raise RuntimeError(f"LLM API error: {e}") from e
