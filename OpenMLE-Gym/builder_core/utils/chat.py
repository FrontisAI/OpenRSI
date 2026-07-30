from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from openmle_gym.llm_config import build_llm_config

class OpenAILLMProvider:
    def __init__(self, api_key=None, base_url=None, model=None, tools=None, stream=True):
        load_dotenv()
        config = build_llm_config()

        self.api_key = api_key or config.api_key
        self.base_url = base_url or config.base_url
        self.model = model or config.model
        self.stream = stream
        
        self.llm = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            max_tokens=5000
        ).bind_tools(tools or [])
        
        self.outlen = 200
        
    def query(self, prompts):
        try:
            if not self.stream:
                response = self.llm.invoke(prompts)
                message = response.content
                print(f"\n[OK] {self.model} response received")
                print(message[:self.outlen]+"..." if len(message)>self.outlen else message)
                return response
            
            print(f"\n[LLM] {self.model} is generating a response:\n" + "-" * 30)
            full_response = None
            for chunk in self.llm.stream(prompts):
                if chunk.content:
                    print(chunk.content, end="", flush=True)
                if full_response is None:
                    full_response = chunk
                else:
                    full_response += chunk
            print(f"\n[OK] {self.model} response received")
            return full_response

        except Exception as e:
            error_msg = str(e).lower()
            print(f"\n[ERROR] {self.model} call failed: {e}")
            
            if "rate limit" in error_msg or "429" in error_msg:
                raise RuntimeError("LLM API error: rate limit exceeded") from e
            elif "authentication" in error_msg or "401" in error_msg:
                raise RuntimeError("LLM API error: invalid API Key") from e
            elif "not found" in error_msg or "404" in error_msg:
                raise RuntimeError(f"LLM API error: model '{self.model}' not found") from e
            elif "timeout" in error_msg:
                raise RuntimeError("LLM API error: request timed out") from e
            elif "connection" in error_msg:
                raise RuntimeError("LLM API error: connection failed") from e
            elif "403" in error_msg:
                raise RuntimeError("LLM API error: insufficient balance") from e
            else:
                raise RuntimeError(f"LLM API error: {e}") from e
