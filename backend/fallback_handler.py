"""
Intelligent fallback response handler for queries below similarity threshold
Uses OpenAI's GPT models to generate contextual responses
"""
import os
from typing import Dict, Any, Optional
from openai import OpenAI
import time


class FallbackResponseHandler:
    """Handler for generating intelligent fallback responses using GPT models"""
    
    # System prompt template
    SYSTEM_PROMPT = """
        You are a smart assistant developed by Mount Sinai Hospital in Toronto with the primary goal of answering questions regarding sarcoma, a rare type of cancer. You need to give a brief and polite response. You are going to give a fallback response because the user's input did not match with any of the prepared responses.
        First, if user mentions any of the below symptoms, just tell them to call 911 immediately:
        - Fever over 100.4°F (38°C)
        - Severe pain that won't improve
        - Trouble breathing or chest pain
        - Uncontrolled bleeding
        - Signs of serious infection
        Otherwise, depending on user input, using your judgement your main task is to encourages user to do one or more of the following:
        - only ask about sarcoma related questions
        - rephrase their question in more detail and in complete sentence, and let them try including the word "sarcoma" if not already done so
        - if still don't have answer you want, please reach out to Mt Sinai Hospital (if it is hospital-related question) or Toronto Sarcoma Program website (torontosarcoma.ca) and the Sarcoma Cancer Foundation of Canada (if it is a sarcoma knowledge question)
        Some scenarios to consider:
        If they are asking very common knowledge or general questions about sarcoma (e.g. "what is sarcoma cancer"), then you can briefly answer.
        If user's question has typos, try your best to interpret with sarcoma, cancer, medical or hospital terminology.
        If user greets you, respond kindly.
        If user mentions any personal information, remind them to not share that information.
        If user asks for diagnosis, med changes, or personal instructions, then apologize, say you can only answer common questions about sarcoma and treatment of sarcoma by Sinai Health, encourage them to consult with medical professional for personalized advice.
        Here is the user question:
        {user_input}
        """
    
    def __init__(self, 
                 api_key: str,
                 model: str = "gpt-4o-mini",  # Using GPT-4o-mini as default (fast and cost-effective)
                 max_tokens: int = 300,
                 temperature: float = 0.7):
        """
        Initialize fallback response handler
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use (gpt-4o, gpt-4o-mini, gpt-3.5-turbo)
            max_tokens: Maximum tokens in response
            temperature: Response creativity (0-1)
        """
        if not api_key:
            raise ValueError("OpenAI API key is required for fallback handler")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Track usage for monitoring
        self.total_fallback_calls = 0
        self.total_tokens_used = 0
    
    def generate_fallback_response(self, 
                                  user_query: str,
                                  similarity_score: float,
                                  closest_match: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate an intelligent fallback response using GPT
        
        Args:
            user_query: The user's original query
            similarity_score: The similarity score that was below threshold
            closest_match: The closest matching question (if any)
            
        Returns:
            Dictionary with fallback response and metadata
        """
        try:
            start_time = time.time()
            
            # Format the prompt with user input
            formatted_prompt = self.SYSTEM_PROMPT.format(user_input=user_query)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant specializing in sarcoma information."},
                    {"role": "user", "content": formatted_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                n=1
            )
            
            # Extract response
            generated_answer = response.choices[0].message.content.strip()
            
            # Track usage
            self.total_fallback_calls += 1
            if response.usage:
                self.total_tokens_used += response.usage.total_tokens
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Build response in compatible format
            return {
                "answer": generated_answer,
                "source": "AI-Generated Fallback Response",  # Clear indication this is AI-generated
                "similarity_score": similarity_score,
                "matched_question": None,  # No match since it's a fallback
                "fallback_used": True,
                "fallback_model": self.model,
                "closest_match_score": similarity_score,
                "closest_question": closest_match,  # Include for debugging/monitoring
                "fallback_generation_time": round(processing_time, 3)
            }
            
        except Exception as e:
            print(f"Error generating fallback response: {str(e)}")
            
            # Return a safe default fallback if GPT fails
            return {
                "answer": self._get_default_fallback_message(),
                "source": "System Default Response",
                "similarity_score": similarity_score,
                "matched_question": None,
                "fallback_used": True,
                "fallback_error": str(e)
            }
    
    def _get_default_fallback_message(self) -> str:
        """
        Get default fallback message when GPT generation fails
        
        Returns:
            Safe default message
        """
        return (
            "I couldn't find a specific answer to your question about sarcoma. "
            "Please try rephrasing your question with more detail, or visit "
            "torontosarcoma.ca for more information. If you have urgent medical "
            "concerns, please contact your healthcare provider or call 911."
        )
    
    def check_emergency_keywords(self, user_query: str) -> Optional[str]:
        """
        Check for emergency keywords that need immediate response
        
        Args:
            user_query: User's input text
            
        Returns:
            Emergency message if keywords detected, None otherwise
        """
        query_lower = user_query.lower()
        
        # Emergency symptoms that need immediate attention
        emergency_keywords = [
            ("fever", "38"),
            ("severe pain", "won't improve"),
            ("trouble breathing", "chest pain"),
            ("uncontrolled bleeding",),
            ("serious infection",),
            ("emergency", "911"),
            ("urgent", "immediately")
        ]
        
        for keyword_group in emergency_keywords:
            if any(keyword in query_lower for keyword in keyword_group):
                return (
                    "⚠️ URGENT: If you're experiencing severe symptoms like high fever, "
                    "severe pain, trouble breathing, chest pain, or uncontrolled bleeding, "
                    "please call 911 immediately or go to your nearest emergency room."
                )
        
        return None
    
    def should_use_fallback(self, similarity_score: float, threshold: float) -> bool:
        """
        Determine if fallback response should be used
        
        Args:
            similarity_score: Calculated similarity score
            threshold: Configured threshold
            
        Returns:
            True if fallback should be used
        """
        return similarity_score < threshold
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for monitoring
        
        Returns:
            Dictionary with usage stats
        """
        return {
            "total_fallback_calls": self.total_fallback_calls,
            "total_tokens_used": self.total_tokens_used,
            "average_tokens_per_call": (
                self.total_tokens_used / self.total_fallback_calls 
                if self.total_fallback_calls > 0 else 0
            ),
            "model_used": self.model
        }
    
    def reset_usage_stats(self):
        """Reset usage statistics"""
        self.total_fallback_calls = 0
        self.total_tokens_used = 0
        print("Fallback handler usage stats reset")


class FallbackConfig:
    """Configuration for fallback response handling"""
    
    # Model selection (Cost and quality tradeoffs)
    # gpt-4o: Most capable, highest quality responses (~$2.50 per 1M input tokens)
    # gpt-4o-mini: Good balance of cost and quality (~$0.15 per 1M input tokens)  
    # gpt-3.5-turbo: Fastest and cheapest (~$0.50 per 1M input tokens)
    DEFAULT_MODEL = "gpt-4o-mini"  # Good balance of cost and quality
    
    # Response generation parameters
    MAX_TOKENS = 300  # Limit response length
    TEMPERATURE = 0.7  # Balance between consistency and creativity
    
    # Fallback triggers
    ENABLE_INTELLIGENT_FALLBACK = True  # Can be toggled via environment
    FALLBACK_THRESHOLD_BUFFER = 0.05  # Additional buffer below threshold
    
    # Emergency response config
    ENABLE_EMERGENCY_CHECK = True
    
    @classmethod
    def from_env(cls) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        return {
            "model": os.getenv("FALLBACK_MODEL", cls.DEFAULT_MODEL),
            "max_tokens": int(os.getenv("FALLBACK_MAX_TOKENS", cls.MAX_TOKENS)),
            "temperature": float(os.getenv("FALLBACK_TEMPERATURE", cls.TEMPERATURE)),
            "enable_intelligent_fallback": os.getenv(
                "ENABLE_INTELLIGENT_FALLBACK", 
                str(cls.ENABLE_INTELLIGENT_FALLBACK)
            ).lower() == "true",
            "enable_emergency_check": os.getenv(
                "ENABLE_EMERGENCY_CHECK",
                str(cls.ENABLE_EMERGENCY_CHECK)
            ).lower() == "true"
        }