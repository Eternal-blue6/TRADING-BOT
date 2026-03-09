"""
Gemini AI Reasoning Engine v1.0
================================
Alternative to Claude - uses Google's Gemini instead!

Why Gemini?
- FREE tier (15 requests/minute)
- Very capable for trading analysis
- Fast responses
- Easy to set up

Setup:
1. Get free API key: https://makersuite.google.com/app/apikey
2. Set environment variable: GEMINI_API_KEY=your_key_here
3. pip install google-generativeai
"""

import os
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURATION
# ==========================================
class GeminiConfig:
    """Gemini AI configuration"""
    
    API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Model - Gemini 1.5 Pro is best for trading
    MODEL = "gemini-1.5-pro-latest"  # Latest and best
    # Alternative: "gemini-1.5-flash" for faster responses
    
    # Generation settings
    MAX_OUTPUT_TOKENS = 2048
    TEMPERATURE = 0.7  # 0 = deterministic, 1 = creative
    
    # Cost tracking (Gemini is FREE up to limits!)
    FREE_TIER_LIMIT_PER_MINUTE = 15
    FREE_TIER_LIMIT_PER_DAY = 1500


def explain_decision(signal, asset, risk_profile="moderate", style="expert", extra_context=None):
    """
    Ask Gemini AI to explain a trading decision with reasoning, risks, opportunities,
    and structured JSON output for programmatic use.
    """
    prompt = f"""
    You are a smart, curious trading expert who calculates every move carefully.
    Speak in clear, normal English, as if explaining to an experienced investor.

    Signal: {signal}
    Asset: {asset}
    Risk Profile: {risk_profile}
    Style: {style}

    Guidelines:
    -if something goes sideways stop and re-plan immediately - dont keep pushing
    -use plan mode for verification steps not just building 
    -offload research,exploration,and parallel analysis to sub agents
    -for more complex problem or scenarios throw  more compte at it .
    -after any correction :update
    -write rules for yourself that prevent same mistake.
    - Investment focus: Focus on crypto, tech stocks, and renewable energy.
    - Income generation: Suggest passive income strategies alongside trading.
    -ruthless iterate untill false positives or mistake rate drops
    - Scenario analysis: Explain best case, worst case, and most likely case.
    - Time horizon: Give advice for short-term (days), medium-term (months), and long-term (years).
    - Educational mode: Teach me the logic behind each indicator as you explain.
    - Explain reasoning step by step.
    - Highlight risks and possible losses.
    - Suggest alternative strategies and new income opportunities.
    - Stay up to date with the latest beneficial investments.
    - Consider multiple scenarios before giving a plan.
    - Be open to creative ideas but remain logical.

    At the end of your explanation, also provide a structured JSON object with fields:
    {{
      "decision": "BUY/SELL/HOLD",
      "reasoning": "summary of logic",
      "risks": ["list of risks"],
      "opportunities": ["list of opportunities"],
      "scenarios": {{
        "best_case": "description",
        "worst_case": "description",
        "most_likely": "description"
      }},
      "time_horizon": {{
        "short_term": "advice",
        "medium_term": "advice",
        "long_term": "advice"
      }},
      "education": "explanation of indicators used"
    }}
    """

    response = requests.post(
        "https://api.gemini.com/v1/messages",  # adjust to Gemini’s actual endpoint
        headers={
            "x-api-key": API_KEY,
            "content-type": "application/json"
        },
        json={
            "model": "gemini-1.5-pro",  # adjust to the Gemini model you’re using
            "max_tokens": 800,
            "messages": [{"role": "user", "content": prompt}]
        }
    )

    text_output = response.json()["content"][0]["text"]

    # Try to parse JSON block from AI output
    try:
        json_start = text_output.find("{")
        json_end = text_output.rfind("}") + 1
        reasoning_json = json.loads(text_output[json_start:json_end])
    except Exception:
        reasoning_json = {"decision": signal, "reasoning": "AI explanation not parsed", "risks": [], "opportunities": []}

    return text_output, reasoning_json







# ==========================================
# INITIALIZE GEMINI
# ==========================================
try:
    import google.generativeai as genai
    
    if GeminiConfig.API_KEY:
        genai.configure(api_key=GeminiConfig.API_KEY)
        GEMINI_AVAILABLE = True
        logger.info("✅ Gemini AI initialized")
    else:
        GEMINI_AVAILABLE = False
        logger.warning("⚠️ GEMINI_API_KEY not set - Gemini unavailable")
        
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("⚠️ google-generativeai not installed. Run: pip install google-generativeai")


# ==========================================
# RESPONSE CACHE (same as Claude version)
# ==========================================
class ResponseCache:
    """Simple in-memory cache"""
    
    def __init__(self):
        self.cache = {}
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Dict]:
        if key in self.cache:
            response, timestamp = self.cache[key]
            age = (datetime.now() - timestamp).total_seconds()
            
            if age < 300:  # 5 minutes
                self.hits += 1
                return response
            else:
                del self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, key: str, response: Dict):
        self.cache[key] = (response, datetime.now())


response_cache = ResponseCache()


# ==========================================
# USAGE TRACKER (Gemini has rate limits!)
# ==========================================
class UsageTracker:
    """Track Gemini usage to avoid rate limits"""
    
    def __init__(self):
        self.requests_this_minute = []
        self.requests_today = 0
        self.last_reset = datetime.now().date()
    
    def can_make_request(self) -> tuple[bool, str]:
        """Check if we can make a request"""
        now = datetime.now()
        
        # Reset daily counter
        if now.date() != self.last_reset:
            self.requests_today = 0
            self.last_reset = now.date()
        
        # Clean old minute requests
        cutoff = now - timedelta(minutes=1)
        self.requests_this_minute = [
            ts for ts in self.requests_this_minute if ts > cutoff
        ]
        
        # Check limits
        if len(self.requests_this_minute) >= GeminiConfig.FREE_TIER_LIMIT_PER_MINUTE:
            return False, "Rate limit: 15 requests per minute exceeded"
        
        if self.requests_today >= GeminiConfig.FREE_TIER_LIMIT_PER_DAY:
            return False, "Rate limit: 1500 requests per day exceeded"
        
        return True, "OK"
    
    def record_request(self):
        """Record a request"""
        self.requests_this_minute.append(datetime.now())
        self.requests_today += 1


usage_tracker = UsageTracker()


# ==========================================
# MAIN GEMINI REASONING FUNCTION
# ==========================================
def explain_decision_with_gemini(
    signal: str,
    asset: str,
    analysis_data: Optional[Dict] = None,
    risk_profile: str = "moderate",
    style: str = "expert",
    extra_context: Optional[str] = None,
    use_cache: bool = True
) -> Dict:
    """
    Get AI explanation using Gemini instead of Claude.
    
    SAME interface as Claude version - just swap the function!
    
    Args:
        signal: Trading signal (BUY/SELL/HOLD)
        asset: Asset being analyzed
        analysis_data: Full analysis dict from strategy engine
        risk_profile: User's risk tolerance
        style: Communication style
        extra_context: Additional context
        use_cache: Whether to use response caching
    
    Returns:
        Dict with AI explanation and structured analysis
    """
    # Check if Gemini available
    if not GEMINI_AVAILABLE:
        return _error_response("Gemini not available. Install: pip install google-generativeai")
    
    # Check rate limits
    can_request, limit_msg = usage_tracker.can_make_request()
    if not can_request:
        return _error_response(f"Rate limit: {limit_msg}")
    
    # Create cache key
    cache_key = f"gemini_{signal}_{asset}_{risk_profile}"
    if analysis_data:
        cache_key += f"_{analysis_data.get('score', 0)}"
    
    # Check cache
    if use_cache:
        cached = response_cache.get(cache_key)
        if cached:
            logger.info("✅ Using cached Gemini response")
            return cached
    
    # Build prompt (same structure as Claude version)
    prompt = _build_prompt(signal, asset, analysis_data, risk_profile, style, extra_context)
    
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel(
            model_name=GeminiConfig.MODEL,
            generation_config={
                "max_output_tokens": GeminiConfig.MAX_OUTPUT_TOKENS,
                "temperature": GeminiConfig.TEMPERATURE,
            }
        )
        
        logger.info(f"🤖 Calling Gemini API ({GeminiConfig.MODEL})...")
        
        # Generate response
        response = model.generate_content(prompt)
        
        # Record usage
        usage_tracker.record_request()
        
        # Parse response
        text = response.text
        
        logger.info(f"✅ Gemini response received")
        
        # Extract structured data
        structured_data = _extract_json(text)
        
        # Build result
        result = {
            "success": True,
            "explanation": text,
            "structured": structured_data,
            "metadata": {
                "signal": signal,
                "asset": asset,
                "timestamp": datetime.now().isoformat(),
                "model": GeminiConfig.MODEL,
                "provider": "Gemini"
            }
        }
        
        # Add analysis data if provided
        if analysis_data:
            result["technical_analysis"] = analysis_data
        
        # Cache result
        if use_cache:
            response_cache.set(cache_key, result)
        
        return result
        
    except Exception as e:
        logger.error(f"Gemini API error: {e}", exc_info=True)
        return _error_response(str(e))


def _build_prompt(signal, asset, analysis_data, risk_profile, style, extra_context):
    """Build prompt for Gemini (same as Claude version)"""
    
    # Extract key data
    analysis_summary = ""
    if analysis_data:
        analysis_summary = f"""

TECHNICAL ANALYSIS DATA:
- Signal: {analysis_data.get('signal', 'UNKNOWN')}
- Confidence: {analysis_data.get('confidence', 0)}%
- Score: {analysis_data.get('score', 0)}/10
- Current Price: ${analysis_data.get('levels', {}).get('current_price', 'N/A')}

Trend: {analysis_data.get('trend', {}).get('direction', 'UNKNOWN')} (Strength: {analysis_data.get('trend', {}).get('strength', 0)}/10)
Momentum: {analysis_data.get('momentum', {}).get('state', 'UNKNOWN')} (RSI: {analysis_data.get('momentum', {}).get('indicators', {}).get('rsi', 'N/A')})
Volume: {analysis_data.get('volume', {}).get('status', 'UNKNOWN')} ({analysis_data.get('volume', {}).get('ratio', 1.0):.2f}x average)
Risk: {analysis_data.get('risk_assessment', {}).get('level', 'UNKNOWN')}

Reasoning:
{chr(10).join('- ' + r for r in analysis_data.get('reasoning', []))}
"""
    
    prompt = f"""You are Whitney's expert trading advisor specializing in crypto, stocks, and trading. You're smart, curious, and calculate every move carefully.

EXPERTISE:
- Deep knowledge of technical analysis and market dynamics
- Expertise in risk management and position sizing
- Focus on high-probability setups with clear risk/reward
- Specialization in crypto, tech stocks, and renewable energy
- Awareness of market manipulation and scams
- Both short-term trading and long-term investing strategies

YOUR TASK:
Analyze this trading opportunity and provide expert guidance.

ASSET: {asset}
SIGNAL: {signal}
RISK PROFILE: {risk_profile}
{analysis_summary}
{f'ADDITIONAL CONTEXT: {extra_context}' if extra_context else ''}

PROVIDE:

1. **Clear Recommendation**: Should Whitney take this trade? Why or why not?

2. **Risk Analysis**: Be honest about risks
   - What could go wrong?
   - How likely is it?
   - What are the warning signs?

3. **Opportunity Analysis**: What's the upside?
   - Best case scenario
   - Most likely outcome
   - Risk/reward ratio

4. **Time Horizons**:
   - Short-term (days): 
   - Medium-term (months):
   - Long-term (years):

5. **Position Sizing**: Recommended % of portfolio and why

6. **Entry/Exit Strategy**:
   - Entry price and why
   - Stop loss placement and why
   - Take profit targets

7. **Key Insights**: What should Whitney watch for?

8. **Educational Note**: What can be learned from this setup?

At the end, provide a JSON object with this EXACT structure:

{{
  "decision": "BUY" or "SELL" or "HOLD" or "WAIT",
  "confidence_pct": 0-100,
  "reasoning_summary": "concise summary",
  "risks": ["risk 1", "risk 2", "risk 3"],
  "opportunities": ["opportunity 1", "opportunity 2"],
  "scenarios": {{
    "best_case": "description",
    "worst_case": "description",
    "most_likely": "description"
  }},
  "time_horizon": {{
    "short_term": "advice",
    "medium_term": "advice",
    "long_term": "advice"
  }},
  "position_sizing": "X% of portfolio",
  "entry_strategy": "entry approach",
  "exit_strategy": "exit approach",
  "key_insights": ["insight 1", "insight 2"],
  "educational_note": "key learning"
}}
"""
    
    return prompt


def _extract_json(text: str) -> Optional[Dict]:
    """Extract JSON from Gemini response"""
    import re
    
    # Try to find JSON block
    patterns = [
        r'\{[^{]*"decision"[^}]*\}',  # Simple JSON
        r'```json\s*(\{.*?\})\s*```',  # Markdown code block
        r'```\s*(\{.*?\})\s*```',      # Generic code block
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                json_str = match.group(1) if match.groups() else match.group(0)
                return json.loads(json_str)
            except:
                continue
    
    # Try parsing entire text
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except:
        pass
    
    logger.warning("Could not extract JSON from Gemini response")
    return None


def _error_response(error_message: str) -> Dict:
    """Return error response"""
    return {
        "success": False,
        "explanation": f"Error: {error_message}",
        "structured": {
            "decision": "ERROR",
            "confidence_pct": 0,
            "reasoning_summary": f"AI analysis failed: {error_message}",
            "risks": ["Unable to analyze"],
            "opportunities": []
        },
        "error": error_message,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "provider": "Gemini"
        }
    }


# ==========================================
# STATISTICS
# ==========================================
def get_gemini_stats() -> Dict:
    """Get Gemini usage statistics"""
    return {
        "provider": "Gemini",
        "model": GeminiConfig.MODEL,
        "available": GEMINI_AVAILABLE,
        "cache_stats": {
            "hits": response_cache.hits,
            "misses": response_cache.misses,
            "hit_rate_pct": (response_cache.hits / (response_cache.hits + response_cache.misses) * 100) if (response_cache.hits + response_cache.misses) > 0 else 0
        },
        "usage": {
            "requests_this_minute": len(usage_tracker.requests_this_minute),
            "requests_today": usage_tracker.requests_today,
            "limit_per_minute": GeminiConfig.FREE_TIER_LIMIT_PER_MINUTE,
            "limit_per_day": GeminiConfig.FREE_TIER_LIMIT_PER_DAY
        }
    }


# ==========================================
# COMPARISON: CLAUDE VS GEMINI
# ==========================================
def compare_providers():
    """
    Compare Claude vs Gemini for trading bot
    """
    comparison = """
    🤖 CLAUDE vs GEMINI COMPARISON
    
    CLAUDE (Anthropic):
    ✅ Best reasoning quality
    ✅ Excellent at complex analysis
    ✅ More conservative/careful recommendations
    ✅ Better at explaining "why not to trade"
    ❌ Costs money ($3-15/million tokens)
    ❌ Lower rate limits on free tier
    
    GEMINI (Google):
    ✅ FREE tier (15 req/min, 1500/day)
    ✅ Very fast responses
    ✅ Good at analysis
    ✅ Easy integration
    ⚠️ Slightly less sophisticated reasoning
    ⚠️ Can be more "optimistic" (might suggest trades more often)
    
    RECOMMENDATION:
    - Start with GEMINI (it's free!)
    - If you need better reasoning, upgrade to Claude
    - Or use BOTH: Gemini for quick checks, Claude for final decisions
    """
    
    print(comparison)


# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    """Test Gemini integration"""
    
    print("\n" + "="*60)
    print("GEMINI AI TRADING ADVISOR - TEST")
    print("="*60)
    
    if not GEMINI_AVAILABLE:
        print("\n❌ Gemini not available!")
        print("\nSetup instructions:")
        print("1. Get API key: https://makersuite.google.com/app/apikey")
        print("2. Set environment variable: export GEMINI_API_KEY=your_key")
        print("3. Install package: pip install google-generativeai")
        print("="*60)
    else:
        # Test request
        sample_analysis = {
            "signal": "BUY",
            "confidence": 78.5,
            "score": 7.2,
            "levels": {"current_price": 45231.50},
            "trend": {"direction": "UPTREND", "strength": 7},
            "momentum": {"state": "BULLISH", "indicators": {"rsi": 35.2}},
            "volume": {"status": "HIGH", "ratio": 1.8},
            "reasoning": ["Strong uptrend", "Oversold RSI", "High volume"]
        }
        
        result = explain_decision_with_gemini(
            signal="BUY",
            asset="BTC/USDT",
            analysis_data=sample_analysis,
            risk_profile="moderate"
        )
        
        if result["success"]:
            print("\n✅ Gemini Response:")
            print("\n" + result["explanation"][:500] + "...")
            
            if result.get("structured"):
                print("\n📊 Structured Analysis:")
                print(json.dumps(result["structured"], indent=2))
        else:
            print(f"\n❌ Error: {result.get('error')}")
        
        # Show stats
        print("\n📊 Usage Statistics:")
        stats = get_gemini_stats()
        print(json.dumps(stats, indent=2))
        
        print("\n" + "="*60)