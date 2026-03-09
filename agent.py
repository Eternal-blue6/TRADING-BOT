"""
Main Trading Agent v2.0
=======================
Orchestrates all components:
- Market data fetching
- Strategy analysis
- AI reasoning
- Trade execution
- Notifications
- Error handling
- Logging
"""

import logging
from typing import Dict, Optional
from datetime import datetime

# Import all our enhanced modules
from market engine import (
    get_crypto_price,
    get_ohlcv,
    get_stock_price,
    get_stock_ohlcv,
    ExchangeManager
)
from strategy engine import analyze_market, StrategyConfig
from ai reasoning gemini import explain_decision, get_ai_stats
from trade executor import TradeExecutor, ExecutionMode, OrderType
from notifications import NotificationManager, NotificationType


# ==========================================
# LOGGING SETUP
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==========================================
# MAIN TRADING AGENT
# ==========================================
class TradingAgent:
    """
    Main trading agent that orchestrates everything.
    
    YOUR ORIGINAL: Probably just had separate function calls scattered around
    
    MY VERSION:
    - ✅ Single class that manages everything
    - ✅ Proper initialization
    - ✅ Error handling at every step
    - ✅ Comprehensive logging
    - ✅ State management
    - ✅ Modular design
    """
    
    def __init__(
        self,
        capital_kes: float = 500000,
        max_risk_per_trade_pct: float = 2.0,
        mode: str = "paper",
        risk_profile: str = "moderate",
        enable_notifications: bool = True
    ):
        """
        Initialize the trading agent.
        
        Args:
            capital_kes: Trading capital in KES
            max_risk_per_trade_pct: Max risk per trade
            mode: 'paper' or 'live'
            risk_profile: 'conservative', 'moderate', or 'aggressive'
            enable_notifications: Whether to send notifications
        """
        logger.info("=" * 60)
        logger.info("INITIALIZING WHITNEY TRADING AGENT v2.0")
        logger.info("=" * 60)
        
        # Initialize components
        self.exchange_manager = ExchangeManager()
        
        self.executor = TradeExecutor(
            capital_kes=capital_kes,
            max_risk_per_trade_pct=max_risk_per_trade_pct,
            mode=ExecutionMode.PAPER if mode == "paper" else ExecutionMode.LIVE
        )
        
        self.notifier = NotificationManager() if enable_notifications else None
        
        self.risk_profile = risk_profile
        self.capital = capital_kes
        
        # State tracking
        self.analysis_history = []
        self.last_run = None
        
        logger.info(f"✅ Agent initialized (Mode: {mode}, Capital: {capital_kes:,.0f} KES)")
        logger.info("=" * 60)
    
    def run_analysis(
        self,
        asset: str,
        asset_type: str = "crypto",
        timeframe: str = "1h",
        execute_trades: bool = False,
        get_ai_explanation: bool = True
    ) -> Dict:
        """
        Main analysis pipeline.
        
        This is the CORE function that runs everything:
        1. Fetch market data
        2. Analyze with strategy engine
        3. Get AI explanation
        4. Execute trade (if enabled)
        5. Send notifications
        
        Args:
            asset: Asset to analyze (e.g., 'BTC/USDT' or 'PLTR')
            asset_type: 'crypto' or 'stock'
            timeframe: Chart timeframe
            execute_trades: Whether to execute trades automatically
            get_ai_explanation: Whether to get AI reasoning
        
        Returns:
            Complete analysis result dict
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"RUNNING ANALYSIS FOR {asset}")
        logger.info(f"{'='*60}")
        
        result = {
            "success": False,
            "asset": asset,
            "asset_type": asset_type,
            "timestamp": datetime.now(),
            "steps": {}
        }
        
        try:
            # ==========================================
            # STEP 1: FETCH MARKET DATA
            # ==========================================
            logger.info("📊 Step 1/5: Fetching market data...")
            
            if asset_type == "crypto":
                df = get_ohlcv(
                    symbol=asset,
                    timeframe=timeframe,
                    limit=200,
                    add_indicators=True
                )
                current_price = get_crypto_price(asset)
            else:
                period_map = {"1h": "5d", "1d": "3mo", "1wk": "1y"}
                period = period_map.get(timeframe, "1mo")
                df = get_stock_ohlcv(
                    ticker=asset,
                    period=period,
                    interval=timeframe
                )
                current_price = get_stock_price(asset)
            
            if df.empty:
                raise ValueError(f"No data available for {asset}")
            
            result["steps"]["data_fetch"] = {
                "success": True,
                "candles": len(df),
                "current_price": current_price
            }
            
            logger.info(f"✅ Fetched {len(df)} candles, current price: ${current_price:,.2f}")
            
            # ==========================================
            # STEP 2: ANALYZE WITH STRATEGY ENGINE
            # ==========================================
            logger.info("🔍 Step 2/5: Running strategy analysis...")
            
            analysis = analyze_market(
                df=df,
                asset=asset,
                timeframe=timeframe,
                risk_profile=self.risk_profile
            )
            
            result["steps"]["strategy_analysis"] = analysis
            
            signal = analysis.get('signal', 'ERROR')
            confidence = analysis.get('confidence', 0)
            
            logger.info(
                f"✅ Analysis complete: {signal} "
                f"(Confidence: {confidence:.0f}%, Score: {analysis.get('score', 0):.1f}/10)"
            )
            
            # ==========================================
            # STEP 3: GET AI EXPLANATION
            # ==========================================
            if get_ai_explanation:
                logger.info("🤖 Step 3/5: Getting AI explanation...")
                
                try:
                    ai_result = explain_decision(
                        signal=signal,
                        asset=asset,
                        analysis_data=analysis,
                        risk_profile=self.risk_profile
                    )
                    
                    result["steps"]["ai_reasoning"] = ai_result
                    
                    if ai_result.get('success'):
                        logger.info("✅ AI explanation received")
                    else:
                        logger.warning(f"⚠️ AI explanation failed: {ai_result.get('error')}")
                
                except Exception as e:
                    logger.error(f"❌ AI explanation error: {e}")
                    result["steps"]["ai_reasoning"] = {"success": False, "error": str(e)}
            else:
                logger.info("⏭️ Step 3/5: Skipped AI explanation")
                result["steps"]["ai_reasoning"] = {"skipped": True}
            
            # ==========================================
            # STEP 4: EXECUTE TRADE
            # ==========================================
            if execute_trades and signal in ["BUY", "SELL"]:
                logger.info(f"⚡ Step 4/5: Executing {signal} trade...")
                
                try:
                    trade_result = self.executor.execute_trade(
                        signal=signal,
                        asset=asset,
                        analysis_data=analysis
                    )
                    
                    result["steps"]["trade_execution"] = trade_result
                    
                    if trade_result.get('success'):
                        logger.info(f"✅ Trade executed: {signal} {asset}")
                        
                        # Send notification
                        if self.notifier:
                            self.notifier.send_notification(
                                NotificationType.TRADE_EXECUTED,
                                trade_result
                            )
                    else:
                        logger.error(f"❌ Trade failed: {trade_result.get('message')}")
                
                except Exception as e:
                    logger.error(f"❌ Trade execution error: {e}")
                    result["steps"]["trade_execution"] = {"success": False, "error": str(e)}
            
            elif execute_trades and signal == "HOLD":
                logger.info("⏸️ Step 4/5: No trade - HOLD signal")
                result["steps"]["trade_execution"] = {
                    "success": True,
                    "action": "HOLD",
                    "message": "No trade executed"
                }
            else:
                logger.info("⏭️ Step 4/5: Skipped trade execution")
                result["steps"]["trade_execution"] = {"skipped": True}
            
            # ==========================================
            # STEP 5: SEND NOTIFICATION
            # ==========================================
            if self.notifier and signal in ["BUY", "SELL"]:
                logger.info("📧 Step 5/5: Sending notification...")
                
                try:
                    notif_result = self.notifier.send_notification(
                        NotificationType.TRADE_SIGNAL,
                        {
                            "signal": signal,
                            "asset": asset,
                            "confidence": confidence,
                            "reasoning": analysis.get('reasoning', [])
                        }
                    )
                    
                    result["steps"]["notification"] = notif_result
                    
                    if notif_result.get('success'):
                        logger.info("✅ Notification sent")
                    else:
                        logger.warning("⚠️ Notification failed")
                
                except Exception as e:
                    logger.error(f"❌ Notification error: {e}")
                    result["steps"]["notification"] = {"success": False, "error": str(e)}
            else:
                logger.info("⏭️ Step 5/5: Skipped notification")
                result["steps"]["notification"] = {"skipped": True}
            
            # ==========================================
            # COMPLETE
            # ==========================================
            result["success"] = True
            self.last_run = datetime.now()
            self.analysis_history.append(result)
            
            logger.info("=" * 60)
            logger.info("✅ ANALYSIS COMPLETE")
            logger.info("=" * 60)
            
            return result
        
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}", exc_info=True)
            result["success"] = False
            result["error"] = str(e)
            return result
    
    def run_multi_asset_scan(
        self,
        crypto_pairs: Optional[list] = None,
        stock_tickers: Optional[list] = None,
        timeframe: str = "1h"
    ) -> Dict:
        """
        Scan multiple assets and find best opportunities.
        
        NEW FEATURE: Your original probably didn't have this!
        """
        logger.info("\n" + "=" * 60)
        logger.info("RUNNING MULTI-ASSET SCAN")
        logger.info("=" * 60)
        
        if crypto_pairs is None:
            crypto_pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        
        if stock_tickers is None:
            stock_tickers = ["PLTR", "T", "IBM", "CSCO"]
        
        results = {
            "crypto": {},
            "stocks": {},
            "best_opportunity": None,
            "timestamp": datetime.now()
        }
        
        # Scan crypto
        for pair in crypto_pairs:
            logger.info(f"Scanning {pair}...")
            try:
                analysis = self.run_analysis(
                    asset=pair,
                    asset_type="crypto",
                    timeframe=timeframe,
                    execute_trades=False,
                    get_ai_explanation=False
                )
                results["crypto"][pair] = analysis
            except Exception as e:
                logger.error(f"Error scanning {pair}: {e}")
        
        # Scan stocks
        for ticker in stock_tickers:
            logger.info(f"Scanning {ticker}...")
            try:
                analysis = self.run_analysis(
                    asset=ticker,
                    asset_type="stock",
                    timeframe=timeframe,
                    execute_trades=False,
                    get_ai_explanation=False
                )
                results["stocks"][ticker] = analysis
            except Exception as e:
                logger.error(f"Error scanning {ticker}: {e}")
        
        # Find best opportunity
        best_score = 0
        best_asset = None
        
        for asset_type in ["crypto", "stocks"]:
            for asset, data in results[asset_type].items():
                if data.get("success"):
                    analysis = data.get("steps", {}).get("strategy_analysis", {})
                    score = analysis.get("score", 0)
                    signal = analysis.get("signal", "HOLD")
                    
                    if signal in ["BUY", "SELL"] and abs(score) > abs(best_score):
                        best_score = score
                        best_asset = {
                            "asset": asset,
                            "type": asset_type,
                            "signal": signal,
                            "score": score,
                            "confidence": analysis.get("confidence", 0)
                        }
        
        results["best_opportunity"] = best_asset
        
        if best_asset:
            logger.info(
                f"🎯 Best opportunity: {best_asset['signal']} {best_asset['asset']} "
                f"(Score: {best_asset['score']:.1f}, Confidence: {best_asset['confidence']:.0f}%)"
            )
        else:
            logger.info("No strong opportunities found")
        
        logger.info("=" * 60)
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get agent statistics"""
        return {
            "executor_stats": self.executor.get_statistics(),
            "ai_stats": get_ai_stats(),
            "notification_stats": self.notifier.get_stats() if self.notifier else {},
            "total_analyses": len(self.analysis_history),
            "last_run": self.last_run
        }
    
    def emergency_stop(self, reason: str = "Emergency"):
        """Emergency stop - closes all positions"""
        logger.critical(f"🚨 EMERGENCY STOP: {reason}")
        self.executor.trigger_emergency_stop(reason)


# ==========================================
# SIMPLE FUNCTION (BACKWARDS COMPATIBLE)
# ==========================================
def run_agent(asset: str, mode: str = "paper") -> Dict:
    """
    Simple function compatible with your original interface.
    
    Now it actually does everything properly!
    """
    agent = TradingAgent(mode=mode)
    
    # Determine asset type
    asset_type = "crypto" if "/" in asset else "stock"
    
    return agent.run_analysis(
        asset=asset,
        asset_type=asset_type,
        execute_trades=(mode == "live")
    )


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    """
    Example usage showing the agent in action
    """
    print("\n" + "=" * 60)
    print("WHITNEY TRADING AGENT v2.0 - DEMO")
    print("=" * 60)
    
    # Initialize agent
    agent = TradingAgent(
        capital_kes=500000,
        mode="paper",
        risk_profile="moderate"
    )
    
    # Run single analysis
    print("\n1. Running single asset analysis...")
    result = agent.run_analysis(
        asset="BTC/USDT",
        asset_type="crypto",
        timeframe="1h",
        execute_trades=True,
        get_ai_explanation=True
    )
    
    print(f"\nResult: {result['steps']['strategy_analysis']['signal']}")
    print(f"Confidence: {result['steps']['strategy_analysis']['confidence']:.0f}%")
    
    # Run multi-asset scan
    print("\n2. Running multi-asset scan...")
    scan_results = agent.run_multi_asset_scan(
        crypto_pairs=["BTC/USDT", "ETH/USDT"],
        stock_tickers=["PLTR", "AAPL"],
        timeframe="1h"
    )
    
    if scan_results["best_opportunity"]:
        best = scan_results["best_opportunity"]
        print(f"\nBest opportunity: {best['signal']} {best['asset']}")
        print(f"Score: {best['score']:.1f}, Confidence: {best['confidence']:.0f}%")
    
    # Get statistics
    print("\n3. Agent statistics:")
    stats = agent.get_statistics()
    print(f"Total analyses: {stats['total_analyses']}")
    print(f"Total trades: {stats['executor_stats']['total_trades']}")
    
    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)