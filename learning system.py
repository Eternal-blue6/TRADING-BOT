"""
Continuous Learning System v1.0
================================
Makes your bot LEARN from past trades!

Features:
- Records every trade with outcome
- Analyzes what worked and what didn't
- Adjusts strategy parameters based on results
- Detects patterns in winning/losing trades
- Improves over time (like a real trader!)

Your bot will:
✅ Remember past mistakes
✅ Identify what conditions lead to wins
✅ Adapt strategy parameters
✅ Get better with each trade
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


# ==========================================
# CONFIGURATION
# ==========================================
class LearningConfig:
    """Learning system configuration"""
    
    # Storage
    TRADES_DB_FILE = "trade_history.json"
    LEARNING_DB_FILE = "learning_insights.json"
    
    # Learning parameters
    MIN_TRADES_FOR_LEARNING = 10  # Need at least 10 trades to learn
    LEARNING_UPDATE_FREQUENCY = 5  # Update insights every 5 trades
    
    # Performance thresholds
    GOOD_WIN_RATE = 0.60  # 60% win rate is good
    GOOD_PROFIT_FACTOR = 1.5  # 1.5:1 profit factor is good
    
    # Adjustment sensitivity
    PARAMETER_ADJUSTMENT_RATE = 0.1  # Adjust by 10% each time


# ==========================================
# TRADE RECORD
# ==========================================
class TradeRecord:
    """Single trade record with all details"""
    
    def __init__(
        self,
        trade_id: str,
        timestamp: datetime,
        asset: str,
        signal: str,
        entry_price: float,
        exit_price: Optional[float] = None,
        position_size: float = 0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        # Analysis data at time of trade
        analysis_data: Optional[Dict] = None,
        # Trade outcome
        outcome: Optional[str] = None,  # WIN, LOSS, BREAKEVEN
        profit_loss_pct: Optional[float] = None,
        exit_reason: Optional[str] = None,
        duration_hours: Optional[float] = None
    ):
        self.trade_id = trade_id
        self.timestamp = timestamp
        self.asset = asset
        self.signal = signal
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.analysis_data = analysis_data or {}
        self.outcome = outcome
        self.profit_loss_pct = profit_loss_pct
        self.exit_reason = exit_reason
        self.duration_hours = duration_hours
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            "trade_id": self.trade_id,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            "asset": self.asset,
            "signal": self.signal,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "position_size": self.position_size,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "analysis_data": self.analysis_data,
            "outcome": self.outcome,
            "profit_loss_pct": self.profit_loss_pct,
            "exit_reason": self.exit_reason,
            "duration_hours": self.duration_hours
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TradeRecord':
        """Create from dictionary"""
        timestamp = data.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        return cls(
            trade_id=data.get('trade_id'),
            timestamp=timestamp,
            asset=data.get('asset'),
            signal=data.get('signal'),
            entry_price=data.get('entry_price'),
            exit_price=data.get('exit_price'),
            position_size=data.get('position_size'),
            stop_loss=data.get('stop_loss'),
            take_profit=data.get('take_profit'),
            analysis_data=data.get('analysis_data'),
            outcome=data.get('outcome'),
            profit_loss_pct=data.get('profit_loss_pct'),
            exit_reason=data.get('exit_reason'),
            duration_hours=data.get('duration_hours')
        )


# ==========================================
# LEARNING INSIGHTS
# ==========================================
class LearningInsights:
    """What the bot has learned from past trades"""
    
    def __init__(self):
        # Overall performance
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.win_rate = 0.0
        self.average_win_pct = 0.0
        self.average_loss_pct = 0.0
        self.profit_factor = 0.0
        
        # Pattern recognition
        self.best_conditions = {}  # What conditions lead to wins
        self.worst_conditions = {}  # What conditions lead to losses
        
        # Parameter adjustments
        self.recommended_adjustments = {}
        
        # Asset-specific insights
        self.asset_performance = {}  # Performance per asset
        
        # Time-based insights
        self.best_trading_hours = []
        self.worst_trading_hours = []
        
        # Recent performance
        self.last_10_trades_win_rate = 0.0
        self.trend = "NEUTRAL"  # IMPROVING, DECLINING, NEUTRAL
        
        # Last update
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "average_win_pct": self.average_win_pct,
            "average_loss_pct": self.average_loss_pct,
            "profit_factor": self.profit_factor,
            "best_conditions": self.best_conditions,
            "worst_conditions": self.worst_conditions,
            "recommended_adjustments": self.recommended_adjustments,
            "asset_performance": self.asset_performance,
            "best_trading_hours": self.best_trading_hours,
            "worst_trading_hours": self.worst_trading_hours,
            "last_10_trades_win_rate": self.last_10_trades_win_rate,
            "trend": self.trend,
            "last_updated": self.last_updated.isoformat() if isinstance(self.last_updated, datetime) else str(self.last_updated)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LearningInsights':
        """Create from dictionary"""
        insights = cls()
        insights.total_trades = data.get('total_trades', 0)
        insights.winning_trades = data.get('winning_trades', 0)
        insights.losing_trades = data.get('losing_trades', 0)
        insights.win_rate = data.get('win_rate', 0.0)
        insights.average_win_pct = data.get('average_win_pct', 0.0)
        insights.average_loss_pct = data.get('average_loss_pct', 0.0)
        insights.profit_factor = data.get('profit_factor', 0.0)
        insights.best_conditions = data.get('best_conditions', {})
        insights.worst_conditions = data.get('worst_conditions', {})
        insights.recommended_adjustments = data.get('recommended_adjustments', {})
        insights.asset_performance = data.get('asset_performance', {})
        insights.best_trading_hours = data.get('best_trading_hours', [])
        insights.worst_trading_hours = data.get('worst_trading_hours', [])
        insights.last_10_trades_win_rate = data.get('last_10_trades_win_rate', 0.0)
        insights.trend = data.get('trend', 'NEUTRAL')
        
        last_updated = data.get('last_updated')
        if isinstance(last_updated, str):
            insights.last_updated = datetime.fromisoformat(last_updated)
        
        return insights


# ==========================================
# MAIN LEARNING SYSTEM
# ==========================================
class LearningSystem:
    """
    The brain that makes your bot learn from experience!
    
    HOW IT WORKS:
    1. Records every trade you make
    2. Analyzes what worked (winning trades)
    3. Analyzes what didn't work (losing trades)
    4. Finds patterns (e.g., "high RSI trades always lose")
    5. Adjusts parameters to improve
    6. Gets better over time!
    
    EXAMPLE:
    After 50 trades, bot notices:
    - Trades with RSI < 25 have 80% win rate (GOOD!)
    - Trades with low volume have 30% win rate (BAD!)
    
    Bot learns:
    - Lower RSI threshold from 30 to 25 ✅
    - Increase volume requirement ✅
    - Next 50 trades: Win rate improves from 50% to 65%! 🎉
    """
    
    def __init__(self, storage_dir: str = "."):
        self.storage_dir = Path(storage_dir)
        self.trades_file = self.storage_dir / LearningConfig.TRADES_DB_FILE
        self.insights_file = self.storage_dir / LearningConfig.LEARNING_DB_FILE
        
        # Load existing data
        self.trades: List[TradeRecord] = self._load_trades()
        self.insights: LearningInsights = self._load_insights()
        
        logger.info(
            f"📚 Learning System initialized "
            f"(Trades: {len(self.trades)}, Win Rate: {self.insights.win_rate:.1%})"
        )
    
    def record_trade(self, trade_data: Dict) -> TradeRecord:
        """
        Record a new trade.
        Call this when a trade is OPENED.
        """
        trade = TradeRecord(
            trade_id=trade_data.get('trade_id', f"TRADE_{datetime.now().strftime('%Y%m%d%H%M%S')}"),
            timestamp=trade_data.get('timestamp', datetime.now()),
            asset=trade_data.get('asset'),
            signal=trade_data.get('signal'),
            entry_price=trade_data.get('entry_price'),
            position_size=trade_data.get('units', 0),
            stop_loss=trade_data.get('stop_loss'),
            take_profit=trade_data.get('take_profit_1'),
            analysis_data=trade_data.get('analysis', {})
        )
        
        self.trades.append(trade)
        self._save_trades()
        
        logger.info(f"✅ Trade recorded: {trade.signal} {trade.asset} @ {trade.entry_price}")
        
        return trade
    
    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str = "Manual close"
    ) -> Optional[TradeRecord]:
        """
        Close a trade and record the outcome.
        Call this when a trade is CLOSED.
        
        This is where LEARNING happens!
        """
        # Find the trade
        trade = None
        for t in self.trades:
            if t.trade_id == trade_id:
                trade = t
                break
        
        if not trade:
            logger.error(f"Trade {trade_id} not found")
            return None
        
        # Calculate outcome
        if trade.signal == "BUY":
            profit_loss_pct = ((exit_price - trade.entry_price) / trade.entry_price) * 100
        else:  # SELL
            profit_loss_pct = ((trade.entry_price - exit_price) / trade.entry_price) * 100
        
        # Determine outcome
        if profit_loss_pct > 0.5:
            outcome = "WIN"
        elif profit_loss_pct < -0.5:
            outcome = "LOSS"
        else:
            outcome = "BREAKEVEN"
        
        # Calculate duration
        duration = (datetime.now() - trade.timestamp).total_seconds() / 3600  # hours
        
        # Update trade record
        trade.exit_price = exit_price
        trade.outcome = outcome
        trade.profit_loss_pct = profit_loss_pct
        trade.exit_reason = exit_reason
        trade.duration_hours = duration
        
        self._save_trades()
        
        logger.info(
            f"📊 Trade closed: {trade.signal} {trade.asset} "
            f"{outcome} ({profit_loss_pct:+.2f}%) - {exit_reason}"
        )
        
        # Trigger learning if enough trades
        if len(self.trades) % LearningConfig.LEARNING_UPDATE_FREQUENCY == 0:
            self.analyze_and_learn()
        
        return trade
    
    def analyze_and_learn(self) -> LearningInsights:
        """
        Analyze all past trades and generate insights.
        This is the LEARNING ENGINE!
        """
        # Get only closed trades
        closed_trades = [t for t in self.trades if t.outcome is not None]
        
        if len(closed_trades) < LearningConfig.MIN_TRADES_FOR_LEARNING:
            logger.info(
                f"📚 Not enough trades to learn yet "
                f"({len(closed_trades)}/{LearningConfig.MIN_TRADES_FOR_LEARNING})"
            )
            return self.insights
        
        logger.info(f"🧠 Analyzing {len(closed_trades)} trades for learning...")
        
        # Reset insights
        insights = LearningInsights()
        
        # 1. OVERALL PERFORMANCE
        insights.total_trades = len(closed_trades)
        wins = [t for t in closed_trades if t.outcome == "WIN"]
        losses = [t for t in closed_trades if t.outcome == "LOSS"]
        
        insights.winning_trades = len(wins)
        insights.losing_trades = len(losses)
        insights.win_rate = len(wins) / len(closed_trades) if closed_trades else 0
        
        if wins:
            insights.average_win_pct = np.mean([t.profit_loss_pct for t in wins])
        if losses:
            insights.average_loss_pct = np.mean([abs(t.profit_loss_pct) for t in losses])
        
        # Profit factor
        total_wins = sum([t.profit_loss_pct for t in wins])
        total_losses = sum([abs(t.profit_loss_pct) for t in losses])
        insights.profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # 2. PATTERN RECOGNITION - What conditions lead to wins?
        insights.best_conditions = self._find_winning_patterns(wins)
        insights.worst_conditions = self._find_losing_patterns(losses)
        
        # 3. ASSET PERFORMANCE
        for asset in set([t.asset for t in closed_trades]):
            asset_trades = [t for t in closed_trades if t.asset == asset]
            asset_wins = [t for t in asset_trades if t.outcome == "WIN"]
            
            insights.asset_performance[asset] = {
                "total_trades": len(asset_trades),
                "win_rate": len(asset_wins) / len(asset_trades) if asset_trades else 0,
                "avg_profit": np.mean([t.profit_loss_pct for t in asset_trades])
            }
        
        # 4. TIME-BASED PATTERNS
        insights.best_trading_hours, insights.worst_trading_hours = self._find_time_patterns(closed_trades)
        
        # 5. RECENT TREND
        if len(closed_trades) >= 10:
            last_10 = closed_trades[-10:]
            last_10_wins = [t for t in last_10 if t.outcome == "WIN"]
            insights.last_10_trades_win_rate = len(last_10_wins) / 10
            
            # Compare to overall win rate
            if insights.last_10_trades_win_rate > insights.win_rate + 0.1:
                insights.trend = "IMPROVING"
            elif insights.last_10_trades_win_rate < insights.win_rate - 0.1:
                insights.trend = "DECLINING"
            else:
                insights.trend = "STABLE"
        
        # 6. RECOMMENDED ADJUSTMENTS
        insights.recommended_adjustments = self._generate_recommendations(insights, closed_trades)
        
        insights.last_updated = datetime.now()
        
        # Save insights
        self.insights = insights
        self._save_insights()
        
        # Log summary
        logger.info("=" * 60)
        logger.info("🧠 LEARNING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total Trades: {insights.total_trades}")
        logger.info(f"Win Rate: {insights.win_rate:.1%}")
        logger.info(f"Profit Factor: {insights.profit_factor:.2f}")
        logger.info(f"Trend: {insights.trend}")
        logger.info(f"Best Asset: {max(insights.asset_performance.items(), key=lambda x: x[1]['win_rate'])[0] if insights.asset_performance else 'N/A'}")
        
        if insights.recommended_adjustments:
            logger.info("\n📝 Recommended Adjustments:")
            for param, value in insights.recommended_adjustments.items():
                logger.info(f"  - {param}: {value}")
        
        logger.info("=" * 60)
        
        return insights
    
    def _find_winning_patterns(self, wins: List[TradeRecord]) -> Dict:
        """Find common patterns in winning trades"""
        patterns = {}
        
        if not wins:
            return patterns
        
        # RSI patterns
        rsi_values = [
            t.analysis_data.get('momentum', {}).get('indicators', {}).get('rsi')
            for t in wins
            if t.analysis_data.get('momentum', {}).get('indicators', {}).get('rsi') is not None
        ]
        if rsi_values:
            patterns['avg_winning_rsi'] = np.mean(rsi_values)
            patterns['rsi_range'] = (min(rsi_values), max(rsi_values))
        
        # Confidence patterns
        confidence_values = [
            t.analysis_data.get('confidence', 0)
            for t in wins
            if t.analysis_data.get('confidence') is not None
        ]
        if confidence_values:
            patterns['avg_winning_confidence'] = np.mean(confidence_values)
        
        # Trend patterns
        trend_directions = [
            t.analysis_data.get('trend', {}).get('direction')
            for t in wins
            if t.analysis_data.get('trend', {}).get('direction') is not None
        ]
        if trend_directions:
            from collections import Counter
            patterns['best_trend'] = Counter(trend_directions).most_common(1)[0][0]
        
        # Volume patterns
        volume_statuses = [
            t.analysis_data.get('volume', {}).get('status')
            for t in wins
            if t.analysis_data.get('volume', {}).get('status') is not None
        ]
        if volume_statuses:
            from collections import Counter
            patterns['best_volume'] = Counter(volume_statuses).most_common(1)[0][0]
        
        return patterns
    
    def _find_losing_patterns(self, losses: List[TradeRecord]) -> Dict:
        """Find common patterns in losing trades"""
        patterns = {}
        
        if not losses:
            return patterns
        
        # Similar analysis as winning patterns
        # RSI patterns
        rsi_values = [
            t.analysis_data.get('momentum', {}).get('indicators', {}).get('rsi')
            for t in losses
            if t.analysis_data.get('momentum', {}).get('indicators', {}).get('rsi') is not None
        ]
        if rsi_values:
            patterns['avg_losing_rsi'] = np.mean(rsi_values)
        
        # Low confidence trades
        confidence_values = [
            t.analysis_data.get('confidence', 0)
            for t in losses
            if t.analysis_data.get('confidence') is not None
        ]
        if confidence_values:
            patterns['avg_losing_confidence'] = np.mean(confidence_values)
        
        return patterns
    
    def _find_time_patterns(self, trades: List[TradeRecord]) -> Tuple[List[int], List[int]]:
        """Find which hours are best/worst for trading"""
        hour_performance = {}
        
        for trade in trades:
            hour = trade.timestamp.hour
            if hour not in hour_performance:
                hour_performance[hour] = {'wins': 0, 'total': 0}
            
            hour_performance[hour]['total'] += 1
            if trade.outcome == "WIN":
                hour_performance[hour]['wins'] += 1
        
        # Calculate win rates per hour
        hour_win_rates = {
            hour: data['wins'] / data['total'] if data['total'] >= 3 else 0
            for hour, data in hour_performance.items()
        }
        
        # Sort by win rate
        sorted_hours = sorted(hour_win_rates.items(), key=lambda x: x[1], reverse=True)
        
        best_hours = [h for h, wr in sorted_hours[:3]]
        worst_hours = [h for h, wr in sorted_hours[-3:]]
        
        return best_hours, worst_hours
    
    def _generate_recommendations(self, insights: LearningInsights, trades: List[TradeRecord]) -> Dict:
        """Generate parameter adjustment recommendations based on learning"""
        recommendations = {}
        
        # If win rate is low, suggest being more selective
        if insights.win_rate < LearningConfig.GOOD_WIN_RATE:
            # Increase confidence threshold
            if insights.best_conditions.get('avg_winning_confidence'):
                winning_conf = insights.best_conditions['avg_winning_confidence']
                recommendations['min_confidence_threshold'] = round(winning_conf * 0.9, 1)
            
            # Adjust RSI thresholds based on what worked
            if insights.best_conditions.get('avg_winning_rsi'):
                winning_rsi = insights.best_conditions['avg_winning_rsi']
                recommendations['suggested_rsi_threshold'] = round(winning_rsi, 1)
        
        # If profit factor is low, suggest tighter stops
        if insights.profit_factor < LearningConfig.GOOD_PROFIT_FACTOR:
            recommendations['suggestion'] = "Consider tighter stop losses or wider take profits"
        
        # If a specific asset performs poorly, suggest avoiding it
        poor_assets = [
            asset for asset, perf in insights.asset_performance.items()
            if perf['win_rate'] < 0.4 and perf['total_trades'] >= 5
        ]
        if poor_assets:
            recommendations['avoid_assets'] = poor_assets
        
        # If trend is declining, suggest more conservative approach
        if insights.trend == "DECLINING":
            recommendations['risk_adjustment'] = "Reduce position sizes by 20%"
        
        return recommendations
    
    def get_trading_advice(self, current_analysis: Dict) -> Dict:
        """
        Get advice based on past learning.
        Call this BEFORE making a trade to get insights!
        """
        if len(self.trades) < LearningConfig.MIN_TRADES_FOR_LEARNING:
            return {
                "advice": "Not enough historical data yet. Trade carefully!",
                "confidence_adjustment": 0,
                "warnings": []
            }
        
        advice = {
            "advice": "",
            "confidence_adjustment": 0,
            "warnings": [],
            "encouragements": []
        }
        
        # Check against learned patterns
        current_rsi = current_analysis.get('momentum', {}).get('indicators', {}).get('rsi')
        current_confidence = current_analysis.get('confidence', 0)
        current_trend = current_analysis.get('trend', {}).get('direction')
        current_asset = current_analysis.get('asset', '')
        
        # RSI check
        if current_rsi and self.insights.best_conditions.get('avg_winning_rsi'):
            winning_rsi = self.insights.best_conditions['avg_winning_rsi']
            if abs(current_rsi - winning_rsi) < 5:
                advice['encouragements'].append(f"✅ RSI {current_rsi:.1f} is similar to past winners")
                advice['confidence_adjustment'] += 5
            elif abs(current_rsi - self.insights.worst_conditions.get('avg_losing_rsi', 50)) < 5:
                advice['warnings'].append(f"⚠️ RSI {current_rsi:.1f} is similar to past losers")
                advice['confidence_adjustment'] -= 10
        
        # Confidence check
        if current_confidence < (self.insights.best_conditions.get('avg_winning_confidence', 70) - 10):
            advice['warnings'].append(
                f"⚠️ Confidence {current_confidence:.0f}% is lower than typical winners "
                f"({self.insights.best_conditions.get('avg_winning_confidence', 0):.0f}%)"
            )
            advice['confidence_adjustment'] -= 10
        
        # Asset check
        if current_asset in self.insights.asset_performance:
            asset_perf = self.insights.asset_performance[current_asset]
            if asset_perf['win_rate'] > 0.6:
                advice['encouragements'].append(
                    f"✅ {current_asset} has {asset_perf['win_rate']:.0%} win rate in past"
                )
                advice['confidence_adjustment'] += 5
            elif asset_perf['win_rate'] < 0.4:
                advice['warnings'].append(
                    f"⚠️ {current_asset} has only {asset_perf['win_rate']:.0%} win rate in past"
                )
                advice['confidence_adjustment'] -= 10
        
        # Trend check
        if current_trend and self.insights.best_conditions.get('best_trend'):
            if current_trend == self.insights.best_conditions['best_trend']:
                advice['encouragements'].append(
                    f"✅ {current_trend} trend - historically performs well"
                )
                advice['confidence_adjustment'] += 5
        
        # Overall advice
        if advice['confidence_adjustment'] > 10:
            advice['advice'] = "🎯 This setup looks similar to past winners!"
        elif advice['confidence_adjustment'] < -10:
            advice['advice'] = "⚠️ Be cautious - this setup resembles past losers"
        else:
            advice['advice'] = "📊 Neutral - no strong historical pattern"
        
        return advice
    
    def _save_trades(self):
        """Save trades to disk"""
        try:
            data = [t.to_dict() for t in self.trades]
            with open(self.trades_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trades: {e}")
    
    def _load_trades(self) -> List[TradeRecord]:
        """Load trades from disk"""
        if not self.trades_file.exists():
            return []
        
        try:
            with open(self.trades_file, 'r') as f:
                data = json.load(f)
                return [TradeRecord.from_dict(t) for t in data]
        except Exception as e:
            logger.error(f"Error loading trades: {e}")
            return []
    
    def _save_insights(self):
        """Save insights to disk"""
        try:
            with open(self.insights_file, 'w') as f:
                json.dump(self.insights.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving insights: {e}")
    
    def _load_insights(self) -> LearningInsights:
        """Load insights from disk"""
        if not self.insights_file.exists():
            return LearningInsights()
        
        try:
            with open(self.insights_file, 'r') as f:
                data = json.load(f)
                return LearningInsights.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading insights: {e}")
            return LearningInsights()
    
    def get_report(self) -> str:
        """Get a human-readable learning report"""
        report = f"""
╔══════════════════════════════════════════════════════════╗
║              🧠 LEARNING SYSTEM REPORT                   ║
╚══════════════════════════════════════════════════════════╝

📊 OVERALL PERFORMANCE:
   Total Trades: {self.insights.total_trades}
   Win Rate: {self.insights.win_rate:.1%}
   Profit Factor: {self.insights.profit_factor:.2f}
   Trend: {self.insights.trend}

✅ WINNING PATTERNS:
{self._format_conditions(self.insights.best_conditions)}

❌ LOSING PATTERNS:
{self._format_conditions(self.insights.worst_conditions)}

📈 ASSET PERFORMANCE:
{self._format_asset_performance()}

💡 RECOMMENDATIONS:
{self._format_recommendations()}

⏰ BEST TRADING HOURS: {', '.join(map(str, self.insights.best_trading_hours))}
⏰ WORST TRADING HOURS: {', '.join(map(str, self.insights.worst_trading_hours))}

Last Updated: {self.insights.last_updated.strftime('%Y-%m-%d %H:%M:%S')}
"""
        return report
    
    def _format_conditions(self, conditions: Dict) -> str:
        """Format conditions for display"""
        if not conditions:
            return "   Not enough data yet"
        
        lines = []
        for key, value in conditions.items():
            lines.append(f"   - {key}: {value}")
        return '\n'.join(lines) if lines else "   Not enough data yet"
    
    def _format_asset_performance(self) -> str:
        """Format asset performance for display"""
        if not self.insights.asset_performance:
            return "   Not enough data yet"
        
        lines = []
        for asset, perf in sorted(
            self.insights.asset_performance.items(),
            key=lambda x: x[1]['win_rate'],
            reverse=True
        ):
            lines.append(
                f"   {asset}: {perf['win_rate']:.1%} win rate "
                f"({perf['total_trades']} trades, "
                f"avg: {perf['avg_profit']:+.2f}%)"
            )
        return '\n'.join(lines)
    
    def _format_recommendations(self) -> str:
        """Format recommendations for display"""
        if not self.insights.recommended_adjustments:
            return "   System performing well - no changes needed"
        
        lines = []
        for param, value in self.insights.recommended_adjustments.items():
            lines.append(f"   - {param}: {value}")
        return '\n'.join(lines)


# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    """Test the learning system"""
    
    print("\n" + "="*60)
    print("LEARNING SYSTEM TEST")
    print("="*60)
    
    # Initialize
    learning = LearningSystem()
    
    # Simulate recording a trade
    trade_data = {
        'trade_id': 'TEST_001',
        'asset': 'BTC/USDT',
        'signal': 'BUY',
        'entry_price': 45000,
        'units': 0.1,
        'stop_loss': 44000,
        'take_profit_1': 46500,
        'analysis': {
            'confidence': 75,
            'momentum': {'indicators': {'rsi': 35}},
            'trend': {'direction': 'UPTREND'}
        }
    }
    
    trade = learning.record_trade(trade_data)
    print(f"\n✅ Trade recorded: {trade.trade_id}")
    
    # Simulate closing the trade (winner)
    learning.close_trade(trade.trade_id, 46500, "Take profit hit")
    print(f"✅ Trade closed")
    
    # Show learning report
    if learning.insights.total_trades >= 1:
        print("\n" + learning.get_report())
    
    print("\n" + "="*60)