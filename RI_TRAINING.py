"""
Reinforcement Learning Training System
=======================================
Train your bot using REAL market data!

Framework: Stable-Baselines3 (RECOMMENDED!)
Why SB3 over TensorTrade:
✅ More stable, better maintained
✅ Easier to use
✅ Better documentation
✅ Pre-built algorithms (PPO, A2C, SAC)
✅ Used by thousands of researchers

Reward Design:
✅ Maximize profit
✅ Penalize large drawdowns
✅ Reward steady gains
✅ Penalize excessive trades (trading costs)
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


# ==========================================
# TRADING ENVIRONMENT
# ==========================================
class TradingEnvironment(gym.Env):
    """
    Custom Trading Environment for RL.
    
    State Space:
    - Technical indicators (RSI, MACD, etc.)
    - Price data (normalized)
    - Position info (long/short/neutral)
    - Account info (cash, equity, drawdown)
    
    Action Space:
    - 0: HOLD (do nothing)
    - 1: BUY (or increase position)
    - 2: SELL (or decrease position)
    
    Reward Function:
    - +reward for profit
    - -penalty for drawdown
    - -penalty for excessive trading
    - +bonus for steady gains
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000,
        transaction_cost_pct: float = 0.001,  # 0.1% per trade
        max_position_size: float = 1.0,  # Max 100% of capital
        lookback_window: int = 50
    ):
        """
        Args:
            df: OHLCV data with indicators
            initial_balance: Starting capital
            transaction_cost_pct: Trading fee
            max_position_size: Max position as % of capital
            lookback_window: How many candles to look back
        """
        super(TradingEnvironment, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.max_position_size = max_position_size
        self.lookback_window = lookback_window
        self.equity_curve = [initial_balance]
        self.max_equity = initial_balance
        self.trade=[]


        
        # Define action space: HOLD, BUY, SELL
        self.action_space = spaces.Discrete(3)
        
        # Define observation space
        # [price features, indicators, position info, account info]
        n_features = 5 + 10 + 3 + 4  # Adjust based on your indicators
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(lookback_window, n_features),
            dtype=np.float32
        )
        
        # Episode state
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # Shares/coins held
        self.entry_price = 0
        self.max_equity = initial_balance
        self.trades = []
        self.equity_curve = []
        
    def reset(self, seed=None, options=None):
        """Reset environment to start new episode"""
        # If your class inherits from gym.Env or gymnasium.Env, call super().reset
        try:
            super().reset(seed=seed)
        except Exception:
            pass  # safe fallback if no parent reset

        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.max_equity = self.initial_balance
        self.trades = []
        self.equity_curve = [self.initial_balance]
        self.max_equity = self.initial_balance

        
        obs = self._get_observation()
        # Gymnasium requires (obs, info) return signature

        return obs, {}


    
    def step(self, action):
        """
        Execute one timestep.
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get current price
        current_price = self.df.loc[self.current_step, 'close']
        
        # Execute action
        if action == 1:  # BUY
            self._execute_buy(current_price)
        elif action == 2:  # SELL
            self._execute_sell(current_price)
        # action == 0: HOLD (do nothing)
        
        # Calculate current equity
        equity = self.balance + (self.position * current_price)
        self.equity_curve.append(equity)
        
        # Update max equity for drawdown calculation
        if equity > self.max_equity:
            self.max_equity = equity
        
        # Calculate reward
        reward = self._calculate_reward(equity)
        
        # Move to next step
        self.current_step += 1

        # Define termination conditions
        terminated = (
        self.current_step >= len(self.df) - 1 or
        equity <= self.initial_balance * 0.5  # 50% loss = stop
        )

        # Define truncation (optional cutoff, e.g. max steps)
        truncated = False
        
        # Check if episode done
        done = (self.current_step >= len(self.df) - 1 or 
                equity <= self.initial_balance * 0.5)  # 50% loss = stop
        
        # Get next observation
        obs = self._get_observation()
        
        # Info dict
        info = {
            'equity': equity,
            'balance': self.balance,
            'position': self.position,
            'total_trades': len(self.trades)
        }
        
        return obs, reward, terminated, truncated, info


    
    def _execute_buy(self, price: float):
        """Execute buy order"""
        if self.balance <= 0:
            return
        
        # Calculate how much we can buy
        max_buy = (self.balance * self.max_position_size)
        shares_to_buy = max_buy / price
        
        # Account for transaction costs
        cost = shares_to_buy * price * (1 + self.transaction_cost_pct)
        
        if cost <= self.balance:
            self.position += shares_to_buy
            self.balance -= cost
            self.entry_price = price
            
            self.trades.append({
                'step': self.current_step,
                'action': 'BUY',
                'price': price,
                'shares': shares_to_buy,
                'cost': cost
            })
    
    def _execute_sell(self, price: float):
        """Execute sell order"""
        if self.position <= 0:
            return
        
        # Sell all position
        proceeds = self.position * price * (1 - self.transaction_cost_pct)
        
        self.balance += proceeds
        profit = (price - self.entry_price) * self.position if self.entry_price > 0 else 0
        
        self.trades.append({
            'step': self.current_step,
            'action': 'SELL',
            'price': price,
            'shares': self.position,
            'proceeds': proceeds,
            'profit': profit
        })
        
        self.position = 0
        self.entry_price = 0
    
    def _calculate_reward(self, current_equity: float) -> float:
        """
        Calculate reward for current step.
        
        Reward Design:
        1. Profit: Reward for making money
        2. Drawdown: Penalize losses from peak
        3. Steady gains: Bonus for consistent growth
        4. Trading costs: Penalize excessive trading
        """
        reward = 0
        
        # 1. PROFIT REWARD
        # Reward based on equity change
        if len(self.equity_curve) > 1:
            equity_change = current_equity - self.equity_curve[-2]
            profit_reward = equity_change / self.initial_balance * 100  # Normalize
            reward += profit_reward
        
        # 2. DRAWDOWN PENALTY
        # Penalize being below peak equity
        if current_equity < self.max_equity:
            drawdown = (self.max_equity - current_equity) / self.max_equity
            # Large penalty for large drawdowns!
            drawdown_penalty = -(drawdown ** 2) * 50  # Squared = stronger penalty for large drawdowns
            reward += drawdown_penalty
        
        # 3. STEADY GAINS BONUS
        # Reward consistent upward trend
        if len(self.equity_curve) >= 10:
            recent_equity = self.equity_curve[-10:]
            if all(recent_equity[i] <= recent_equity[i+1] for i in range(len(recent_equity)-1)):
                # Perfect uptrend!
                reward += 10
        
        # 4. TRADING COST PENALTY
        # Penalize too many trades (overtrading)
        if len(self.trades) > 0:
            if self.trades[-1]['step'] == self.current_step:
                # Just traded, small penalty
                reward -= 0.5
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (state).
        
        Returns normalized feature matrix.
        """
        # Get data window
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step
        
        window = self.df.iloc[start_idx:end_idx].copy()
        
        # Extract features
        features = []
        
        for _, row in window.iterrows():
            step_features = []
            
            # Price features (normalized)
            step_features.extend([
                row['open'] / row['close'],  # Normalized
                row['high'] / row['close'],
                row['low'] / row['close'],
                row['close'] / row.get('sma_20', row['close']),  # Relative to MA
                row['volume'] / row.get('volume_sma', row['volume'] + 1)  # Relative volume
            ])
            
            # Indicators (already mostly normalized)
            step_features.extend([
                row.get('rsi', 50) / 100,  # Normalize to 0-1
                row.get('macd', 0) / 100,
                row.get('macd_signal', 0) / 100,
                (row.get('bb_position', 0.5)),  # Already 0-1
                row.get('atr_percent', 2) / 10,  # Normalize
                row.get('adx', 25) / 100,
                row.get('stoch_k', 50) / 100,
                row.get('stoch_d', 50) / 100,
                row.get('volume_ratio', 1),
                int(row.get('above_ema_20', False))
            ])
            
            # Position info
            step_features.extend([
                1 if self.position > 0 else 0,  # In position?
                self.position / (self.balance + 1),  # Position size
                (row['close'] - self.entry_price) / (self.entry_price + 1) if self.position > 0 else 0  # Profit %
            ])
            
            # Account info
            current_equity = self.balance + (self.position * row['close'])
            step_features.extend([
                self.balance / self.initial_balance,  # Cash ratio
                current_equity / self.initial_balance,  # Equity ratio
                (self.max_equity - current_equity) / (self.max_equity + 1),  # Drawdown
                len(self.trades) / 100  # Trade count (normalized)
            ])
            
            features.append(step_features)
        
        # Pad if needed
        while len(features) < self.lookback_window:
            features.insert(0, [0] * len(features[0]))
        
        return np.array(features, dtype=np.float32)
    
    def render(self, mode='human'):
        """Render environment (for debugging)"""
        if len(self.equity_curve) > 0:
            print(f"Step: {self.current_step}, Equity: ${self.equity_curve[-1]:.2f}, Trades: {len(self.trades)}")


# ==========================================
# TRAINING PIPELINE
# ==========================================
class RLTrainer:
    """
    Reinforcement Learning Trainer.
    
    Handles:
    - Data preparation
    - Model training
    - Evaluation
    - Checkpointing
    - Performance tracking
    """
    
    def __init__(
        self,
        algorithm: str = "PPO",  # PPO, A2C, or SAC
        total_timesteps: int = 600000,
        learning_rate: float = 0.0003,
        batch_size: int = 64,
        model_dir: str = "rl_models"
    ):
        """
        Args:
            algorithm: RL algorithm to use
            total_timesteps: How long to train
            learning_rate: Learning rate
            batch_size: Batch size
            model_dir: Where to save models
        """
        self.algorithm = algorithm
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.env = None
        
        logger.info(f"✅ RL Trainer initialized (Algorithm: {algorithm})")
    
    def prepare_environment(self, training_data: pd.DataFrame) -> TradingEnvironment:
        """Prepare training environment"""
        logger.info(f"📊 Preparing environment with {len(training_data)} records...")
        
        env = TradingEnvironment(
            df=training_data,
            initial_balance=10000,
            transaction_cost_pct=0.001,
            max_position_size=0.95,
            lookback_window=50
        )
        
        # Wrap in Monitor for logging
        env = Monitor(env, str(self.model_dir / "monitor"))
        
        # Vectorize
        env = DummyVecEnv([lambda: env])
        
        self.env = env
        return env
    
    def train(
        self,
        training_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None
    ):
        """
        Train the RL agent.
        
        Args:
            training_data: Data to train on
            validation_data: Data to validate on (optional)
        """
        logger.info("=" * 60)
        logger.info("🚀 STARTING RL TRAINING")
        logger.info("=" * 60)
        
        # Prepare environment
        env = self.prepare_environment(training_data)
        
        # Create model
        logger.info(f"🤖 Creating {self.algorithm} model...")
        
        if self.algorithm == "PPO":
            self.model = PPO(
                "MlpPolicy",
                env,
                learning_rate=self.learning_rate,
                n_steps=2048,
                batch_size=self.batch_size,
                n_epochs=10,
                verbose=1,
                tensorboard_log=str(self.model_dir / "tensorboard")
            )
        elif self.algorithm == "A2C":
            self.model = A2C(
                "MlpPolicy",
                env,
                learning_rate=self.learning_rate,
                verbose=1,
                tensorboard_log=str(self.model_dir / "tensorboard")
            )
        elif self.algorithm == "SAC":
            self.model = SAC(
                "MlpPolicy",
                env,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                verbose=1,
                tensorboard_log=str(self.model_dir / "tensorboard")
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        # Setup callbacks
        callbacks = []
        
        # Checkpoint callback (save every 10k steps)
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=str(self.model_dir / "checkpoints"),
            name_prefix=f"{self.algorithm}_model"
        )
        callbacks.append(checkpoint_callback)
        
        # Eval callback (if validation data provided)
        if validation_data is not None:
            eval_env = self.prepare_environment(validation_data)
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(self.model_dir / "best_model"),
                log_path=str(self.model_dir / "eval"),
                eval_freq=5000,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
        
        from stable_baselines3.common.callbacks import EvalCallback

        # Create evaluation environment (can be same env or a separate test env)
        eval_env = self.env  # or DummyVecEnv([lambda: SimpleTradingEnv(test_df)])

        # Define evaluation callback
         eval_callback = EvalCallback(
             eval_env,
             eval_freq=50000,          # evaluate every 50k steps
             n_eval_episodes=10,       # run 10 episodes per evaluation
             deterministic=True,
             render=False
         )

        # Train!
        logger.info(f"🎯 Training for {self.total_timesteps:,} timesteps...")
        start_time = datetime.now()
        
        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=[callbacks, eval_callback]
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # Save final model
        model_path = self.model_dir / f"{self.algorithm}_final.zip"
        self.model.save(model_path)
        
        logger.info("=" * 60)
        logger.info("✅ TRAINING COMPLETE")
        logger.info(f"Duration: {duration/60:.1f} minutes")
        logger.info(f"Model saved: {model_path}")
        logger.info("=" * 60)
    
    def evaluate(
        self,
        test_data: pd.DataFrame,
        n_episodes: int = 10
    ) -> Dict:
        """
        Evaluate trained model.
        
        Returns performance metrics.
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        logger.info(f"📊 Evaluating model on {len(test_data)} records...")
        
        # Prepare test environment
        test_env = TradingEnvironment(df=test_data, initial_balance=10000)
        
        results = []
        
        for episode in range(n_episodes):
            obs = test_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = test_env.step(action)
                episode_reward += reward
            
            final_equity = info['equity']
            profit_pct = ((final_equity - test_env.initial_balance) / test_env.initial_balance) * 100
            
            results.append({
                'episode': episode + 1,
                'final_equity': final_equity,
                'profit_pct': profit_pct,
                'total_reward': episode_reward,
                'num_trades': len(test_env.trades)
            })
            
            logger.info(
                f"  Episode {episode+1}: "
                f"Profit: {profit_pct:+.2f}%, "
                f"Trades: {len(test_env.trades)}, "
                f"Reward: {episode_reward:.2f}"
            )
        
        # Calculate statistics
        avg_profit = np.mean([r['profit_pct'] for r in results])
        avg_trades = np.mean([r['num_trades'] for r in results])
        win_rate = sum(1 for r in results if r['profit_pct'] > 0) / len(results)
        
        summary = {
            "episodes": n_episodes,
            "avg_profit_pct": avg_profit,
            "avg_trades": avg_trades,
            "win_rate": win_rate,
            "best_profit": max(r['profit_pct'] for r in results),
            "worst_profit": min(r['profit_pct'] for r in results),
            "results": results
        }
        
        logger.info("\n📊 EVALUATION SUMMARY:")
        logger.info(f"  Average Profit: {avg_profit:+.2f}%")
        logger.info(f"  Win Rate: {win_rate:.1%}")
        logger.info(f"  Average Trades: {avg_trades:.1f}")
        logger.info(f"  Best Profit: {summary['best_profit']:+.2f}%")
        logger.info(f"  Worst Profit: {summary['worst_profit']:+.2f}%")
        
        return summary


# ==========================================
# COMPLETE TRAINING SCRIPT
# ==========================================
if __name__ == "__main__":
    """
    Complete RL training pipeline.
    
    Steps:
    1. Load training data
    2. Split into train/val/test
    3. Train model
    4. Evaluate
    5. Save results
    """
    
    print("\n" + "="*60)
    print("🤖 RL TRADING BOT TRAINING")
    print("="*60)
    
    # Load data
    print("\n📚 Loading training data...")
    from data_collection import TrainingDataLoader
    
    loader = TrainingDataLoader()
    all_data = loader.load_all_training_data()
    
    # Focus on one asset for initial training
    btc_data = all_data[
        (all_data['asset'] == 'BTC/USDT') & 
        (all_data['timeframe'] == '1h')
    ].copy()
    
    print(f"✅ Loaded {len(btc_data)} BTC records")
    
    # Split data
    train_size = int(len(btc_data) * 0.7)
    val_size = int(len(btc_data) * 0.15)
    
    train_data = btc_data[:train_size]
    val_data = btc_data[train_size:train_size+val_size]
    test_data = btc_data[train_size+val_size:]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Train model
    print("\n🎯 Training PPO model...")
    trainer = RLTrainer(
        algorithm="PPO",
        total_timesteps=100000,  # Adjust based on your time/compute
        learning_rate=0.0003
    )
    
    trainer.train(train_data, val_data)
    
    # Evaluate
    print("\n📊 Evaluating on test data...")
    results = trainer.evaluate(test_data, n_episodes=10)
    
    # Save results
    results_file = Path("rl_models") / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to {results_file}")
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)