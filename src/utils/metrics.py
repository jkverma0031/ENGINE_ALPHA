# ==============================================================================
# ENGINE_ALPHA - QUANTITATIVE FINANCIAL METRICS
# Core Component: src/utils/metrics.py
# Description: Enterprise-grade risk management calculators. Includes Maximum 
# Drawdown (MDD), Conditional Value at Risk (CVaR), Sortino Ratios, and Brier 
# Skill Scores for deep probabilistic calibration testing.
# ==============================================================================

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, List

logger = logging.getLogger(__name__)

class QuantMetrics:
    """
    Static calculation engine for financial risk and probabilistic accuracy.
    """

    @staticmethod
    def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
        """
        Calculates Maximum Drawdown (MDD). 
        The largest percentage drop from an all-time high to a subsequent trough.
        """
        if len(equity_curve) < 2:
            return 0.0
            
        rolling_max = np.maximum.accumulate(equity_curve)
        drawdowns = (rolling_max - equity_curve) / rolling_max
        max_drawdown = np.max(drawdowns)
        return float(max_drawdown)

    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        Calculates the Sharpe Ratio to measure risk-adjusted return.
        (Mean Return - Risk Free Rate) / Standard Deviation of Return
        """
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        
        expected_return = np.mean(returns)
        volatility = np.std(returns)
        sharpe = (expected_return - risk_free_rate) / volatility
        
        # Annualization assumes ~2880 bets per day (if 1 bet per 30s)
        annualized_sharpe = sharpe * np.sqrt(2880 * 365)
        return float(annualized_sharpe)

    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, target_return: float = 0.0) -> float:
        """
        Calculates the Sortino Ratio.
        Superior to Sharpe for betting algorithms because it only penalizes 
        DOWNSIDE volatility (losing bets). Winning streaks shouldn't lower the score.
        """
        if len(returns) < 2:
            return 0.0
            
        downside_returns = returns[returns < target_return]
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return float('inf') # Infinite Sortino (No downside risk occurred)
            
        expected_return = np.mean(returns)
        downside_deviation = np.std(downside_returns)
        
        sortino = (expected_return - risk_free_rate) / downside_deviation
        return float(sortino * np.sqrt(2880 * 365))

    @staticmethod
    def calculate_cvar(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculates Conditional Value at Risk (CVaR) / Expected Shortfall.
        Answers: "If the worst 5% of our bets happen, what is our average loss?"
        Crucial for preventing sudden ruin from PRNG black swan events.
        """
        if len(returns) < 2:
            return 0.0
            
        # Sort returns from worst to best
        sorted_returns = np.sort(returns)
        
        # Find the index marking the worst (1 - confidence_level) tail
        tail_index = int((1.0 - confidence_level) * len(sorted_returns))
        
        if tail_index == 0:
            return float(sorted_returns[0])
            
        # Average the returns in that worst-case tail
        cvar = np.mean(sorted_returns[:tail_index])
        return float(cvar)

    @staticmethod
    def brier_skill_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Brier Skill Score (BSS).
        A Brier Score measures how far a probability is from the outcome (0 to 1).
        BSS measures our model's Brier Score against a naive baseline (e.g., guessing 50/50).
        Positive BSS means we possess a mathematical edge. Negative means we are worse than guessing.
        """
        if len(y_true) == 0:
            return 0.0
            
        # Model's Brier Score
        model_brier = np.mean((y_prob - y_true) ** 2)
        
        # Naive Baseline Brier Score (Assuming exactly 50% probability every time)
        baseline_prob = np.full_like(y_prob, 0.5)
        baseline_brier = np.mean((baseline_prob - y_true) ** 2)
        
        if baseline_brier == 0:
            return 0.0
            
        # Calculate Skill Score
        bss = 1.0 - (model_brier / baseline_brier)
        return float(bss)

    @staticmethod
    def expected_value_edge(probability: float, payout_multiplier: float = 0.96) -> float:
        """
        Calculates the true Expected Value (EV) of a specific trade.
        If EV < 0, mathematically you will lose money over infinite trials.
        """
        probability_of_loss = 1.0 - probability
        expected_win = probability * payout_multiplier
        expected_loss = probability_of_loss * 1.0 # You lose 100% of your bet amount
        return float(expected_win - expected_loss)