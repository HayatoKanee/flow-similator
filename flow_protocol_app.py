# flow_protocol_app_complete.py
# Complete Enhanced Version with All Features
# Run with: streamlit run flow_protocol_app_complete.py

import streamlit as st
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import base64
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
from enum import Enum
import networkx as nx
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Enable numba if available for additional speed
try:
    from numba import jit, njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(func):
        return func

st.set_page_config(
    page_title="Flow Protocol Simulator",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .scenario-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============= ENHANCED SCENARIO DEFINITIONS =============

STRESS_TEST_SCENARIOS = {
    "Custom Settings": None,
    "🚀 Micro Pool High APY ($10K)": {
        "initial_pool": 10000,
        "good_users": 80,
        "mostly_good": 18,
        "opportunist": 2,
        "fraud": 0,
        "insurance_target": 10,
        "bull_prob": 30,
        "bear_prob": 10,
        "crash_prob": 2,
        "days": 365,
        "max_coverage_ratio": 0.95,
        "kyc_cost": 500,
        "network_effect": True,
        "wipo_enforcement": False,
        "seed": 42,
        "expected": "80-120% APY, Very High Risk"
    },
    "💎 Small Pool Optimal ($50K)": {
        "initial_pool": 50000,
        "good_users": 75,
        "mostly_good": 20,
        "opportunist": 4,
        "fraud": 1,
        "insurance_target": 10,
        "bull_prob": 20,
        "bear_prob": 15,
        "crash_prob": 5,
        "days": 365,
        "max_coverage_ratio": 0.90,
        "kyc_cost": 750,
        "network_effect": True,
        "wipo_enforcement": True,
        "seed": 42,
        "expected": "40-60% APY, High Utilization"
    },
    "⚡ Aggressive Growth ($25K)": {
        "initial_pool": 25000,
        "good_users": 70,
        "mostly_good": 25,
        "opportunist": 4,
        "fraud": 1,
        "insurance_target": 12,
        "bull_prob": 25,
        "bear_prob": 10,
        "crash_prob": 3,
        "days": 365,
        "max_coverage_ratio": 0.95,
        "kyc_cost": 600,
        "network_effect": True,
        "wipo_enforcement": False,
        "seed": 42,
        "expected": "60-80% APY, Moderate Risk"
    },
    "🛡️ Conservative Starter ($100K)": {
        "initial_pool": 100000,
        "good_users": 80,
        "mostly_good": 17,
        "opportunist": 2,
        "fraud": 1,
        "insurance_target": 15,
        "bull_prob": 15,
        "bear_prob": 20,
        "crash_prob": 5,
        "days": 365,
        "max_coverage_ratio": 0.80,
        "kyc_cost": 1000,
        "network_effect": True,
        "wipo_enforcement": True,
        "seed": 42,
        "expected": "25-35% APY, Low Risk"
    },
    "🔥 Stress Test - High Fraud ($100K)": {
        "initial_pool": 100000,
        "good_users": 40,
        "mostly_good": 25,
        "opportunist": 20,
        "fraud": 15,
        "insurance_target": 20,
        "bull_prob": 10,
        "bear_prob": 30,
        "crash_prob": 10,
        "days": 720,
        "max_coverage_ratio": 0.70,
        "kyc_cost": 1500,
        "network_effect": True,
        "wipo_enforcement": True,
        "seed": 42,
        "expected": "Negative initially, Recovery after 6 months"
    },
    "📈 Scale Test - Start Small ($10K)": {
        "initial_pool": 10000,
        "good_users": 85,
        "mostly_good": 13,
        "opportunist": 2,
        "fraud": 0,
        "insurance_target": 8,
        "bull_prob": 35,
        "bear_prob": 5,
        "crash_prob": 1,
        "days": 180,
        "max_coverage_ratio": 1.0,
        "kyc_cost": 400,
        "network_effect": False,
        "wipo_enforcement": False,
        "seed": 42,
        "expected": "100%+ APY, Then add capital"
    },
    "🌊 Market Crash Test ($50K)": {
        "initial_pool": 50000,
        "good_users": 60,
        "mostly_good": 30,
        "opportunist": 8,
        "fraud": 2,
        "insurance_target": 15,
        "bull_prob": 5,
        "bear_prob": 40,
        "crash_prob": 20,
        "days": 365,
        "max_coverage_ratio": 0.75,
        "kyc_cost": 1000,
        "network_effect": True,
        "wipo_enforcement": True,
        "seed": 42,
        "expected": "10-20% APY in bear market"
    },
    "🎯 Utilization Focus ($75K)": {
        "initial_pool": 75000,
        "good_users": 82,
        "mostly_good": 15,
        "opportunist": 2,
        "fraud": 1,
        "insurance_target": 8,
        "bull_prob": 25,
        "bear_prob": 10,
        "crash_prob": 2,
        "days": 365,
        "max_coverage_ratio": 0.98,
        "kyc_cost": 600,
        "network_effect": True,
        "wipo_enforcement": False,
        "seed": 42,
        "expected": "50-70% APY, 15-20x utilization"
    },
    "🏛️ Institutional Grade ($1M)": {
        "initial_pool": 1000000,
        "good_users": 85,
        "mostly_good": 12,
        "opportunist": 2,
        "fraud": 1,
        "insurance_target": 12,
        "bull_prob": 18,
        "bear_prob": 17,
        "crash_prob": 5,
        "days": 730,
        "max_coverage_ratio": 0.85,
        "kyc_cost": 1200,
        "network_effect": True,
        "wipo_enforcement": True,
        "seed": 42,
        "expected": "15-25% APY, Very stable"
    },
    "⚡ Flash Growth ($20K)": {
        "initial_pool": 20000,
        "good_users": 78,
        "mostly_good": 18,
        "opportunist": 3,
        "fraud": 1,
        "insurance_target": 9,
        "bull_prob": 40,
        "bear_prob": 5,
        "crash_prob": 1,
        "days": 90,
        "max_coverage_ratio": 1.0,
        "kyc_cost": 450,
        "network_effect": False,
        "wipo_enforcement": False,
        "seed": 42,
        "expected": "100%+ APY short-term"
    }
}

# ============= OPTIMIZED SIMULATION CLASSES =============

class MarketCondition(Enum):
    BULL = "bull"
    NORMAL = "normal"
    BEAR = "bear"
    CRASH = "crash"

@dataclass
class MarketState:
    """Enhanced market state with scenario-based transitions"""
    condition: MarketCondition = MarketCondition.NORMAL
    price_multiplier: float = 1.0
    volatility: float = 0.02
    days_in_condition: int = 0
    market_params: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize transition probabilities based on market parameters"""
        if self.market_params:
            self._transition_probs = self._calculate_transition_probs(self.market_params)
        else:
            self._transition_probs = {
                MarketCondition.NORMAL: {"bull": 0.01, "bear": 0.01, "crash": 0.001},
                MarketCondition.BULL: {"normal": 0.02, "bear": 0.005, "crash": 0.002},
                MarketCondition.BEAR: {"normal": 0.02, "bull": 0.005, "crash": 0.01},
                MarketCondition.CRASH: {"bear": 0.05, "normal": 0.02}
            }
    
    def _calculate_transition_probs(self, market_params: Dict) -> Dict:
        """Calculate transition probabilities based on scenario parameters"""
        bull_weight = market_params.get('bull', 0.2)
        bear_weight = market_params.get('bear', 0.15)
        crash_weight = market_params.get('crash', 0.05)
        normal_weight = market_params.get('normal', 0.6)
        
        return {
            MarketCondition.NORMAL: {
                "bull": bull_weight * 0.05,
                "bear": bear_weight * 0.05,
                "crash": crash_weight * 0.02
            },
            MarketCondition.BULL: {
                "normal": normal_weight * 0.1,
                "bear": bear_weight * 0.025,
                "crash": crash_weight * 0.04
            },
            MarketCondition.BEAR: {
                "normal": normal_weight * 0.1,
                "bull": bull_weight * 0.025,
                "crash": crash_weight * 0.2
            },
            MarketCondition.CRASH: {
                "bear": bear_weight * 0.25,
                "normal": normal_weight * 0.1
            }
        }
    
    def update(self, day: int) -> Dict:
        """Update market state with realistic transitions"""
        self.days_in_condition += 1
        
        # Check for transition
        if self.days_in_condition > 30:
            probs = self._transition_probs.get(self.condition, {})
            if probs:
                roll = np.random.random()
                cumsum = 0
                for new_condition, prob in probs.items():
                    cumsum += prob
                    if roll < cumsum:
                        self.condition = MarketCondition(new_condition)
                        self.days_in_condition = 0
                        break
        
        # Pre-calculated parameters for each condition
        if self.condition == MarketCondition.BULL:
            daily_change = np.random.normal(0.002, 0.015)
            self.volatility = 0.03
        elif self.condition == MarketCondition.BEAR:
            daily_change = np.random.normal(-0.001, 0.01)
            self.volatility = 0.025
        elif self.condition == MarketCondition.CRASH:
            daily_change = np.random.normal(-0.01, 0.02)
            self.volatility = 0.05
        else:
            daily_change = np.random.normal(0, 0.01)
            self.volatility = 0.02
        
        self.price_multiplier = np.clip(self.price_multiplier * (1 + daily_change), 0.3, 3.0)
        
        return {
            "condition": self.condition.value,
            "price_multiplier": self.price_multiplier,
            "volatility": self.volatility
        }

@dataclass
class User:
    """Enhanced user model with WIPO support"""
    id: int
    true_type: str
    observed_risk: float = 0.08
    true_risk: float = 0.08
    
    # Use numpy arrays for better performance
    rental_history: np.ndarray = field(default_factory=lambda: np.zeros((0, 7)))
    default_history: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    
    total_coverage_received: float = 0
    total_premiums_paid: float = 0
    
    max_coverage: float = 50
    coverage_growth_rate: float = 1.1
    last_rental_day: int = -30
    account_age: int = 0
    
    preferred_skin_value: float = 100
    price_sensitivity: float = 1.0
    rental_frequency: float = 0.05
    
    connections: Set[int] = field(default_factory=set)
    location: str = ""
    device_fingerprint: str = ""
    
    suspicious_patterns: int = 0
    rapid_value_escalation: bool = False
    network_risk_score: float = 0.0
    
    # Cache frequently accessed values
    _rental_count: int = 0
    _default_count: int = 0
    
    def add_rental(self, rental_data: np.ndarray):
        """Efficiently add rental to history"""
        if self.rental_history.size == 0:
            self.rental_history = rental_data.reshape(1, -1)
        else:
            self.rental_history = np.vstack([self.rental_history, rental_data])
        self._rental_count += 1
        
    def add_default(self, default_data: np.ndarray):
        """Efficiently add default to history"""
        if self.default_history.size == 0:
            self.default_history = default_data.reshape(1, -1)
        else:
            self.default_history = np.vstack([self.default_history, default_data])
        self._default_count += 1
    
    @property
    def rentals(self):
        """Backward compatibility property"""
        return [{'day': r[0], 'skin_value': r[1], 'coverage': r[2], 
                'premium': r[3], 'rental_fee': r[4], 'risk_score': r[5], 
                'return_day': r[6]} for r in self.rental_history]
    
    @property
    def defaults(self):
        """Backward compatibility property"""
        return [{'day': d[0], 'skin_value': d[1], 'coverage': d[2], 
                'market_price': d[3]} for d in self.default_history]
    
    def can_afford_kyc(self, kyc_cost: float) -> bool:
        """Check if user can afford new KYC after ban"""
        if self.true_type == "professional_fraud":
            return np.random.random() < 0.3
        elif self.true_type == "opportunist":
            return np.random.random() < 0.1
        else:
            return False
    
    def calculate_default_probability(self, skin_value: float, 
                                    market_condition: str,
                                    price_multiplier: float,
                                    coverage: float,
                                    wipo_enforcement: bool = True) -> float:
        """Enhanced default probability with WIPO effect"""
        base_risk = self.true_risk
        
        # Market condition effects
        if market_condition == "bull":
            market_multiplier = 1.5
        elif market_condition == "crash":
            market_multiplier = 0.3
        else:
            market_multiplier = 1.0
        
        # Value-based adjustment
        if skin_value > 1000:
            if self.true_type in ["good", "mostly_good"]:
                value_multiplier = 0.1
            else:
                value_multiplier = 0.7
        else:
            value_multiplier = 1.0
        
        # Coverage-based adjustment
        coverage_ratio = coverage / skin_value
        coverage_multiplier = 1.2 if coverage_ratio > 0.8 else 1.0
        
        # Reputation cost
        reputation_multiplier = 0.5 if self._rental_count > 50 and self.max_coverage > 500 else 1.0
        
        # WIPO enforcement effect
        wipo_multiplier = 0.5 if wipo_enforcement and skin_value > 5000 else 1.0
        
        final_risk = base_risk * market_multiplier * value_multiplier * \
                     coverage_multiplier * reputation_multiplier * wipo_multiplier
        
        return min(1.0, final_risk)

class OptimizedUserFactory:
    """Enhanced user factory with pool-size-aware creation"""
    
    def __init__(self):
        self.user_id_counter = 0
        self.locations = np.array(["US", "EU", "ASIA", "LATAM", "OTHER"])
        self.location_probs = np.array([0.3, 0.25, 0.25, 0.15, 0.05])
        self.location_risk = {"US": 1.0, "EU": 1.0, "ASIA": 1.2, "LATAM": 1.5, "OTHER": 2.0}
        
        # Pre-calculate user type parameters
        self.user_type_params = {
            "good": {
                "risk_alpha": 1, "risk_beta": 50,
                "value_mean": 5, "value_std": 1,
                "freq_alpha": 2, "freq_beta": 20
            },
            "mostly_good": {
                "risk_alpha": 2, "risk_beta": 30,
                "value_mean": 5.5, "value_std": 1,
                "freq_alpha": 2, "freq_beta": 15
            },
            "opportunist": {
                "risk_alpha": 5, "risk_beta": 20,
                "value_mean": 6, "value_std": 1,
                "freq_alpha": 3, "freq_beta": 10
            },
            "professional_fraud": {
                "risk_alpha": 20, "risk_beta": 10,
                "value_mean": 6.5, "value_std": 1,
                "freq_alpha": 5, "freq_beta": 10
            }
        }
    
    def create_users_batch(self, n_users: int, user_distribution: Dict, 
                          network: nx.Graph = None, pool_size: float = 100000) -> List[User]:
        """Create multiple users at once for better performance"""
        users = []
        
        # Generate user types in batch
        type_probs = [user_distribution.get(t, 0) for t in 
                     ["good", "mostly_good", "opportunist", "professional_fraud"]]
        user_types = np.random.choice(
            ["good", "mostly_good", "opportunist", "professional_fraud"],
            size=n_users, p=type_probs
        )
        
        # Generate locations in batch
        locations = np.random.choice(self.locations, size=n_users, p=self.location_probs)
        
        for i in range(n_users):
            self.user_id_counter += 1
            user_type = user_types[i]
            params = self.user_type_params[user_type]
            
            # Generate user parameters
            true_risk = np.random.beta(params["risk_alpha"], params["risk_beta"])
            preferred_value = np.random.lognormal(params["value_mean"], params["value_std"])
            rental_freq = np.random.beta(params["freq_alpha"], params["freq_beta"])
            
            # Apply location risk
            location = locations[i]
            true_risk *= self.location_risk[location]
            
            user = User(
                id=self.user_id_counter,
                true_type=user_type,
                true_risk=min(1.0, true_risk),
                preferred_skin_value=preferred_value,
                rental_frequency=rental_freq,
                price_sensitivity=np.random.lognormal(0, 0.5),
                location=location,
                device_fingerprint=f"device_{np.random.randint(1, 10000)}"
            )
            
            # Pool-size-aware coverage limits
            if pool_size < 25000:
                base_max_coverage = 25
                growth_rate = 1.15
            elif pool_size < 50000:
                base_max_coverage = 40
                growth_rate = 1.12
            elif pool_size < 100000:
                base_max_coverage = 60
                growth_rate = 1.10
            elif pool_size < 500000:
                base_max_coverage = 100
                growth_rate = 1.08
            else:
                base_max_coverage = 200
                growth_rate = 1.05
            
            user.max_coverage = base_max_coverage * (1 + np.random.uniform(-0.2, 0.2))
            user.coverage_growth_rate = growth_rate
            
            users.append(user)
            
            # Add to network if provided
            if network is not None:
                network.add_node(user.id, user=user)
        
        # Batch create connections
        if network is not None and len(network) > 1:
            for user in users:
                # Fraudsters more likely to know each other
                if user.true_type == "professional_fraud":
                    n_connections = min(len(network) - 1, np.random.poisson(3))
                else:
                    n_connections = min(len(network) - 1, np.random.poisson(1))
                
                if n_connections > 0:
                    existing_nodes = list(network.nodes())
                    existing_nodes.remove(user.id)
                    if existing_nodes:
                        connections = np.random.choice(
                            existing_nodes, 
                            size=min(n_connections, len(existing_nodes)), 
                            replace=False
                        )
                        for conn in connections:
                            network.add_edge(user.id, conn)
                            user.connections.add(conn)
                            network.nodes[conn]['user'].connections.add(user.id)
        
        return users

class OptimizedRiskEngine:
    """Enhanced risk engine with network effect support"""
    
    def __init__(self, network: nx.Graph, network_effect_enabled: bool = True):
        self.network = network
        self.network_effect_enabled = network_effect_enabled
        self.risk_cache = {}
        self.pattern_memory = defaultdict(list)
        
    def assess_users_batch(self, users: List[User], skin_values: np.ndarray, 
                          market_state: MarketState) -> np.ndarray:
        """Assess multiple users at once"""
        risk_scores = np.zeros(len(users))
        
        for i, (user, skin_value) in enumerate(zip(users, skin_values)):
            # Check cache first
            cache_key = (user.id, int(skin_value), market_state.condition.value)
            if cache_key in self.risk_cache:
                risk_scores[i] = self.risk_cache[cache_key]
            else:
                risk_score = self._calculate_risk(user, skin_value, market_state)
                self.risk_cache[cache_key] = risk_score
                risk_scores[i] = risk_score
                
                # Clear old cache entries periodically
                if len(self.risk_cache) > 10000:
                    self.risk_cache = dict(list(self.risk_cache.items())[-5000:])
        
        return risk_scores
    
    def _calculate_risk(self, user: User, skin_value: float, 
                       market_state: MarketState) -> float:
        """Calculate risk for a single user"""
        # Base risk from history
        if user._rental_count == 0:
            base_risk = 0.08
        else:
            default_rate = user._default_count / user._rental_count
            base_risk = default_rate * 0.7 + 0.08 * 0.3
        
        # Simplified pattern detection
        pattern_risk = 0.0
        if user._rental_count > 3:
            recent_values = user.rental_history[-3:, 1]  # skin values
            if np.all(recent_values < 100) and skin_value > 500:
                pattern_risk += 0.3
        
        # Network effect
        network_risk = 0.0
        if self.network_effect_enabled and user.connections:
            fraud_connections = sum(
                1 for conn_id in user.connections
                if conn_id in self.network.nodes 
                and self.network.nodes[conn_id]['user'].true_type == "professional_fraud"
            )
            if fraud_connections > 0:
                network_risk = min(0.2 * fraud_connections, 0.5)
        
        # Location risk (pre-calculated)
        location_multiplier = {
            "US": 1.0, "EU": 1.0, "ASIA": 1.2, 
            "LATAM": 1.5, "OTHER": 2.0
        }.get(user.location, 1.5)
        
        # Market condition
        market_multiplier = {
            MarketCondition.BULL: 1.3,
            MarketCondition.CRASH: 0.5
        }.get(market_state.condition, 1.0)
        
        # Combine factors
        final_risk = base_risk * (1 + pattern_risk + network_risk) * location_multiplier * market_multiplier
        
        user.observed_risk = min(1.0, final_risk)
        return user.observed_risk

class OptimizedCoverageEngine:
    """Enhanced coverage engine with max_coverage_ratio parameter"""
    
    def __init__(self, risk_engine: OptimizedRiskEngine, pool_size: float = 100000, 
                 max_coverage_ratio: float = 0.95):
        self.risk_engine = risk_engine
        self.pool_size = pool_size
        self.min_coverage = 10
        self.max_coverage_ratio = max_coverage_ratio

        # Dynamic parameters based on pool size
        if pool_size < 25000:
            self.min_coverage = 5
            self.coverage_multiplier = 0.6
        elif pool_size < 50000:
            self.min_coverage = 10
            self.coverage_multiplier = 0.8
        elif pool_size < 100000:
            self.min_coverage = 15
            self.coverage_multiplier = 1.0
        else:
            self.min_coverage = 20
            self.coverage_multiplier = 1.2
        
        # Pre-calculate adjustment factors
        self.market_adjustments = {
            MarketCondition.BULL: 0.8,
            MarketCondition.CRASH: 1.2,
            MarketCondition.NORMAL: 1.0,
            MarketCondition.BEAR: 1.0
        }
    
    def calculate_coverage_batch(self, users: List[User], skin_values: np.ndarray,
                               market_state: MarketState, pool_health: float) -> np.ndarray:
        """Calculate coverage for multiple users at once"""
        n_users = len(users)
        coverages = np.zeros(n_users)
        coverage_ratios = np.zeros(n_users)
        
        # Get risk scores in batch
        risk_scores = self.risk_engine.assess_users_batch(users, skin_values, market_state)
        
        # Vectorized calculations
        market_adj = self.market_adjustments[market_state.condition]
        pool_adj = 0.7 if pool_health < 50 else (1.1 if pool_health > 150 else 1.0)
        
        for i, (user, skin_value, risk_score) in enumerate(zip(users, skin_values, risk_scores)):
            # Base coverage
            if user._rental_count == 0:
                base_coverage = min(50, skin_value * 0.1)
            else:
                successful_rentals = user._rental_count - user._default_count
                trust_factor = 1 - np.exp(-successful_rentals / 20)
                max_user_coverage = user.max_coverage * (1 - risk_score * 0.5)
                base_coverage = max_user_coverage * trust_factor
            
            # Value adjustment
            if skin_value > 5000:
                value_adj = 0.5
            elif skin_value < 100:
                value_adj = 1.2
            else:
                value_adj = 1.0
            
            # Calculate final coverage
            coverage = base_coverage * market_adj * pool_adj * value_adj * self.coverage_multiplier
            coverage = np.clip(coverage, self.min_coverage, skin_value * self.max_coverage_ratio)
            
            coverages[i] = coverage
            coverage_ratios[i] = coverage / skin_value
            
            # Update user's max coverage
            if user._default_count == 0 and user._rental_count > 0:
                user.max_coverage = min(user.max_coverage * user.coverage_growth_rate, 10000)
        
        return coverages, coverage_ratios, risk_scores

class OptimizedFlowProtocolSimulation:
    """Complete simulation engine with all parameters"""
    
    def __init__(self, initial_lp_pool: float = 1_000_000, params: Dict = None):
        # Extract all parameters
        self.params = params or {}
        self.market_params = self.params.get('market', {})
        self.user_distribution = self.params.get('users', {
            'good': 0.60,
            'mostly_good': 0.25,
            'opportunist': 0.10,
            'professional_fraud': 0.05
        })
        
        # Economic parameters
        self.insurance_target = self.params.get('insurance_target', 0.1)
        self.kyc_cost = self.params.get('kyc_cost', 750)
        self.max_coverage_ratio = self.params.get('max_coverage_ratio', 0.95)
        
        # Risk parameters
        self.network_effect_enabled = self.params.get('network_effect', True)
        self.wipo_enforcement = self.params.get('wipo_enforcement', True)
        
        # Core components
        self.user_network = nx.Graph()
        self.user_factory = OptimizedUserFactory()
        self.risk_engine = OptimizedRiskEngine(self.user_network, self.network_effect_enabled)
        self.coverage_engine = OptimizedCoverageEngine(
            self.risk_engine, 
            pool_size=initial_lp_pool,
            max_coverage_ratio=self.max_coverage_ratio
        )
        self.market_state = MarketState(market_params=self.market_params)
        
        # Pools
        self.lp_pool = initial_lp_pool
        self.insurance_reserve = 0
        
        # Optimized tracking
        self.active_rentals_by_day = defaultdict(list)
        self.all_rentals = []
        self.daily_metrics = []
        self.banned_users = set()
        
        # User management
        self.all_users = {}
        self.active_users = set()
        
    def simulate_day(self, day: int) -> Dict:
        """Optimized day simulation"""
        metrics = {
            'day': day,
            'rentals': 0,
            'premiums': 0,
            'losses': 0,
            'defaults': 0,
            'new_users': 0,
            'banned_users': 0,
            'active_users': len(self.active_users),
            'market_condition': self.market_state.condition.value,
            'price_multiplier': self.market_state.price_multiplier
        }
        
        # Update market state
        self.market_state.update(day)
        
        # Process returns
        self._process_returns_optimized(day, metrics)
        
        # User acquisition
        self._acquire_new_users_optimized(day, metrics)
        
        # Process new rentals
        self._process_new_rentals_optimized(day, metrics)
        
        # Update pools
        daily_pnl = metrics['premiums'] - metrics['losses']
        self._update_pools_optimized(daily_pnl)
        
        # Calculate coverage outstanding for utilization
        coverage_outstanding = sum(
            self.all_rentals[idx]['coverage']
            for rentals in self.active_rentals_by_day.values()
            for idx in rentals
        )
        
        # Record final metrics
        metrics.update({
            'lp_pool': self.lp_pool,
            'insurance_reserve': self.insurance_reserve,
            'total_pool': self.lp_pool + self.insurance_reserve,
            'pool_health': (self.insurance_reserve / (self.lp_pool * 0.1)) * 100 if self.lp_pool > 0 else 0,
            'daily_pnl': daily_pnl,
            'coverage_outstanding': coverage_outstanding,
            'utilization_rate': (coverage_outstanding / self.lp_pool) * 100 if self.lp_pool > 0 else 0
        })
        
        self.daily_metrics.append(metrics)
        return metrics
    
    def _process_returns_optimized(self, day: int, metrics: Dict):
        """Optimized return processing"""
        rental_indices = self.active_rentals_by_day.get(day, [])
        if not rental_indices:
            return
        
        # Process in batch
        for idx in rental_indices:
            rental = self.all_rentals[idx]
            user = self.all_users[rental['user_id']]
            
            if user.id in self.banned_users:
                continue
            
            # Calculate default probability with WIPO
            default_prob = user.calculate_default_probability(
                rental['skin_value'],
                self.market_state.condition.value,
                self.market_state.price_multiplier,
                rental['coverage'],
                self.wipo_enforcement
            )
            
            # Check default
            if np.random.random() < default_prob:
                metrics['defaults'] += 1
                metrics['losses'] += rental['coverage']
                
                # Record default
                default_data = np.array([
                    day,
                    rental['skin_value'],
                    rental['coverage'],
                    rental['skin_value'] * self.market_state.price_multiplier
                ])
                user.add_default(default_data)
                
                # Ban user
                self.banned_users.add(user.id)
                self.active_users.discard(user.id)
                metrics['banned_users'] += 1
                
                # Create new identity if applicable
                if user.can_afford_kyc(self.kyc_cost):
                    new_users = self.user_factory.create_users_batch(
                        1, self.user_distribution, self.user_network, self.lp_pool
                    )
                    new_user = new_users[0]
                    new_user.rental_frequency *= np.random.uniform(0.8, 1.2)
                    new_user.preferred_skin_value *= np.random.uniform(0.7, 1.3)
                    
                    self.all_users[new_user.id] = new_user
                    self.active_users.add(new_user.id)
        
        # Clear processed returns
        del self.active_rentals_by_day[day]
    
    def _acquire_new_users_optimized(self, day: int, metrics: Dict):
        """Optimized user acquisition"""
        # Growth curve
        if day < 30:
            base_new_users = 10
        elif day < 90:
            base_new_users = 20
        elif day < 180:
            base_new_users = 30
        else:
            base_new_users = 40
        
        # Market influence
        growth_mult = {
            MarketCondition.BULL: 1.5,
            MarketCondition.CRASH: 0.5
        }.get(self.market_state.condition, 1.0)
        
        # Pool size influence
        if self.lp_pool < 25000:
            size_mult = 0.5
        elif self.lp_pool < 100000:
            size_mult = 0.8
        else:
            size_mult = 1.0
        
        n_new_users = np.random.poisson(base_new_users * growth_mult * size_mult)
        
        if n_new_users > 0:
            # Create users in batch
            new_users = self.user_factory.create_users_batch(
                n_new_users, self.user_distribution, self.user_network, self.lp_pool
            )
            
            for user in new_users:
                self.all_users[user.id] = user
                self.active_users.add(user.id)
            
            metrics['new_users'] = n_new_users
    
    def _process_new_rentals_optimized(self, day: int, metrics: Dict):
        """Optimized rental processing"""
        # Get eligible users
        eligible_users = [
            self.all_users[uid] for uid in self.active_users 
            if uid not in self.banned_users and 
            day - self.all_users[uid].last_rental_day >= 3
        ]
        
        if not eligible_users:
            return
        
        # Vectorized rental decisions
        n_users = len(eligible_users)
        rental_decisions = np.random.random(n_users) < np.array([u.rental_frequency for u in eligible_users])
        
        renting_users = [u for u, decision in zip(eligible_users, rental_decisions) if decision]
        
        if not renting_users:
            return
        
        # Generate skin values for all renting users
        skin_values = np.array([
            np.clip(np.random.lognormal(np.log(u.preferred_skin_value), 0.5), 10, 10000)
            for u in renting_users
        ])
        
        # Calculate rental fees
        daily_rates = np.clip(np.random.normal(0.001, 0.0002, len(renting_users)), 0.0005, 0.002)
        rental_fees = skin_values * daily_rates * 7
        
        # Get coverage decisions in batch
        pool_health = (self.insurance_reserve / (self.lp_pool * 0.1)) * 100 if self.lp_pool > 0 else 0
        coverages, coverage_ratios, risk_scores = self.coverage_engine.calculate_coverage_batch(
            renting_users, skin_values, self.market_state, pool_health
        )
        
        # Calculate premiums
        expected_losses = coverages * risk_scores
        margins = np.where(risk_scores < 0.05, 1.5,
                          np.where(risk_scores < 0.10, 2.0,
                                  np.where(risk_scores < 0.20, 2.5, 3.0)))
        
        volatility_adj = 1 + self.market_state.volatility * 2
        margins *= volatility_adj
        margins = np.clip(margins, 1.3, 3.0)
        
        base_premiums = expected_losses * margins
        max_premiums = rental_fees * np.where(risk_scores > 0.15, 5.0, 3.0)
        premiums = np.minimum(base_premiums, max_premiums)
        
        # User acceptance decisions
        total_costs = rental_fees + premiums
        deposits_saved = skin_values - coverages
        value_scores = np.where(deposits_saved > 0, 
                               deposits_saved / (total_costs * np.array([u.price_sensitivity for u in renting_users])),
                               0)
        
        accept_probs = np.where(value_scores > 10, 0.9,
                               np.where(value_scores > 5, 0.7,
                                       np.where(value_scores > 2, 0.5,
                                               np.where(value_scores > 1, 0.3, 0.1))))
        
        acceptances = np.random.random(len(renting_users)) < accept_probs
        
        # Create accepted rentals
        for i, accepted in enumerate(acceptances):
            if accepted:
                user = renting_users[i]
                
                # Create rental
                rental = {
                    'user_id': user.id,
                    'day': day,
                    'skin_value': skin_values[i],
                    'coverage': coverages[i],
                    'premium': premiums[i],
                    'rental_fee': rental_fees[i],
                    'risk_score': risk_scores[i],
                    'return_day': day + 7
                }
                
                # Store rental
                rental_idx = len(self.all_rentals)
                self.all_rentals.append(rental)
                self.active_rentals_by_day[day + 7].append(rental_idx)
                
                # Update user
                rental_data = np.array([
                    day, skin_values[i], coverages[i], premiums[i],
                    rental_fees[i], risk_scores[i], day + 7
                ])
                user.add_rental(rental_data)
                user.last_rental_day = day
                user.total_coverage_received += coverages[i]
                user.total_premiums_paid += premiums[i]
                
                # Update metrics
                metrics['rentals'] += 1
                metrics['premiums'] += premiums[i]
    
    def _update_pools_optimized(self, daily_pnl: float):
        """Optimized pool updates"""
        if daily_pnl > 0:
            # Calculate insurance allocation
            target_reserve = self.lp_pool * self.insurance_target
            current_ratio = self.insurance_reserve / target_reserve if target_reserve > 0 else 0
            
            if current_ratio < 0.5:
                insurance_rate = 0.20
            elif current_ratio < 0.8:
                insurance_rate = 0.15
            elif current_ratio < 1.2:
                insurance_rate = 0.10
            else:
                insurance_rate = 0.05
            
            to_insurance = daily_pnl * insurance_rate
            to_lp = daily_pnl - to_insurance
            
            self.insurance_reserve += to_insurance
            self.lp_pool += to_lp
        else:
            loss = abs(daily_pnl)
            if loss <= self.insurance_reserve:
                self.insurance_reserve -= loss
            else:
                remaining_loss = loss - self.insurance_reserve
                self.insurance_reserve = 0
                self.lp_pool = max(0, self.lp_pool - remaining_loss)
    
    def run_simulation(self, days: int = 365, progress_callback=None):
        """Run optimized simulation"""
        for day in range(days):
            metrics = self.simulate_day(day)
            
            if progress_callback:
                progress_callback(day / days)
        
        return pd.DataFrame(self.daily_metrics)

# ============= PARAMETER VALIDATION =============

def validate_parameters(params: Dict) -> Tuple[List[str], List[str]]:
    """Validate parameter combinations"""
    warnings = []
    errors = []
    
    # Check user distribution
    if params.get('fraud', 0) > 20:
        warnings.append("⚠️ High fraud rate (>20%) will likely result in losses")
    
    # Check market conditions
    if params.get('crash_prob', 0) > 15:
        warnings.append("⚠️ High crash probability may severely impact returns")
    
    # Check pool size vs parameters
    if params.get('initial_pool', 100000) < 50000 and params.get('fraud', 0) > 5:
        warnings.append("⚠️ Small pools are vulnerable to fraud - consider reducing fraud %")
    
    # Check coverage ratio
    if params.get('max_coverage_ratio', 0.95) > 0.9 and params.get('fraud', 0) > 10:
        errors.append("❌ High coverage with high fraud is extremely risky")
    
    # Check insurance target
    bear_crash = params.get('bear_prob', 0) + params.get('crash_prob', 0)
    if params.get('insurance_target', 10) < 10 and bear_crash > 30:
        warnings.append("⚠️ Low insurance reserve in volatile markets is risky")
    
    return warnings, errors

# ============= ADVANCED ANALYTICS (same as before) =============
# [Previous add_advanced_analytics function remains the same]

# ============= STREAMLIT UI =============

# Title and description
st.title("🔄 Flow Protocol Interactive Simulator")
st.markdown("""
Explore how Flow Protocol performs under different market conditions and parameters. 
This **complete enhanced version** includes all scenario overrides, parameter validation, and advanced analytics.
""")

# Sidebar for parameters
st.sidebar.header("⚙️ Simulation Parameters")

# Enhanced Scenario Selection
st.sidebar.subheader("📋 Scenario Selection")

# Group scenarios by category
scenario_categories = {
    "High Risk / High Return": [
        "🚀 Micro Pool High APY ($10K)",
        "⚡ Flash Growth ($20K)",
        "⚡ Aggressive Growth ($25K)"
    ],
    "Balanced Approach": [
        "💎 Small Pool Optimal ($50K)",
        "🎯 Utilization Focus ($75K)",
        "🛡️ Conservative Starter ($100K)"
    ],
    "Stress Testing": [
        "🔥 Stress Test - High Fraud ($100K)",
        "🌊 Market Crash Test ($50K)",
        "📈 Scale Test - Start Small ($10K)"
    ],
    "Low Risk / Institutional": [
        "🏛️ Institutional Grade ($1M)"
    ]
}

# First select category
category = st.sidebar.selectbox(
    "Select Category",
    ["Custom Settings"] + list(scenario_categories.keys()),
    help="Choose a category based on your risk tolerance and goals"
)

if category == "Custom Settings":
    scenario_choice = "Custom Settings"
    st.sidebar.info("👉 Adjust all parameters manually using the controls below")
    scenario_params = None
else:
    # Then select scenario within category
    scenario_choice = st.sidebar.selectbox(
        "Select Scenario",
        scenario_categories[category],
        help="Pre-configured scenarios for different testing purposes"
    )
    
    # Display scenario details if selected
    if scenario_choice in STRESS_TEST_SCENARIOS:
        scenario_params = STRESS_TEST_SCENARIOS[scenario_choice]
        
        # Create a nice display of what will be applied
        st.sidebar.markdown("### 🎯 Scenario Details")
        
        # Use columns for better layout
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.markdown("**Economics**")
            st.markdown(f"💰 Pool: ${scenario_params['initial_pool']:,}")
            st.markdown(f"🛡️ Insurance: {scenario_params['insurance_target']}%")
            st.markdown(f"🆔 KYC: ${scenario_params.get('kyc_cost', 750)}")
        
        with col2:
            st.markdown("**Risk Profile**")
            st.markdown(f"✅ Good: {scenario_params['good_users']}%")
            st.markdown(f"❌ Fraud: {scenario_params['fraud']}%")
            st.markdown(f"📊 Coverage: {int(scenario_params.get('max_coverage_ratio', 0.95)*100)}%")
        
        # Market conditions bar
        st.sidebar.markdown("**Market Mix**")
        market_data = {
            'Bull': scenario_params['bull_prob'],
            'Normal': 100 - scenario_params['bull_prob'] - scenario_params['bear_prob'] - scenario_params['crash_prob'],
            'Bear': scenario_params['bear_prob'],
            'Crash': scenario_params['crash_prob']
        }
        
        # Create a simple bar visualization
        market_colors = {
            'Bull': '🟢',
            'Normal': '🔵',
            'Bear': '🟡',
            'Crash': '🔴'
        }
        
        for market, prob in market_data.items():
            if prob > 0:
                bar = market_colors[market] * int(prob/5)  # Simple text bar
                st.sidebar.markdown(f"{market}: {bar} {prob}%")
        
        st.sidebar.markdown(f"**Duration**: {scenario_params['days']} days")
        st.sidebar.success(f"**Expected**: {scenario_params['expected']}")
        
        # Warning about override
        st.sidebar.warning("⚠️ This will override manual slider settings!")

def add_advanced_analytics(st, df, sim, results):
    """Advanced analytics with all the missing charts"""
    
    # 1. UTILIZATION METRICS
    st.markdown("### 📊 Capital Utilization Analysis")
    
    # Use pre-calculated utilization from simulation
    if 'utilization_rate' in df.columns:
        # Already calculated in optimized version
        pass
    else:
        # Calculate if not present
        total_coverage_outstanding = []
        for day in range(len(df)):
            coverage_on_day = sum(
                rental['coverage'] 
                for return_day, rental_indices in sim.active_rentals_by_day.items() 
                for idx in rental_indices
                for rental in [sim.all_rentals[idx]]
                if rental['day'] <= day < rental['return_day']
            )
            total_coverage_outstanding.append(coverage_on_day)
        
        df['coverage_outstanding'] = total_coverage_outstanding
        df['utilization_rate'] = (df['coverage_outstanding'] / df['lp_pool']) * 100
    
    # Utilization chart
    fig_util = go.Figure()
    fig_util.add_trace(go.Scatter(
        x=df['day'], 
        y=df['utilization_rate'],
        mode='lines',
        name='Utilization Rate (%)',
        line=dict(color='rgb(255, 99, 132)', width=3)
    ))
    
    # Add target zones
    fig_util.add_hrect(y0=800, y1=1500, fillcolor="green", opacity=0.2, 
                      annotation_text="Optimal Zone (8-15x)")
    fig_util.add_hline(y=1000, line_dash="dash", line_color="green",
                      annotation_text="Target 10x")
    
    fig_util.update_layout(
        title='Capital Utilization Rate (Coverage Outstanding / LP Pool)',
        xaxis_title='Day',
        yaxis_title='Utilization Rate (%)',
        height=400,
        template='plotly_white'
    )
    st.plotly_chart(fig_util, use_container_width=True)
    
    # Key utilization metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Final Utilization", f"{df['utilization_rate'].iloc[-1]:.0f}%")
    col2.metric("Avg Utilization", f"{df['utilization_rate'].mean():.0f}%")
    col3.metric("Peak Utilization", f"{df['utilization_rate'].max():.0f}%")
    
    # Add utilization insight
    avg_util = df['utilization_rate'].mean()
    if avg_util < 500:
        st.warning(f"⚠️ **Low Utilization**: Average {avg_util:.0f}% is well below the 800-1500% target. This explains the lower returns. Consider: increasing max coverage ratios, reducing insurance premiums, or improving user acquisition.")
    elif avg_util > 1500:
        st.info(f"📈 **High Utilization**: Average {avg_util:.0f}% exceeds target. While profitable, consider risk management measures.")
    else:
        st.success(f"✅ **Optimal Utilization**: Average {avg_util:.0f}% is within the target range of 800-1500%.")
    
    # 2. COVERAGE DISTRIBUTION ANALYSIS
    st.markdown("### 🎯 Coverage Distribution Analysis")
    
    # Get all rentals efficiently
    all_rentals = sim.all_rentals if hasattr(sim, 'all_rentals') else []
    
    if all_rentals:
        coverage_ratios = [(r['coverage'] / r['skin_value']) * 100 for r in all_rentals]
        
        # Histogram of coverage ratios
        fig_coverage = go.Figure()
        fig_coverage.add_trace(go.Histogram(
            x=coverage_ratios,
            nbinsx=20,
            name='Coverage Ratio Distribution',
            marker_color='rgba(54, 162, 235, 0.7)'
        ))
        
        fig_coverage.update_layout(
            title='Distribution of Coverage Ratios',
            xaxis_title='Coverage Ratio (%)',
            yaxis_title='Number of Rentals',
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig_coverage, use_container_width=True)
        
        # Coverage evolution by user trust level
        st.markdown("#### Coverage Evolution by User Trust Level")
        
        # Group users by rental count
        trust_levels = {
            'New (0-5 rentals)': [],
            'Building (6-20 rentals)': [],
            'Established (21-50 rentals)': [],
            'Elite (50+ rentals)': []
        }
        
        for rental in all_rentals:
            user = sim.all_users.get(rental['user_id'])
            if user:
                rental_count = user._rental_count if hasattr(user, '_rental_count') else len(user.rentals)
                coverage_ratio = (rental['coverage'] / rental['skin_value']) * 100
                
                if rental_count <= 5:
                    trust_levels['New (0-5 rentals)'].append(coverage_ratio)
                elif rental_count <= 20:
                    trust_levels['Building (6-20 rentals)'].append(coverage_ratio)
                elif rental_count <= 50:
                    trust_levels['Established (21-50 rentals)'].append(coverage_ratio)
                else:
                    trust_levels['Elite (50+ rentals)'].append(coverage_ratio)
        
        # Box plot
        fig_trust = go.Figure()
        for level, ratios in trust_levels.items():
            if ratios:
                fig_trust.add_trace(go.Box(
                    y=ratios,
                    name=level,
                    boxmean='sd'
                ))
        
        fig_trust.update_layout(
            title='Coverage Ratio by User Trust Level',
            yaxis_title='Coverage Ratio (%)',
            height=400,
            showlegend=False,
            template='plotly_white'
        )
        st.plotly_chart(fig_trust, use_container_width=True)
    
    # 3. COHORT ANALYSIS
    st.markdown("### 👥 User Cohort Analysis")
    
    # Create monthly cohorts
    cohort_data = {}
    for user in sim.all_users.values():
        rental_count = user._rental_count if hasattr(user, '_rental_count') else len(user.rentals)
        if rental_count > 0:
            first_rental_day = user.rental_history[0, 0] if hasattr(user, 'rental_history') and len(user.rental_history) > 0 else user.rentals[0]['day']
            first_rental_month = int(first_rental_day // 30)
            cohort_key = f"Month {first_rental_month + 1}"
            
            if cohort_key not in cohort_data:
                cohort_data[cohort_key] = {
                    'users': 0,
                    'defaults': 0,
                    'total_premiums': 0,
                    'total_coverage': 0,
                    'retention': 0
                }
            
            default_count = user._default_count if hasattr(user, '_default_count') else len(user.defaults)
            cohort_data[cohort_key]['users'] += 1
            cohort_data[cohort_key]['defaults'] += default_count
            cohort_data[cohort_key]['total_premiums'] += user.total_premiums_paid
            cohort_data[cohort_key]['total_coverage'] += user.total_coverage_received
            
            # Check if still active
            if user.id not in sim.banned_users and user.last_rental_day > len(df) - 30:
                cohort_data[cohort_key]['retention'] += 1
    
    # Display cohort metrics
    if cohort_data:
        cohort_df = pd.DataFrame.from_dict(cohort_data, orient='index')
        cohort_df['default_rate'] = (cohort_df['defaults'] / cohort_df['users'] * 100).round(1)
        cohort_df['retention_rate'] = (cohort_df['retention'] / cohort_df['users'] * 100).round(1)
        cohort_df['avg_premium'] = (cohort_df['total_premiums'] / cohort_df['users']).round(0)
        
        # Cohort performance chart
        fig_cohort = go.Figure()
        fig_cohort.add_trace(go.Bar(
            x=cohort_df.index,
            y=cohort_df['retention_rate'],
            name='Retention Rate (%)',
            marker_color='rgba(75, 192, 192, 0.7)',
            yaxis='y'
        ))
        fig_cohort.add_trace(go.Scatter(
            x=cohort_df.index,
            y=cohort_df['default_rate'],
            name='Default Rate (%)',
            line=dict(color='rgb(255, 99, 132)', width=2),
            yaxis='y2'
        ))
        
        fig_cohort.update_layout(
            title='Cohort Performance Analysis',
            xaxis_title='User Cohort',
            yaxis=dict(title='Retention Rate (%)', side='left'),
            yaxis2=dict(title='Default Rate (%)', side='right', overlaying='y'),
            height=400,
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig_cohort, use_container_width=True)
        
        # Cohort summary table
        display_cohort = cohort_df[['users', 'default_rate', 'retention_rate', 'avg_premium']].copy()
        display_cohort.columns = ['Users', 'Default Rate (%)', 'Retention Rate (%)', 'Avg Premium ($)']
        st.dataframe(display_cohort, use_container_width=True)
    
    # 4. PREMIUM EFFICIENCY ANALYSIS
    st.markdown("### 💰 Premium Efficiency Analysis")
    
    # Calculate loss ratio by coverage tier
    coverage_tiers = {
        '0-20%': {'premiums': 0, 'losses': 0, 'count': 0},
        '20-40%': {'premiums': 0, 'losses': 0, 'count': 0},
        '40-60%': {'premiums': 0, 'losses': 0, 'count': 0},
        '60-80%': {'premiums': 0, 'losses': 0, 'count': 0},
        '80-100%': {'premiums': 0, 'losses': 0, 'count': 0}
    }
    
    for rental in all_rentals:
        coverage_ratio = (rental['coverage'] / rental['skin_value']) * 100
        
        if coverage_ratio <= 20:
            tier = '0-20%'
        elif coverage_ratio <= 40:
            tier = '20-40%'
        elif coverage_ratio <= 60:
            tier = '40-60%'
        elif coverage_ratio <= 80:
            tier = '60-80%'
        else:
            tier = '80-100%'
        
        coverage_tiers[tier]['premiums'] += rental['premium']
        coverage_tiers[tier]['count'] += 1
        
        # Check if this rental defaulted
        user = sim.all_users.get(rental['user_id'])
        if user:
            # Check defaults
            if hasattr(user, 'default_history') and len(user.default_history) > 0:
                default_days = user.default_history[:, 0]
                if rental['return_day'] in default_days:
                    coverage_tiers[tier]['losses'] += rental['coverage']
    
    # Create tier analysis chart
    tier_names = list(coverage_tiers.keys())
    loss_ratios = []
    rental_counts = []
    
    for tier in tier_names:
        if coverage_tiers[tier]['premiums'] > 0:
            loss_ratio = (coverage_tiers[tier]['losses'] / coverage_tiers[tier]['premiums']) * 100
        else:
            loss_ratio = 0
        loss_ratios.append(loss_ratio)
        rental_counts.append(coverage_tiers[tier]['count'])
    
    # Dual axis chart
    fig_tiers = go.Figure()
    fig_tiers.add_trace(go.Bar(
        x=tier_names,
        y=rental_counts,
        name='Rental Count',
        marker_color='rgba(54, 162, 235, 0.7)',
        yaxis='y'
    ))
    fig_tiers.add_trace(go.Scatter(
        x=tier_names,
        y=loss_ratios,
        name='Loss Ratio (%)',
        line=dict(color='rgb(255, 99, 132)', width=3),
        yaxis='y2'
    ))
    fig_tiers.add_hline(y=60, line_dash="dash", line_color="green",
                       annotation_text="Target 60%", yref='y2')
    
    fig_tiers.update_layout(
        title='Performance by Coverage Tier',
        xaxis_title='Coverage Tier',
        yaxis=dict(title='Number of Rentals', side='left'),
        yaxis2=dict(title='Loss Ratio (%)', side='right', overlaying='y'),
        height=400,
        hovermode='x unified',
        template='plotly_white'
    )
    st.plotly_chart(fig_tiers, use_container_width=True)
    
    # 5. STRESS TESTING VISUALIZATION
    st.markdown("### 🔥 Stress Testing & Risk Scenarios")
    
    # Calculate maximum drawdown periods
    rolling_max = df['lp_pool'].expanding().max()
    df['drawdown'] = ((df['lp_pool'] - rolling_max) / rolling_max) * 100
    
    # Drawdown chart
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=df['day'],
        y=df['drawdown'],
        mode='lines',
        fill='tozeroy',
        name='Drawdown (%)',
        line=dict(color='rgb(255, 99, 132)', width=2)
    ))
    
    # Mark stress periods
    stress_periods = df[df['drawdown'] < -5]
    if not stress_periods.empty:
        for _, period in stress_periods.iterrows():
            fig_dd.add_annotation(
                x=period['day'],
                y=period['drawdown'],
                text=f"{period['market_condition']}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red"
            )
    
    fig_dd.update_layout(
        title='Pool Drawdown Analysis',
        xaxis_title='Day',
        yaxis_title='Drawdown (%)',
        height=400,
        template='plotly_white'
    )
    st.plotly_chart(fig_dd, use_container_width=True)
    
    # Risk metrics summary
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Max Drawdown", f"{df['drawdown'].min():.1f}%")
    col2.metric("Days in Drawdown", f"{(df['drawdown'] < 0).sum()}")
    
    # Recovery time calculation
    recovery_indices = df[df['drawdown'] >= 0].index
    if len(recovery_indices) > 0:
        recovery_time = recovery_indices[-1]
    else:
        recovery_time = "N/A"
    col3.metric("Recovery Time", f"{recovery_time} days")
    
    # Sharpe ratio calculation
    if df['daily_pnl'].std() > 0:
        sharpe = (results['annualized_return'] - 5) / (df['daily_pnl'].std() * np.sqrt(365))
    else:
        sharpe = 0
    col4.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    # 6. REVENUE BREAKDOWN
    st.markdown("### 💸 Revenue & Efficiency Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue composition pie chart
        fig_revenue = go.Figure()
        fig_revenue.add_trace(go.Pie(
            labels=['Insurance Premiums', 'Rental Fees (est)'],
            values=[
                results['total_premiums'],
                results['total_rentals'] * 50 * 0.001 * 7  # Rough estimate
            ],
            marker_colors=['rgba(54, 162, 235, 0.8)', 'rgba(75, 192, 192, 0.8)'],
            hole=0.3
        ))
        fig_revenue.update_layout(
            title='Revenue Sources',
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    with col2:
        # Capital efficiency metrics
        st.markdown("#### Capital Efficiency Metrics")
        
        revenue_per_dollar = (results['total_premiums'] / results['initial_pool']) * (365 / len(df))
        coverage_per_dollar = df['coverage_outstanding'].mean() / results['initial_pool']
        
        efficiency_metrics = pd.DataFrame({
            'Metric': [
                'Revenue per LP Dollar (Annual)',
                'Coverage per LP Dollar',
                'Avg Premium Rate',
                'Capital Turnover'
            ],
            'Value': [
                f"${revenue_per_dollar:.3f}",
                f"{coverage_per_dollar:.1f}x",
                f"{(results['total_premiums'] / results['avg_coverage'] / results['total_rentals'] * 100):.2f}%",
                f"{df['utilization_rate'].mean() / 100:.1f}x"
            ]
        })
        
        st.dataframe(efficiency_metrics, use_container_width=True, hide_index=True)
# Manual Parameter Controls
if scenario_choice == "Custom Settings":
    # Market Parameters
    st.sidebar.subheader("Market Conditions")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        bull_prob = st.slider("Bull Market %", 0, 50, 20, 5)
        bear_prob = st.slider("Bear Market %", 0, 50, 15, 5)
    with col2:
        crash_prob = st.slider("Crash %", 0, 20, 5, 1)
        normal_prob = 100 - bull_prob - bear_prob - crash_prob
        st.metric("Normal %", normal_prob)
    
    # User Distribution
    st.sidebar.subheader("User Distribution")
    good_users = st.slider("Good Users %", 40, 80, 60, 5)
    mostly_good = st.slider("Mostly Good %", 10, 40, 25, 5)
    opportunist = st.slider("Opportunists %", 5, 20, 10, 5)
    fraud = 100 - good_users - mostly_good - opportunist
    st.sidebar.metric("Fraudsters %", fraud)
    
    # Visual representation
    user_bar = "🟢" * int(good_users/5) + "🟡" * int(mostly_good/5) + "🟠" * int(opportunist/5) + "🔴" * int(fraud/5)
    st.sidebar.markdown(f"Distribution: {user_bar}")
    
    # Economic Parameters
    st.sidebar.subheader("Economic Parameters")
    
    # Pool size selection
    pool_preset = st.sidebar.radio(
        "Select Pool Size",
        ["$10K", "$25K", "$50K", "$100K", "$250K", "$500K", "$1M", "Custom"],
        index=3,
        horizontal=True
    )
    
    pool_values = {
        "$10K": 10000, "$25K": 25000, "$50K": 50000,
        "$100K": 100000, "$250K": 250000, "$500K": 500000, "$1M": 1000000
    }
    
    if pool_preset == "Custom":
        initial_pool = st.sidebar.number_input(
            "Custom LP Pool ($)",
            min_value=5000,
            max_value=10000000,
            value=100000,
            step=5000
        )
    else:
        initial_pool = pool_values[pool_preset]
    
    kyc_cost = st.slider("KYC Cost ($)", 100, 2000, 750, 50)
    insurance_target = st.slider("Insurance Target %", 5, 20, 10, 1)
    
    # Risk Parameters
    st.sidebar.subheader("Risk Parameters")
    max_coverage_ratio = st.slider("Max Coverage Ratio", 0.5, 1.0, 0.95, 0.05)
    network_effect = st.checkbox("Enable Network Analysis", True)
    wipo_enforcement = st.checkbox("WIPO for High-Value Items", True)
    
    # Simulation Controls
    st.sidebar.subheader("Simulation")
    simulation_days = st.slider("Days to Simulate", 30, 730, 365, 30)
    random_seed = st.number_input("Random Seed", 0, 9999, 42)
else:
    # Initialize with defaults that will be overridden
    bull_prob = 20
    bear_prob = 15
    crash_prob = 5
    normal_prob = 60
    good_users = 60
    mostly_good = 25
    opportunist = 10
    fraud = 5
    initial_pool = 100000
    kyc_cost = 750
    insurance_target = 10
    max_coverage_ratio = 0.95
    network_effect = True
    wipo_enforcement = True
    simulation_days = 365
    random_seed = 42

# Show pool size guidance
pool_guidance = {
    10000: {"users": "200-500", "coverage": "$20-50", "apy": "80-120%", "risk": "Very High"},
    25000: {"users": "500-1000", "coverage": "$30-80", "apy": "60-80%", "risk": "High"},
    50000: {"users": "1000-2000", "coverage": "$50-150", "apy": "40-60%", "risk": "Moderate"},
    100000: {"users": "2000-4000", "coverage": "$100-300", "apy": "25-40%", "risk": "Moderate"},
    250000: {"users": "5000-8000", "coverage": "$200-500", "apy": "20-30%", "risk": "Low"},
    500000: {"users": "8000-15000", "coverage": "$300-800", "apy": "15-25%", "risk": "Low"},
    1000000: {"users": "15000-30000", "coverage": "$500-1500", "apy": "10-20%", "risk": "Very Low"}
}

# Find closest guidance
closest_pool = min(pool_guidance.keys(), key=lambda x: abs(x - initial_pool))
guidance = pool_guidance[closest_pool]

st.sidebar.markdown(f"""
💡 **Pool Size: ${initial_pool:,}**
- Target Users: {guidance['users']}
- Avg Coverage: {guidance['coverage']}
- Expected APY: {guidance['apy']}
- Risk Level: {guidance['risk']}
""")

# Add update frequency control
st.sidebar.markdown("### ⚡ Real-Time Updates")
update_freq_pct = st.sidebar.slider(
    "Update Frequency (%)", 
    1, 10, 2, 1,
    help="How often to update charts during simulation (lower = more updates)"
)
st.sidebar.info("Charts will update in real-time as the simulation runs!")

# Run simulation button
if st.sidebar.button("🚀 Run Simulation", type="primary"):
    # Collect all current parameter values
    current_params = {
        'good_users': good_users,
        'mostly_good': mostly_good,
        'opportunist': opportunist,
        'fraud': fraud,
        'bull_prob': bull_prob,
        'bear_prob': bear_prob,
        'crash_prob': crash_prob,
        'initial_pool': initial_pool,
        'insurance_target': insurance_target,
        'kyc_cost': kyc_cost,
        'max_coverage_ratio': max_coverage_ratio,
        'network_effect': network_effect,
        'wipo_enforcement': wipo_enforcement,
        'simulation_days': simulation_days,
        'random_seed': random_seed
    }
    
    # Apply scenario overrides if selected
    if scenario_params:
        st.info(f"🎯 Applying scenario: {scenario_choice}")
        
        # Override all available parameters
        for key in ['good_users', 'mostly_good', 'opportunist', 'fraud',
                   'bull_prob', 'bear_prob', 'crash_prob', 'initial_pool',
                   'insurance_target', 'kyc_cost', 'max_coverage_ratio',
                   'network_effect', 'wipo_enforcement', 'seed']:
            if key in scenario_params:
                if key == 'seed':
                    current_params['random_seed'] = scenario_params[key]
                elif key == 'days':
                    current_params['simulation_days'] = scenario_params[key]
                else:
                    current_params[key] = scenario_params[key]
        
        # Override days if specified
        if 'days' in scenario_params:
            current_params['simulation_days'] = scenario_params['days']
    
    # Recalculate normal probability
    current_params['normal_prob'] = 100 - current_params['bull_prob'] - \
                                   current_params['bear_prob'] - current_params['crash_prob']
    
    # Validate parameters
    warnings, errors = validate_parameters(current_params)
    
    if errors:
        for error in errors:
            st.error(error)
        st.stop()
    
    if warnings:
        for warning in warnings:
            st.warning(warning)
    
    # Create simulation parameters
    sim_params = {
        'market': {
            'bull': current_params['bull_prob'] / 100,
            'bear': current_params['bear_prob'] / 100,
            'crash': current_params['crash_prob'] / 100,
            'normal': current_params['normal_prob'] / 100
        },
        'users': {
            'good': current_params['good_users'] / 100,
            'mostly_good': current_params['mostly_good'] / 100,
            'opportunist': current_params['opportunist'] / 100,
            'professional_fraud': current_params['fraud'] / 100
        },
        'initial_pool': current_params['initial_pool'],
        'kyc_cost': current_params['kyc_cost'],
        'insurance_target': current_params['insurance_target'] / 100,
        'max_coverage_ratio': current_params['max_coverage_ratio'],
        'network_effect': current_params['network_effect'],
        'wipo_enforcement': current_params['wipo_enforcement'],
        'days': current_params['simulation_days'],
        'seed': current_params['random_seed']
    }
    
    # Store in session state
    st.session_state['sim_params'] = sim_params
    st.session_state['scenario_name'] = scenario_choice
    
    # Initialize simulation with all parameters
    np.random.seed(current_params['random_seed'])
    sim = OptimizedFlowProtocolSimulation(
        initial_lp_pool=current_params['initial_pool'],
        params=sim_params
    )
    
    # Create containers for real-time updates
    if scenario_choice != "Custom Settings":
        st.markdown(f"### 🎯 Running: {scenario_choice}")
    else:
        st.markdown("### 🔄 Running Custom Simulation")
    
    # Time estimate
    estimated_time = current_params['simulation_days'] * 0.002
    st.info(f"⏱️ Estimated time: {estimated_time:.1f} seconds")
    
    # Progress and status
    col1, col2 = st.columns([3, 1])
    with col1:
        progress_bar = st.progress(0)
        status_text = st.empty()
    with col2:
        current_day_display = st.empty()
    
    # Key metrics containers
    st.markdown("### 📊 Live Metrics")
    metrics_container = st.container()
    with metrics_container:
        metric_cols = st.columns(6)
        metric_displays = [col.empty() for col in metric_cols]
    
    # Chart containers
    st.markdown("### 📈 Real-Time Charts")
    last_update_text = st.empty()
    
    # Main chart tabs
    chart_tabs = st.tabs(["📈 Pool & Returns", "📊 Utilization", "👥 Users", "🎯 Risk"])
    chart_placeholders = [tab.empty() for tab in chart_tabs]
    
    # Run simulation day by day
    daily_metrics = []
    update_frequency = max(1, int(current_params['simulation_days'] * update_freq_pct / 100))
    start_time = datetime.now()
    
    for day in range(current_params['simulation_days']):
        # Simulate one day
        metrics = sim.simulate_day(day)
        daily_metrics.append(metrics)
        
        # Update progress
        progress = (day + 1) / current_params['simulation_days']
        progress_bar.progress(progress)
        current_day_display.metric("Day", f"{day + 1} / {current_params['simulation_days']}")
        
        # Update status periodically
        if day % 30 == 0:
            status_text.info(f"📅 Month {day // 30 + 1}: {metrics['active_users']} active users, "
                           f"Market: {metrics['market_condition'].upper()}, "
                           f"Utilization: {metrics['utilization_rate']:.0f}%")
        
        # Update charts periodically or on last day
        if day % update_frequency == 0 or day == current_params['simulation_days'] - 1:
            df_current = pd.DataFrame(daily_metrics)
            
            # Calculate current metrics
            current_return = (metrics['lp_pool'] / current_params['initial_pool'] - 1) * 100 * (365 / (day + 1))
            current_loss_ratio = (df_current['losses'].sum() / df_current['premiums'].sum() * 100) if df_current['premiums'].sum() > 0 else 0
            current_default_rate = (df_current['defaults'].sum() / df_current['rentals'].sum() * 100) if df_current['rentals'].sum() > 0 else 0
            
            # Update metric displays
            metric_displays[0].metric(
                "Annualized Return", 
                f"{current_return:.1f}%", 
                f"{current_return - 25:.1f}%",
                delta_color="normal" if current_return > 25 else "inverse"
            )
            metric_displays[1].metric(
                "Utilization Rate",
                f"{metrics['utilization_rate']:.0f}%",
                f"{metrics['utilization_rate'] - 1000:.0f}%",
                delta_color="normal" if metrics['utilization_rate'] > 800 else "inverse"
            )
            metric_displays[2].metric(
                "Loss Ratio", 
                f"{current_loss_ratio:.1f}%",
                f"{current_loss_ratio - 60:.1f}%",
                delta_color="inverse"
            )
            metric_displays[3].metric(
                "Total Rentals", 
                f"{df_current['rentals'].sum():,}",
                f"+{df_current['rentals'].iloc[-1] if len(df_current) > 0 else 0}"
            )
            metric_displays[4].metric(
                "Default Rate", 
                f"{current_default_rate:.2f}%",
                delta_color="inverse"
            )
            metric_displays[5].metric(
                "Active Users", 
                f"{metrics['active_users']:,}",
                f"+{metrics['new_users'] - metrics['banned_users']}"
            )
            
            # Update timestamp
            elapsed = (datetime.now() - start_time).total_seconds()
            last_update_text.caption(
                f"📍 Last updated: Day {day + 1} | ⏱️ Elapsed: {elapsed:.1f}s | "
                f"🚀 Speed: {(day + 1) / elapsed:.0f} days/sec"
            )
            
            # Update charts
            # 1. Pool & Returns
            with chart_placeholders[0].container():
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('LP Pool Evolution', 'Annualized Returns'),
                    row_heights=[0.5, 0.5]
                )
                
                # Pool evolution
                fig.add_trace(
                    go.Scatter(x=df_current['day'], y=df_current['lp_pool'],
                             mode='lines', name='LP Pool',
                             line=dict(color='rgb(75, 192, 192)', width=2)),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df_current['day'], y=df_current['insurance_reserve'],
                             mode='lines', name='Insurance',
                             line=dict(color='rgb(255, 159, 64)', width=2)),
                    row=1, col=1
                )
                
                # Returns
                df_current['annualized_return'] = ((df_current['lp_pool'] / current_params['initial_pool'] - 1) * 365 / (df_current['day'] + 1)) * 100
                fig.add_trace(
                    go.Scatter(x=df_current['day'], y=df_current['annualized_return'],
                             mode='lines', name='Return %',
                             line=dict(color='rgb(54, 162, 235)', width=3)),
                    row=2, col=1
                )
                fig.add_hline(y=25, line_dash="dash", line_color="green",
                            annotation_text="Target", row=2, col=1)
                
                fig.update_layout(height=600, showlegend=True, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            
            # 2. Utilization
            with chart_placeholders[1].container():
                fig_util = go.Figure()
                fig_util.add_trace(go.Scatter(
                    x=df_current['day'], 
                    y=df_current['utilization_rate'],
                    mode='lines',
                    name='Utilization Rate (%)',
                    line=dict(color='rgb(255, 99, 132)', width=3)
                ))
                
                # Add target zones
                fig_util.add_hrect(y0=800, y1=1500, fillcolor="green", opacity=0.2, 
                                  annotation_text="Optimal Zone (8-15x)")
                fig_util.add_hline(y=1000, line_dash="dash", line_color="green",
                                  annotation_text="Target 10x")
                
                fig_util.update_layout(
                    title='Capital Utilization Rate',
                    xaxis_title='Day',
                    yaxis_title='Utilization Rate (%)',
                    height=500,
                    template='plotly_white'
                )
                st.plotly_chart(fig_util, use_container_width=True)
            
            # 3. Users
            with chart_placeholders[2].container():
                df_current['cumulative_bans'] = df_current['banned_users'].cumsum()
                
                fig_users = go.Figure()
                fig_users.add_trace(go.Scatter(
                    x=df_current['day'], y=df_current['active_users'],
                    mode='lines',
                    name='Active Users',
                    line=dict(color='rgb(75, 192, 192)', width=2)
                ))
                fig_users.add_trace(go.Scatter(
                    x=df_current['day'], y=df_current['cumulative_bans'],
                    mode='lines',
                    name='Banned Users',
                    line=dict(color='rgb(255, 99, 132)', width=2, dash='dash')
                ))
                fig_users.update_layout(
                    title='User Base Evolution',
                    xaxis_title='Day',
                    yaxis_title='User Count',
                    height=500,
                    hovermode='x unified',
                    template='plotly_white'
                )
                st.plotly_chart(fig_users, use_container_width=True)
            
            # 4. Risk
            with chart_placeholders[3].container():
                df_current['loss_ratio'] = (df_current['losses'] / (df_current['premiums'] + 1)) * 100
                df_current['loss_ratio_ma30'] = df_current['loss_ratio'].rolling(window=30, min_periods=1).mean()
                
                fig_risk = go.Figure()
                fig_risk.add_trace(go.Scatter(
                    x=df_current['day'], y=df_current['loss_ratio_ma30'],
                    mode='lines',
                    name='Loss Ratio (30-day MA)',
                    line=dict(color='rgb(153, 102, 255)', width=2)
                ))
                fig_risk.add_hline(y=60, line_dash="dash", line_color="green",
                                 annotation_text="Target 60%")
                fig_risk.add_hline(y=100, line_dash="dash", line_color="red",
                                 annotation_text="Break-even")
                fig_risk.update_layout(
                    title='Loss Ratio Evolution',
                    xaxis_title='Day',
                    yaxis_title='Loss Ratio (%)',
                    height=500,
                    hovermode='x unified',
                    template='plotly_white'
                )
                st.plotly_chart(fig_risk, use_container_width=True)
    
    # Final processing
    df = pd.DataFrame(daily_metrics)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.success("✅ Simulation completed!")
    current_day_display.empty()
    last_update_text.empty()
    
    # Process final results
    results = {
        'df': df,
        'sim': sim,
        'annualized_return': (df.iloc[-1]['lp_pool'] / current_params['initial_pool'] - 1) * 100 * (365 / len(df)),
        'loss_ratio': (df['losses'].sum() / df['premiums'].sum() * 100) if df['premiums'].sum() > 0 else 0,
        'total_rentals': df['rentals'].sum(),
        'total_defaults': df['defaults'].sum(),
        'default_rate': (df['defaults'].sum() / df['rentals'].sum() * 100) if df['rentals'].sum() > 0 else 0,
        'final_users': df.iloc[-1]['active_users'],
        'initial_users': 0,
        'total_premiums': df['premiums'].sum(),
        'total_losses': df['losses'].sum(),
        'net_profit': df['premiums'].sum() - df['losses'].sum(),
        'final_lp_pool': df.iloc[-1]['lp_pool'],
        'final_insurance': df.iloc[-1]['insurance_reserve'],
        'users_banned': df['banned_users'].sum(),
        'avg_daily_rentals': df['rentals'].mean(),
        'avg_coverage': np.mean([r['coverage'] for r in sim.all_rentals]) if sim.all_rentals else 500,
        'reserve_health': df.iloc[-1]['pool_health'],
        'max_drawdown': ((df['lp_pool'].min() - current_params['initial_pool']) / current_params['initial_pool'] * 100),
        'avg_utilization': df['utilization_rate'].mean(),
        'peak_utilization': df['utilization_rate'].max(),
        'initial_pool': current_params['initial_pool']
    }
    
    # Calculate user distribution
    user_types = defaultdict(int)
    for user in sim.all_users.values():
        if user.id not in sim.banned_users:
            user_types[user.true_type] += 1
    results['user_distribution'] = dict(user_types)
    
    st.session_state['results'] = results
    st.session_state['sim_time'] = datetime.now()
    
    # Show completion summary
    total_elapsed = (datetime.now() - start_time).total_seconds()
    st.balloons()
    st.success(
        f"🎉 Simulation complete! Final annualized return: {results['annualized_return']:.1f}% | "
        f"Avg utilization: {results['avg_utilization']:.0f}% | "
        f"Total time: {total_elapsed:.1f}s ({current_params['simulation_days'] / total_elapsed:.0f} days/sec)"
    )

# Display results if available
if 'results' in st.session_state:
    results = st.session_state['results']
    params = st.session_state['sim_params']
    df = results['df']
    sim = results['sim']
    
    # Key Metrics
    st.header("📊 Key Performance Metrics")
    
    # Add scenario name if applicable
    if 'scenario_name' in st.session_state and st.session_state['scenario_name'] != "Custom Settings":
        st.info(f"📋 Scenario: {st.session_state['scenario_name']}")
    
    # Add utilization-based insights
    if results['avg_utilization'] < 500:
        st.warning(f"⚠️ **Low Capital Utilization**: Average {results['avg_utilization']:.0f}% is significantly below the 800-1500% target. This is the primary reason for lower returns.")
        st.info("💡 **To improve returns**: Increase coverage limits, reduce insurance premiums, or improve user acquisition/retention.")
    elif results['annualized_return'] < 0:
        st.error("❌ **System is unprofitable!** Despite utilization, high default rates are causing losses.")
    elif results['annualized_return'] < 15:
        st.warning("📊 **Below target returns.** Consider optimizing premium pricing or risk assessment.")
    else:
        st.success(f"✅ **Healthy returns of {results['annualized_return']:.1f}%** with {results['avg_utilization']:.0f}% utilization!")
    
    # Metrics display
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric(
            "Annualized Return",
            f"{results['annualized_return']:.1f}%",
            f"{results['annualized_return'] - 25:.1f}%"
        )
    
    with col2:
        st.metric(
            "Avg Utilization",
            f"{results['avg_utilization']:.0f}%",
            f"{results['avg_utilization'] - 1000:.0f}%",
            delta_color="normal" if results['avg_utilization'] > 800 else "inverse"
        )
    
    with col3:
        st.metric(
            "Loss Ratio",
            f"{results['loss_ratio']:.1f}%",
            f"{results['loss_ratio'] - 60:.1f}%",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            "Total Rentals",
            f"{results['total_rentals']:,}",
            f"{results['total_rentals'] / params['days']:.0f}/day"
        )
    
    with col5:
        st.metric(
            "Default Rate",
            f"{results['default_rate']:.2f}%",
            delta_color="inverse"
        )
    
    with col6:
        st.metric(
            "Active Users",
            f"{results['final_users']:,}",
            f"+{results['final_users'] - results['initial_users']}"
        )
    
    # Detailed Analytics Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Performance", "🎯 Risk Analysis", "👥 User Dynamics", 
        "🌊 Market Impact", "🔍 Advanced Analytics", "📊 Utilization Deep Dive",
        "📥 Export Data"
    ])
    
    with tab1:
        st.subheader("Pool Evolution & Returns")
        col1, col2 = st.columns(2)
        
        with col1:
            # Pool evolution chart
            fig_pool = go.Figure()
            fig_pool.add_trace(go.Scatter(
                x=df['day'], y=df['lp_pool'],
                mode='lines',
                name='LP Pool',
                fill='tonexty',
                line=dict(color='rgb(75, 192, 192)', width=2)
            ))
            fig_pool.add_trace(go.Scatter(
                x=df['day'], y=df['insurance_reserve'],
                mode='lines',
                name='Insurance Reserve',
                fill='tozeroy',
                line=dict(color='rgb(255, 159, 64)', width=2)
            ))
            fig_pool.update_layout(
                title='Pool Evolution',
                xaxis_title='Day',
                yaxis_title='Value ($)',
                height=400,
                hovermode='x unified',
                template='plotly_white'
            )
            st.plotly_chart(fig_pool, use_container_width=True)
        
        with col2:
            # Returns chart
            df['annualized_return'] = ((df['lp_pool'] / params['initial_pool'] - 1) * 365 / (df['day'] + 1)) * 100
            
            fig_returns = go.Figure()
            fig_returns.add_trace(go.Scatter(
                x=df['day'], y=df['annualized_return'],
                mode='lines',
                name='Annualized Return (%)',
                line=dict(color='rgb(54, 162, 235)', width=3)
            ))
            fig_returns.add_hline(y=0, line_dash="dash", line_color="black")
            fig_returns.add_hline(y=25, line_dash="dash", line_color="green",
                                annotation_text="Target 25%")
            fig_returns.update_layout(
                title='LP Returns (Annualized)',
                xaxis_title='Day',
                yaxis_title='Return (%)',
                height=400,
                hovermode='x unified',
                template='plotly_white'
            )
            st.plotly_chart(fig_returns, use_container_width=True)
    
    with tab2:
        st.subheader("Risk Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            # Loss ratio chart
            df['loss_ratio'] = (df['losses'] / (df['premiums'] + 1)) * 100
            df['loss_ratio_ma30'] = df['loss_ratio'].rolling(window=30, min_periods=1).mean()
            
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=df['day'], y=df['loss_ratio_ma30'],
                mode='lines',
                name='Loss Ratio (30-day MA)',
                line=dict(color='rgb(153, 102, 255)', width=2)
            ))
            fig_loss.add_hline(y=60, line_dash="dash", line_color="green",
                             annotation_text="Target 60%")
            fig_loss.add_hline(y=100, line_dash="dash", line_color="red",
                             annotation_text="Break-even")
            fig_loss.update_layout(
                title='Loss Ratio Evolution',
                xaxis_title='Day',
                yaxis_title='Loss Ratio (%)',
                height=400,
                hovermode='x unified',
                template='plotly_white'
            )
            st.plotly_chart(fig_loss, use_container_width=True)
        
        with col2:
            # Defaults by market condition
            market_defaults = df.groupby('market_condition')['defaults'].sum().to_dict()
            
            fig_defaults = go.Figure()
            fig_defaults.add_trace(go.Bar(
                x=list(market_defaults.keys()),
                y=list(market_defaults.values()),
                marker_color=['rgba(75, 192, 192, 0.8)', 'rgba(201, 203, 207, 0.8)',
                             'rgba(255, 159, 64, 0.8)', 'rgba(255, 99, 132, 0.8)'][:len(market_defaults)],
                name='Defaults'
            ))
            fig_defaults.update_layout(
                title='Defaults by Market Condition',
                xaxis_title='Market Condition',
                yaxis_title='Total Defaults',
                height=400,
                template='plotly_white',
                showlegend=False
            )
            st.plotly_chart(fig_defaults, use_container_width=True)
    
    with tab3:
        st.subheader("User Dynamics")
        col1, col2 = st.columns(2)
        
        with col1:
            # User growth chart
            df['cumulative_bans'] = df['banned_users'].cumsum()
            
            fig_users = go.Figure()
            fig_users.add_trace(go.Scatter(
                x=df['day'], y=df['active_users'],
                mode='lines',
                name='Active Users',
                line=dict(color='rgb(75, 192, 192)', width=2)
            ))
            fig_users.add_trace(go.Scatter(
                x=df['day'], y=df['cumulative_bans'],
                mode='lines',
                name='Banned Users',
                line=dict(color='rgb(255, 99, 132)', width=2, dash='dash')
            ))
            fig_users.update_layout(
                title='User Base Evolution',
                xaxis_title='Day',
                yaxis_title='User Count',
                height=400,
                hovermode='x unified',
                template='plotly_white'
            )
            st.plotly_chart(fig_users, use_container_width=True)
        
        with col2:
            # User distribution pie
            user_dist = results['user_distribution']
            
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Pie(
                labels=['Good', 'Mostly Good', 'Opportunist', 'Professional Fraud'],
                values=[
                    user_dist.get('good', 0),
                    user_dist.get('mostly_good', 0),
                    user_dist.get('opportunist', 0),
                    user_dist.get('professional_fraud', 0)
                ],
                marker_colors=[
                    'rgba(75, 192, 192, 0.8)',
                    'rgba(255, 206, 86, 0.8)',
                    'rgba(255, 159, 64, 0.8)',
                    'rgba(255, 99, 132, 0.8)'
                ],
                hole=0.3
            ))
            fig_dist.update_layout(
                title='Final User Distribution',
                height=400,
                template='plotly_white'
            )
            st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab4:
        st.subheader("Market Condition Analysis")
        
        # Create market summary
        market_summary = []
        for condition in ['bull', 'normal', 'bear', 'crash']:
            mask = df['market_condition'] == condition
            if mask.any():
                condition_df = df[mask]
                market_summary.append({
                    'condition': condition.capitalize(),
                    'days': len(condition_df),
                    'avg_utilization': condition_df['utilization_rate'].mean(),
                    'rentals': condition_df['rentals'].sum(),
                    'defaults': condition_df['defaults'].sum(),
                    'net_pnl': condition_df['daily_pnl'].sum()
                })
        
        if market_summary:
            # Market metrics
            summary_df = pd.DataFrame(market_summary)
            
            # Utilization by market condition
            fig_market_util = go.Figure()
            fig_market_util.add_trace(go.Bar(
                x=summary_df['condition'],
                y=summary_df['avg_utilization'],
                marker_color=['rgba(75, 192, 192, 0.8)', 'rgba(201, 203, 207, 0.8)',
                             'rgba(255, 159, 64, 0.8)', 'rgba(255, 99, 132, 0.8)'][:len(summary_df)],
                name='Avg Utilization'
            ))
            fig_market_util.add_hline(y=1000, line_dash="dash", line_color="green",
                                    annotation_text="Target 10x")
            fig_market_util.update_layout(
                title='Average Utilization by Market Condition',
                xaxis_title='Market Condition',
                yaxis_title='Utilization Rate (%)',
                height=400,
                template='plotly_white',
                showlegend=False
            )
            st.plotly_chart(fig_market_util, use_container_width=True)
            
            # Summary table
            display_summary = summary_df.copy()
            display_summary['avg_utilization'] = display_summary['avg_utilization'].round(0).astype(str) + '%'
            display_summary['net_pnl'] = '$' + display_summary['net_pnl'].round(0).astype(int).astype(str)
            st.dataframe(display_summary, use_container_width=True, hide_index=True)
    
    with tab5:
        # Call the advanced analytics function
        add_advanced_analytics(st, df, sim, results)
    
    with tab6:
        st.subheader("📊 Utilization Deep Dive")
        
        # Utilization over time with phases
        fig_util_phases = go.Figure()
        
        # Add utilization line
        fig_util_phases.add_trace(go.Scatter(
            x=df['day'],
            y=df['utilization_rate'],
            mode='lines',
            name='Utilization Rate',
            line=dict(color='rgb(255, 99, 132)', width=3)
        ))
        
        # Add phase annotations
        phases = [
            (0, 90, "Launch Phase", "yellow"),
            (90, 180, "Growth Phase", "orange"),
            (180, len(df), "Mature Phase", "green")
        ]
        
        for start, end, label, color in phases:
            if end > len(df):
                end = len(df)
            fig_util_phases.add_vrect(
                x0=start, x1=end,
                fillcolor=color, opacity=0.1,
                annotation_text=label,
                annotation_position="top left"
            )
        
        # Add target zones
        fig_util_phases.add_hrect(y0=800, y1=1500, fillcolor="green", opacity=0.1)
        fig_util_phases.add_hline(y=1000, line_dash="dash", line_color="green",
                                annotation_text="Target 10x")
        
        fig_util_phases.update_layout(
            title='Utilization Rate Evolution by Phase',
            xaxis_title='Day',
            yaxis_title='Utilization Rate (%)',
            height=500,
            template='plotly_white'
        )
        st.plotly_chart(fig_util_phases, use_container_width=True)
        
        # Utilization efficiency metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Utilization Efficiency Metrics")
            
            # Calculate efficiency metrics
            days_in_target = ((df['utilization_rate'] >= 800) & (df['utilization_rate'] <= 1500)).sum()
            days_over_target = (df['utilization_rate'] > 1500).sum()
            days_under_target = (df['utilization_rate'] < 800).sum()
            
            efficiency_data = pd.DataFrame({
                'Metric': [
                    'Days in Target Zone (8-15x)',
                    'Days Over-utilized (>15x)',
                    'Days Under-utilized (<8x)',
                    'Utilization Efficiency Score'
                ],
                'Value': [
                    f"{days_in_target} ({days_in_target/len(df)*100:.1f}%)",
                    f"{days_over_target} ({days_over_target/len(df)*100:.1f}%)",
                    f"{days_under_target} ({days_under_target/len(df)*100:.1f}%)",
                    f"{days_in_target/len(df)*100:.1f}%"
                ]
            })
            
            st.dataframe(efficiency_data, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### Revenue Impact of Utilization")
            
            # Group by utilization buckets
            df['util_bucket'] = pd.cut(df['utilization_rate'], 
                                       bins=[0, 500, 800, 1000, 1500, 10000],
                                       labels=['<5x', '5-8x', '8-10x', '10-15x', '>15x'])
            
            util_impact = df.groupby('util_bucket').agg({
                'premiums': 'mean',
                'daily_pnl': 'mean',
                'rentals': 'mean'
            }).round(0)
            
            util_impact.columns = ['Avg Daily Premium', 'Avg Daily P&L', 'Avg Daily Rentals']
            st.dataframe(util_impact, use_container_width=True)
    
    # Add insights box
    with st.expander("🎯 Key Insights & Recommendations", expanded=True):
        insights = []
        
        # Utilization insights
        if results['avg_utilization'] < 500:
            insights.append("🔴 **Critical Issue**: Utilization averaging " + 
                          f"{results['avg_utilization']:.0f}% is far below the 800-1500% target. " +
                          "This is the primary bottleneck for returns.")
            insights.append("💡 **Solutions**: Increase coverage limits, reduce insurance premiums, " +
                          "or implement promotional periods to boost rental volume.")
        elif results['avg_utilization'] < 800:
            insights.append("🟡 **Improvement Needed**: Utilization of " + 
                          f"{results['avg_utilization']:.0f}% is below optimal. " +
                          "There's significant room for improvement.")
        else:
            insights.append("🟢 **Good Utilization**: Average " + 
                          f"{results['avg_utilization']:.0f}% utilization is healthy!")
        
        # Return insights
        if results['annualized_return'] < 15:
            insights.append("📊 **Returns Below Target**: Consider adjusting premium margins " +
                          "or improving risk assessment to reduce defaults.")
        
        # Default rate insights
        if results['default_rate'] > 5:
            insights.append("⚠️ **High Default Rate**: " + 
                          f"{results['default_rate']:.1f}% defaults are eroding profits. " +
                          "Consider stricter KYC or lower initial coverage limits.")
        
        # User base insights
        fraud_ratio = results['user_distribution'].get('professional_fraud', 0) / results['final_users']
        if fraud_ratio > 0.02:
            insights.append("🚨 **Fraud Concentration**: Professional fraudsters make up " +
                          f"{fraud_ratio*100:.1f}% of active users. The system needs time to ban them.")
        
        for insight in insights:
            st.markdown(insight)
    
    # Export section
    st.header("💾 Export Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="flow_simulation_optimized.csv">📥 Download Daily Data (CSV)</a>', unsafe_allow_html=True)
    
    with col2:
        params_json = json.dumps(params, indent=2)
        b64 = base64.b64encode(params_json.encode()).decode()
        st.markdown(f'<a href="data:file/json;base64,{b64}" download="flow_parameters.json">📥 Download Parameters (JSON)</a>', unsafe_allow_html=True)
    
    with col3:
        summary = {
            'parameters': params,
            'results': {k: v for k, v in results.items() if k not in ['df', 'sim']},
            'timestamp': st.session_state['sim_time'].isoformat()
        }
        summary_json = json.dumps(summary, indent=2, default=str)
        b64 = base64.b64encode(summary_json.encode()).decode()
        st.markdown(f'<a href="data:file/json;base64,{b64}" download="flow_summary.json">📥 Download Summary (JSON)</a>', unsafe_allow_html=True)
    
    # Add insights box
    with st.expander("🎯 Key Insights & Recommendations", expanded=True):
        insights = []
        
        # Utilization insights
        if results['avg_utilization'] < 500:
            insights.append("🔴 **Critical Issue**: Utilization averaging " + 
                          f"{results['avg_utilization']:.0f}% is far below the 800-1500% target. " +
                          "This is the primary bottleneck for returns.")
            insights.append("💡 **Solutions**: Increase coverage limits, reduce insurance premiums, " +
                          "or implement promotional periods to boost rental volume.")
        elif results['avg_utilization'] < 800:
            insights.append("🟡 **Improvement Needed**: Utilization of " + 
                          f"{results['avg_utilization']:.0f}% is below optimal. " +
                          "There's significant room for improvement.")
        else:
            insights.append("🟢 **Good Utilization**: Average " + 
                          f"{results['avg_utilization']:.0f}% utilization is healthy!")
        
        # Return insights
        if results['annualized_return'] < 15:
            insights.append("📊 **Returns Below Target**: Consider adjusting premium margins " +
                          "or improving risk assessment to reduce defaults.")
        
        # Default rate insights
        if results['default_rate'] > 5:
            insights.append("⚠️ **High Default Rate**: " + 
                          f"{results['default_rate']:.1f}% defaults are eroding profits. " +
                          "Consider stricter KYC or lower initial coverage limits.")
        
        # User base insights
        fraud_ratio = results['user_distribution'].get('professional_fraud', 0) / results['final_users']
        if fraud_ratio > 0.02:
            insights.append("🚨 **Fraud Concentration**: Professional fraudsters make up " +
                          f"{fraud_ratio*100:.1f}% of active users. The system needs time to ban them.")
        
        for insight in insights:
            st.markdown(insight)

# Add scenario comparison in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔬 Tools")
if st.sidebar.button("📊 Compare Scenarios"):
    st.session_state['show_comparison'] = True

# Scenario Comparison Tool
if 'show_comparison' in st.session_state and st.session_state['show_comparison']:
    st.header("📊 Scenario Comparison Tool")
    
    # Select scenarios to compare
    selected_scenarios = st.multiselect(
        "Select scenarios to compare",
        [s for s in STRESS_TEST_SCENARIOS.keys() if s != "Custom Settings"],
        default=["🛡️ Conservative Starter ($100K)", "⚡ Aggressive Growth ($25K)"]
    )
    
    if len(selected_scenarios) >= 2 and st.button("Run Comparison"):
        results_comparison = {}
        progress = st.progress(0)
        
        for i, scenario_name in enumerate(selected_scenarios):
            progress.progress((i + 1) / len(selected_scenarios))
            st.info(f"Running {scenario_name}...")
            
            scenario = STRESS_TEST_SCENARIOS[scenario_name]
            
            # Create parameters
            sim_params = {
                'market': {
                    'bull': scenario['bull_prob'] / 100,
                    'bear': scenario['bear_prob'] / 100,
                    'crash': scenario['crash_prob'] / 100,
                    'normal': (100 - scenario['bull_prob'] - scenario['bear_prob'] - scenario['crash_prob']) / 100
                },
                'users': {
                    'good': scenario['good_users'] / 100,
                    'mostly_good': scenario['mostly_good'] / 100,
                    'opportunist': scenario['opportunist'] / 100,
                    'professional_fraud': scenario['fraud'] / 100
                },
                'initial_pool': scenario['initial_pool'],
                'kyc_cost': scenario.get('kyc_cost', 750),
                'insurance_target': scenario['insurance_target'] / 100,
                'max_coverage_ratio': scenario.get('max_coverage_ratio', 0.95),
                'network_effect': scenario.get('network_effect', True),
                'wipo_enforcement': scenario.get('wipo_enforcement', True),
                'days': scenario['days'],
                'seed': scenario.get('seed', 42)
            }
            
            # Run simulation
            sim = OptimizedFlowProtocolSimulation(
                initial_lp_pool=scenario['initial_pool'],
                params=sim_params
            )
            
            df = sim.run_simulation(days=scenario['days'])
            
            # Calculate key metrics
            results_comparison[scenario_name] = {
                'initial_pool': scenario['initial_pool'],
                'days': scenario['days'],
                'final_pool': df.iloc[-1]['lp_pool'],
                'annualized_return': (df.iloc[-1]['lp_pool'] / scenario['initial_pool'] - 1) * 100 * (365 / len(df)),
                'avg_utilization': df['utilization_rate'].mean(),
                'peak_utilization': df['utilization_rate'].max(),
                'total_rentals': df['rentals'].sum(),
                'default_rate': (df['defaults'].sum() / df['rentals'].sum() * 100) if df['rentals'].sum() > 0 else 0,
                'loss_ratio': (df['losses'].sum() / df['premiums'].sum() * 100) if df['premiums'].sum() > 0 else 0,
                'max_drawdown': ((df['lp_pool'].min() - scenario['initial_pool']) / scenario['initial_pool'] * 100)
            }
        
        progress.empty()
        
        # Display comparison table
        comparison_df = pd.DataFrame(results_comparison).T
        comparison_df = comparison_df.round(2)
        
        # Style the dataframe
        styled_df = comparison_df.style.format({
            'initial_pool': '${:,.0f}',
            'final_pool': '${:,.0f}',
            'annualized_return': '{:.1f}%',
            'avg_utilization': '{:.0f}%',
            'peak_utilization': '{:.0f}%',
            'default_rate': '{:.2f}%',
            'loss_ratio': '{:.1f}%',
            'max_drawdown': '{:.1f}%'
        })
        
        st.dataframe(styled_df, use_container_width=True)

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Complete Features")
st.sidebar.success("""
**This complete version includes:**
- ✅ All scenario parameter overrides
- ✅ Enhanced UI with categories
- ✅ Parameter validation
- ✅ WIPO enforcement modeling
- ✅ Network effect analysis
- ✅ Pool-size-aware user creation
- ✅ Real-time chart updates
- ✅ Scenario comparison tool
- ✅ Advanced analytics
- ✅ Export functionality
""")

st.sidebar.markdown("### 💡 Tips for High Returns")
st.sidebar.info("""
**Target Metrics:**
- Utilization: 800-1500% (8-15x)
- Loss Ratio: <60%
- Default Rate: <5%
- Good Users: >70%

**If returns are low:**
1. Check utilization rate first!
2. Increase coverage limits
3. Reduce insurance premiums
4. Improve user retention
5. Run longer simulations
""")

st.sidebar.markdown("### 📖 About")
st.sidebar.markdown("""
This complete Flow Protocol simulator includes:
- **All features** integrated
- **Scenario overrides** working
- **Enhanced analytics**
- **Parameter validation**

The key to profitability is achieving 
high capital utilization (10x+) while 
maintaining acceptable default rates.
""")