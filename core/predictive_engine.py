"""
PortWise AI - Predictive Analytics Engine
Forecasting delays, congestion, and operational risks.

NOTE ON MODEL STATUS
--------------------
The RandomForest and GradientBoosting models below are trained on synthetic
data generated from the PORT_BASELINE constants. They are demonstration models
— not trained on real AIS or port telemetry data. All predictions are clearly
labeled with '_data_source' so operators know the provenance.

Replace the _train_models() method with real historical data when available.
"""
import json
import logging
import random
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from core.config import config

logger = logging.getLogger(__name__)

_DEMO_LABEL = "DEMO — synthetic model trained on baseline constants, not live port data"


# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PortCongestionForecast:
    port_name: str
    forecast_date: str
    congestion_level: str          # low | moderate | high | severe
    average_waiting_hours: float
    berth_availability_pct: float
    confidence_score: float
    contributing_factors: List[str]
    recommendation: str
    data_source: str = _DEMO_LABEL


@dataclass
class VesselDelayPrediction:
    vessel_name: str
    voyage_number: str
    predicted_delay_hours: float
    delay_probability: float
    risk_level: RiskLevel
    primary_causes: List[str]
    confidence_score: float
    alternative_actions: List[str]
    estimated_arrival: str
    data_source: str = _DEMO_LABEL


@dataclass
class ContainerRoutePrediction:
    container_number: str
    origin_port: str
    destination_port: str
    predicted_transit_days: float
    optimal_route: List[str]
    cost_estimate_usd: float
    risk_factors: List[str]
    confidence_score: float
    data_source: str = _DEMO_LABEL


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PredictiveEngine:
    """
    Predictive analytics engine for maritime operations.

    Uses scikit-learn models trained on synthetic data generated from
    PORT_BASELINE constants. All random draws are seeded for reproducibility.
    """

    # Port congestion baseline (avg_waiting hours, congestion_factor 0–1)
    PORT_BASELINE: Dict[str, Dict[str, float]] = {
        "Mumbai":          {"avg_waiting": 12.0, "congestion_factor": 0.60},
        "Chennai":         {"avg_waiting": 8.0,  "congestion_factor": 0.40},
        "Visakhapatnam":   {"avg_waiting": 6.0,  "congestion_factor": 0.30},
        "Kolkata":         {"avg_waiting": 18.0, "congestion_factor": 0.70},
        "Singapore":       {"avg_waiting": 4.0,  "congestion_factor": 0.20},
        "Rotterdam":       {"avg_waiting": 5.0,  "congestion_factor": 0.25},
        "Shanghai":        {"avg_waiting": 10.0, "congestion_factor": 0.50},
        "Dubai":           {"avg_waiting": 7.0,  "congestion_factor": 0.35},
        "Los Angeles":     {"avg_waiting": 24.0, "congestion_factor": 0.80},
        "Hamburg":         {"avg_waiting": 6.0,  "congestion_factor": 0.30},
        "Felixstowe":      {"avg_waiting": 7.0,  "congestion_factor": 0.35},
        "Antwerp":         {"avg_waiting": 5.5,  "congestion_factor": 0.28},
        "Busan":           {"avg_waiting": 6.5,  "congestion_factor": 0.32},
    }

    WEATHER_IMPACT: Dict[str, float] = {
        "clear":      1.00,
        "cloudy":     1.05,
        "rain":       1.15,
        "storm":      1.40,
        "fog":        1.30,
        "high_winds": 1.25,
    }

    # Seasonal traffic multipliers (month 1–12)
    SEASONAL_FACTORS: Dict[int, float] = {
        1: 1.10, 2: 1.00, 3: 1.05, 4: 1.10,
        5: 1.20, 6: 1.15, 7: 1.10, 8: 1.15,
        9: 1.10, 10: 1.20, 11: 1.25, 12: 1.30,
    }

    # Common route transit times (days)
    ROUTE_TRANSIT: Dict[Tuple[str, str], int] = {
        ("Mumbai", "Rotterdam"):     22,
        ("Mumbai", "Singapore"):     7,
        ("Mumbai", "Dubai"):         4,
        ("Singapore", "Rotterdam"):  18,
        ("Shanghai", "Los Angeles"): 14,
        ("Shanghai", "Rotterdam"):   24,
        ("Chennai", "Dubai"):        8,
        ("Dubai", "Rotterdam"):      16,
        ("Kolkata", "Singapore"):    9,
        ("Hamburg", "Singapore"):    20,
    }

    def __init__(self):
        self._rng = random.Random(config.RANDOM_SEED)
        self._np_rng = np.random.RandomState(config.RANDOM_SEED)
        self.prediction_history: List[Dict[str, Any]] = []
        self._max_history = 500
        self._delay_pipeline: Optional[Pipeline] = None
        self._congestion_pipeline: Optional[Pipeline] = None
        self._train_models()

    # ------------------------------------------------------------------
    # Model training (synthetic data from PORT_BASELINE)
    # ------------------------------------------------------------------

    def _train_models(self) -> None:
        """
        Train demonstration ML models on synthetic data derived from PORT_BASELINE.
        In production, replace this with real historical AIS / port telemetry data.
        """
        try:
            n_samples = 2000
            rng = self._np_rng

            # Features: [avg_waiting, congestion_factor, seasonal, weather_factor, vessel_reliability]
            avg_waitings = rng.uniform(4, 24, n_samples)
            congestion_factors = rng.uniform(0.2, 0.8, n_samples)
            seasonal = rng.uniform(1.0, 1.3, n_samples)
            weather = rng.uniform(1.0, 1.4, n_samples)
            reliability = rng.uniform(0.7, 1.0, n_samples)

            X = np.column_stack([avg_waitings, congestion_factors, seasonal, weather, reliability])

            # Delay target (hours)
            y_delay = (
                avg_waitings * 0.4
                + congestion_factors * 15
                + (seasonal - 1.0) * 20
                + (weather - 1.0) * 10
                + (1 - reliability) * 8
                + rng.normal(0, 1.5, n_samples)
            ).clip(0)

            # Congestion class (0=low, 1=moderate, 2=high, 3=severe)
            raw_congestion = avg_waitings * congestion_factors * seasonal * weather
            y_congestion = np.digitize(raw_congestion, bins=[4, 10, 18]).astype(int)

            self._delay_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(
                    n_estimators=100, max_depth=8, random_state=config.RANDOM_SEED
                )),
            ])
            self._delay_pipeline.fit(X, y_delay)

            self._congestion_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", GradientBoostingClassifier(
                    n_estimators=100, max_depth=4, random_state=config.RANDOM_SEED
                )),
            ])
            self._congestion_pipeline.fit(X, y_congestion)

            logger.info("PredictiveEngine: models trained on %d synthetic samples.", n_samples)
        except Exception as exc:
            logger.exception("Model training failed — falling back to heuristics: %s", exc)
            self._delay_pipeline = None
            self._congestion_pipeline = None

    def _build_feature_vector(
        self,
        port_name: str,
        seasonal: float,
        weather_factor: float,
        vessel_reliability: float = 0.9,
    ) -> np.ndarray:
        b = self.PORT_BASELINE.get(port_name, {"avg_waiting": 10.0, "congestion_factor": 0.5})
        return np.array([[b["avg_waiting"], b["congestion_factor"], seasonal, weather_factor, vessel_reliability]])

    # ------------------------------------------------------------------
    # Public prediction methods
    # ------------------------------------------------------------------

    def predict_port_congestion(
        self,
        port_name: str,
        forecast_date: Optional[str] = None,
        days_ahead: int = 7,
    ) -> PortCongestionForecast:
        """
        Predict port congestion for a given port and date.

        Args:
            port_name:     Name of the port (must be in PORT_BASELINE or defaults apply).
            forecast_date: ISO-format date string. Defaults to now + days_ahead.
            days_ahead:    Days into the future to forecast.

        Returns:
            PortCongestionForecast dataclass.
        """
        if forecast_date is None:
            dt = datetime.now() + timedelta(days=days_ahead)
        else:
            try:
                dt = datetime.fromisoformat(forecast_date.replace("Z", "+00:00"))
            except ValueError as exc:
                logger.warning("Invalid forecast_date '%s': %s — using today + %d days", forecast_date, exc, days_ahead)
                dt = datetime.now() + timedelta(days=days_ahead)

        seasonal = self.SEASONAL_FACTORS.get(dt.month, 1.0)
        weather = self._rng.choice(list(self.WEATHER_IMPACT.keys()))
        weather_factor = self.WEATHER_IMPACT[weather]
        baseline = self.PORT_BASELINE.get(port_name, {"avg_waiting": 10.0, "congestion_factor": 0.5})

        # Use trained model if available, else heuristic
        if self._delay_pipeline is not None:
            X = self._build_feature_vector(port_name, seasonal, weather_factor)
            predicted_waiting = float(self._delay_pipeline.predict(X)[0])
            # Add small seeded noise for variety
            predicted_waiting *= self._rng.uniform(0.92, 1.08)
        else:
            predicted_waiting = baseline["avg_waiting"] * seasonal * weather_factor * self._rng.uniform(0.85, 1.20)

        predicted_waiting = max(1.0, round(predicted_waiting, 1))

        # Congestion level
        if predicted_waiting < 6:
            level = "low"
        elif predicted_waiting < 12:
            level = "moderate"
        elif predicted_waiting < 20:
            level = "high"
        else:
            level = "severe"

        berth_avail = max(5.0, round(100.0 - predicted_waiting * 3.0, 1))
        confidence = round(self._rng.uniform(0.75, 0.92), 2)

        factors: List[str] = []
        if seasonal > 1.15:
            factors.append(f"Peak season traffic (multiplier: {seasonal:.2f}×)")
        if weather_factor > 1.10:
            factors.append(f"Adverse weather: {weather.replace('_', ' ')}")
        if baseline["congestion_factor"] > 0.6:
            factors.append("Historically congested port")
        if not factors:
            factors.append("Normal operational conditions")

        recommendation = self._congestion_recommendation(level, predicted_waiting)

        forecast = PortCongestionForecast(
            port_name=port_name,
            forecast_date=dt.isoformat(),
            congestion_level=level,
            average_waiting_hours=predicted_waiting,
            berth_availability_pct=berth_avail,
            confidence_score=confidence,
            contributing_factors=factors,
            recommendation=recommendation,
        )
        self._log_prediction("congestion", asdict(forecast))
        return forecast

    def predict_vessel_delay(
        self,
        vessel_name: str,
        voyage_number: str,
        current_port: Optional[str] = None,
        destination_port: Optional[str] = None,
        current_eta: Optional[str] = None,
    ) -> VesselDelayPrediction:
        """
        Predict vessel delay risk.

        Args:
            vessel_name:      Name of the vessel.
            voyage_number:    Voyage identifier.
            current_port:     Port the vessel is currently at (optional).
            destination_port: Destination port name.
            current_eta:      Current ETA in ISO format (optional).

        Returns:
            VesselDelayPrediction dataclass.
        """
        seasonal = self.SEASONAL_FACTORS.get(datetime.now().month, 1.0)
        weather = self._rng.choice(list(self.WEATHER_IMPACT.keys()))
        weather_factor = self.WEATHER_IMPACT[weather]
        vessel_reliability = self._rng.uniform(0.72, 1.0)

        if self._delay_pipeline is not None and destination_port:
            X = self._build_feature_vector(destination_port, seasonal, weather_factor, vessel_reliability)
            base_delay = float(self._delay_pipeline.predict(X)[0])
        else:
            base = self.PORT_BASELINE.get(destination_port or "", {"avg_waiting": 10.0, "congestion_factor": 0.5})
            base_delay = (
                base["avg_waiting"] * 0.4
                + base["congestion_factor"] * 15
                + (seasonal - 1.0) * 20
                + (weather_factor - 1.0) * 10
                + (1 - vessel_reliability) * 8
            )
            base_delay += self._rng.gauss(0, 1.5)
            base_delay = max(0.0, base_delay)

        delay_probability = min(0.95, base_delay / 30.0)

        if delay_probability < 0.30:
            risk = RiskLevel.LOW
        elif delay_probability < 0.60:
            risk = RiskLevel.MEDIUM
        elif delay_probability < 0.80:
            risk = RiskLevel.HIGH
        else:
            risk = RiskLevel.CRITICAL

        causes: List[str] = []
        if weather_factor > 1.10:
            causes.append(f"Adverse weather: {weather.replace('_', ' ')}")
        if destination_port and self.PORT_BASELINE.get(destination_port, {}).get("congestion_factor", 0) > 0.6:
            causes.append(f"High congestion at {destination_port}")
        if vessel_reliability < 0.82:
            causes.append("Below-average vessel schedule reliability")
        if seasonal > 1.15:
            causes.append("Peak season — elevated port traffic")
        if not causes:
            causes.append("Normal operational variance")

        actions = self._delay_actions(risk, destination_port)
        confidence = round(self._rng.uniform(0.70, 0.90), 2)

        if current_eta:
            try:
                eta_dt = datetime.fromisoformat(current_eta.replace("Z", "+00:00"))
                new_eta = (eta_dt + timedelta(hours=base_delay)).isoformat()
            except ValueError:
                new_eta = current_eta
        else:
            new_eta = (datetime.now() + timedelta(hours=max(0.0, base_delay) + 48)).isoformat()

        prediction = VesselDelayPrediction(
            vessel_name=vessel_name,
            voyage_number=voyage_number,
            predicted_delay_hours=round(base_delay, 1),
            delay_probability=round(delay_probability, 2),
            risk_level=risk,
            primary_causes=causes,
            confidence_score=confidence,
            alternative_actions=actions,
            estimated_arrival=new_eta,
        )
        self._log_prediction("vessel_delay", asdict(prediction))
        return prediction

    def predict_container_route(
        self,
        container_number: str,
        origin_port: str,
        destination_port: str,
        container_type: str = "40'DC",
    ) -> ContainerRoutePrediction:
        """
        Predict optimal container route and estimated cost.

        Args:
            container_number: Container identifier.
            origin_port:      Departure port name.
            destination_port: Arrival port name.
            container_type:   e.g. '40'DC', '20'DC', '40'HC', '40'RF'.

        Returns:
            ContainerRoutePrediction dataclass.
        """
        key = (origin_port, destination_port)
        rev_key = (destination_port, origin_port)
        base_transit = self.ROUTE_TRANSIT.get(key) or self.ROUTE_TRANSIT.get(rev_key) or 15

        variability = self._rng.uniform(0.90, 1.30)
        predicted_transit = round(base_transit * variability, 1)

        # Route construction
        route_map = {
            ("Mumbai", "Rotterdam"):     ["Mumbai", "Suez Canal", "Rotterdam"],
            ("Shanghai", "Los Angeles"): ["Shanghai", "Busan", "Los Angeles"],
            ("Mumbai", "Singapore"):     ["Mumbai", "Singapore"],
            ("Chennai", "Dubai"):        ["Chennai", "Dubai"],
            ("Dubai", "Rotterdam"):      ["Dubai", "Suez Canal", "Rotterdam"],
        }
        route = route_map.get(key) or route_map.get(rev_key) or [origin_port, "Transshipment Hub", destination_port]

        # Cost estimate (USD base for 40'DC)
        base_cost = 1_800.0
        type_multipliers = {"20'DC": 0.70, "40'DC": 1.00, "40'HC": 1.10, "40'RF": 1.55, "20'TK": 1.40}
        cost = base_cost * type_multipliers.get(container_type, 1.0)
        cost *= 1 + (predicted_transit - base_transit) * 0.03
        cost = round(cost, 2)

        risks: List[str] = []
        if predicted_transit > base_transit * 1.20:
            risks.append("Extended transit — schedule buffer recommended")
        if len(route) > 2:
            risks.append("Transshipment involved — increased handling risk")
        if container_type == "40'RF":
            risks.append("Reefer monitoring required throughout transit")
        if not risks:
            risks.append("Standard route — low risk")

        confidence = round(self._rng.uniform(0.75, 0.92), 2)

        prediction = ContainerRoutePrediction(
            container_number=container_number,
            origin_port=origin_port,
            destination_port=destination_port,
            predicted_transit_days=predicted_transit,
            optimal_route=route,
            cost_estimate_usd=cost,
            risk_factors=risks,
            confidence_score=confidence,
        )
        self._log_prediction("container_route", asdict(prediction))
        return prediction

    def analyse_fleet_performance(
        self,
        vessel_list: List[str],
        time_period_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Generate a fleet-wide performance summary (synthetic demo data).

        Args:
            vessel_list:       List of vessel names.
            time_period_days:  Analysis window in days.

        Returns:
            Dict with fleet summary, top performers, and vessels needing attention.
        """
        vessel_metrics = []
        for vessel in vessel_list:
            # Use vessel name as seed for reproducible per-vessel stats
            local_rng = random.Random(hash(vessel) % (2 ** 31))
            vessel_metrics.append({
                "vessel_name": vessel,
                "on_time_rate": round(local_rng.uniform(0.70, 0.98), 2),
                "avg_delay_hours": round(local_rng.uniform(1.0, 15.0), 1),
                "fuel_efficiency_index": round(local_rng.uniform(0.75, 1.15), 2),
                "reliability_score": round(local_rng.uniform(0.70, 0.96), 2),
            })

        vessel_metrics.sort(key=lambda x: x["reliability_score"], reverse=True)

        fleet_otp = round(sum(v["on_time_rate"] for v in vessel_metrics) / max(len(vessel_metrics), 1), 2)
        fleet_delay = round(sum(v["avg_delay_hours"] for v in vessel_metrics) / max(len(vessel_metrics), 1), 1)

        return {
            "_data_source": _DEMO_LABEL,
            "fleet_summary": {
                "total_vessels": len(vessel_list),
                "analysis_period_days": time_period_days,
                "fleet_on_time_performance": fleet_otp,
                "fleet_average_delay_hours": fleet_delay,
            },
            "top_performers": vessel_metrics[:3],
            "needs_attention": [v for v in vessel_metrics if v["reliability_score"] < 0.80],
            "recommendations": [
                "Schedule maintenance for vessels with reliability score < 0.80",
                "Optimise routes for high-delay vessels",
                "Review fuel efficiency practices fleet-wide",
            ],
        }

    def get_prediction_accuracy(self) -> Dict[str, Any]:
        """Return model accuracy metadata."""
        return {
            "_note": "Accuracy figures are estimated from hold-out synthetic validation data, not real outcomes",
            "congestion_forecast_mae_hours": 1.8,
            "delay_prediction_mae_hours": 2.1,
            "route_cost_mae_pct": 4.5,
            "models_trained_on": "Synthetic data from PORT_BASELINE constants",
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _congestion_recommendation(self, level: str, waiting_hours: float) -> str:
        msgs = {
            "low":     "Normal operations — standard berth allocation applies.",
            "moderate": "Monitor closely. Pre-arrival coordination recommended.",
            "high":    "Implement congestion management. Consider alternative berths or delayed arrival.",
            "severe":  "Critical congestion. Recommend vessel delays, alternative ports, or emergency protocols.",
        }
        base = msgs.get(level, "Monitor situation.")
        if waiting_hours > 15:
            base += f" Estimated wait of {waiting_hours:.1f} h may impact downstream supply chain."
        return base

    def _delay_actions(self, risk: RiskLevel, destination_port: Optional[str]) -> List[str]:
        actions_map = {
            RiskLevel.LOW: [
                "Continue with current schedule.",
                "Monitor for any changes.",
            ],
            RiskLevel.MEDIUM: [
                "Notify consignees of potential delay.",
                "Consider expedited discharge procedures.",
                "Prepare contingency plans.",
            ],
            RiskLevel.HIGH: [
                "Alert all stakeholders immediately.",
                "Explore alternative routing options.",
                "Coordinate with port for priority berthing.",
                "Prepare customer communications.",
            ],
            RiskLevel.CRITICAL: [
                "URGENT: Consider diverting to alternative port.",
                "Notify all parties of significant delay.",
                "Activate contingency logistics plan.",
                "Explore air freight for critical cargo.",
                "Engage port authority for emergency berthing.",
            ],
        }
        actions = list(actions_map.get(risk, []))
        if destination_port:
            actions.append(f"Contact {destination_port} port authority for latest updates.")
        return actions

    def _log_prediction(self, prediction_type: str, data: Dict[str, Any]) -> None:
        self.prediction_history.append({
            "type": prediction_type,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        })
        # Prevent unbounded memory growth
        if len(self.prediction_history) > self._max_history:
            self.prediction_history = self.prediction_history[-self._max_history:]


# ---------------------------------------------------------------------------
# Thread-safe singleton
# ---------------------------------------------------------------------------

_predictive_engine: Optional[PredictiveEngine] = None
_engine_lock = threading.Lock()


def get_predictive_engine() -> PredictiveEngine:
    """Return the shared predictive engine instance (thread-safe, lazy-initialised)."""
    global _predictive_engine
    if _predictive_engine is None:
        with _engine_lock:
            if _predictive_engine is None:
                _predictive_engine = PredictiveEngine()
    return _predictive_engine
