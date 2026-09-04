from __future__ import annotations


class TransitionStrategy:
    name = "transition"

    @staticmethod
    def claims_source(_source: str) -> bool:
        # Transition has no signal sources of its own: unlike the other
        # strategies, it never claims a signal by source prefix in the main
        # per-source strategy_map loop. It only fires through the explicit
        # "mixed signal, no other strategy claimed it" fallback path at the
        # end of TechnicalSignal._produce_strategy_signals, which applies
        # fallback_multiplier() below. Returning True here previously made
        # every signal double as a full-strength "transition" signal
        # alongside whatever strategy actually claimed it.
        return False

    @staticmethod
    def fallback_multiplier() -> float:
        return 0.55
