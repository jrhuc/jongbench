from __future__ import annotations

import math

import pytest

from jongbench.controls import MortalControlConfig


@pytest.mark.parametrize("epsilon", [-0.1, 1.1, math.nan, math.inf])
def test_control_config_rejects_invalid_epsilon(epsilon: float) -> None:
    with pytest.raises(ValueError, match="boltzmann_epsilon"):
        MortalControlConfig(boltzmann_epsilon=epsilon)


@pytest.mark.parametrize("temperature", [0.0, -1.0, math.nan, math.inf])
def test_control_config_rejects_invalid_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="boltzmann_temp"):
        MortalControlConfig(boltzmann_temp=temperature)
