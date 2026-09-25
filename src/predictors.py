"""B1 predictors for the reflex: wrong-physics simulator copies and a learned world model, plus their offline errors.

See docs/superpowers/specs/2026-09-25-b1-staleness-predictors-design.md (sections 3 and 4.2).
"""

import jax
import jax.numpy as jnp

BODY_PARAMS = ("inverse_mass", "inverse_inertia", "friction")


def level_name(level_path: str) -> str:
    return level_path.replace("/", "_").replace(".json", "")


def _params(state):
    """The physical parameters a phys predictor gets wrong: (polygon, circle, motor power, thruster power)."""

    def body(b):
        return {k: getattr(b, k) for k in BODY_PARAMS}

    return body(state.polygon), body(state.circle), state.joint.motor_power, state.thruster.power


def _with(state, polygon, circle, motor_power, thruster_power):
    return state.replace(
        polygon=state.polygon.replace(**polygon),
        circle=state.circle.replace(**circle),
        joint=state.joint.replace(motor_power=motor_power),
        thruster=state.thruster.replace(power=thruster_power),
    )


def phys_factors(key, state, error: float):
    """1 + error * (+-1) for every entry of _params(state), with a random sign per entry (body, joint, thruster).

    A fixed key gives every method and every error size the same signs: a paired comparison and a clean dose curve.
    """
    leaves, treedef = jax.tree.flatten(_params(state))
    keys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(
        treedef, [1 + error * jax.random.rademacher(k, x.shape).astype(x.dtype) for k, x in zip(keys, leaves)]
    )


def phys_step(env, key, state, action, params, factors):
    """One predictor step with wrong physics -> (obs, state), like env.step_env(...)[:2].

    Steps with the parameters x factors, then puts the true parameters back: the symbolic observation contains the
    parameters themselves, and a prediction must differ from the oracle's only through the motion.
    """
    wrong = _with(state, *jax.tree.map(jnp.multiply, _params(state), factors))
    nxt = env.step_env(key, wrong, action, params)[1]
    nxt = _with(nxt, *_params(state))
    return env.get_obs(nxt), nxt
