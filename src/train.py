"""Training loop with Optax optimizer and Orbax checkpointing."""

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import shutil
from pathlib import Path


def create_optimizer(cfg: dict):
    """AdamW with linear warmup + cosine decay."""
    warmup = optax.linear_schedule(0.0, cfg["learning_rate"], cfg["warmup_steps"])
    decay = optax.cosine_decay_schedule(cfg["learning_rate"], cfg["max_steps"] - cfg["warmup_steps"])
    schedule = optax.join_schedules([warmup, decay], [cfg["warmup_steps"]])

    tx = optax.chain(
        optax.clip_by_global_norm(cfg["grad_clip"]),
        optax.adamw(schedule, weight_decay=cfg["weight_decay"]),
    )
    return tx


def make_train_step(model):
    """Create a JIT-compiled train step that closes over the model."""

    @jax.jit
    def train_step(state, x, y, rng):
        rng, dropout_rng = jax.random.split(rng)

        def loss_fn(params):
            logits = model.apply(
                {"params": params}, x,
                deterministic=False, rngs={"dropout": dropout_rng},
            )
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, rng

    return train_step


def make_eval_step(model):
    """Create a JIT-compiled eval step that closes over the model."""

    @jax.jit
    def eval_step(params, x, y):
        logits = model.apply({"params": params}, x, deterministic=True)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))

    return eval_step


class TrainState:
    """Minimal training state container (registered as JAX pytree)."""

    def __init__(self, params, tx):
        self.params = params
        self.tx = tx
        self.opt_state = tx.init(params)
        self.step = jnp.int32(0)

    def apply_gradients(self, grads):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        new_state = TrainState.__new__(TrainState)
        new_state.params = new_params
        new_state.tx = self.tx
        new_state.opt_state = new_opt_state
        new_state.step = self.step + 1
        return new_state

    def tree_flatten(self):
        children = (self.params, self.opt_state, self.step)
        aux_data = self.tx
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        state = cls.__new__(cls)
        state.params, state.opt_state, state.step = children
        state.tx = aux_data
        return state


jax.tree_util.register_pytree_node(
    TrainState,
    lambda s: s.tree_flatten(),
    TrainState.tree_unflatten,
)


def save_checkpoint(state, path: str, step: int):
    """Save checkpoint using Orbax."""
    ckpt_dir = (Path(path) / f"step_{step}").resolve()
    params_dir = ckpt_dir / "params"
    if params_dir.exists():
        shutil.rmtree(params_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(str(params_dir), state.params)
    print(f"  Checkpoint saved: {ckpt_dir}")


def load_checkpoint(path: str, params_template):
    """Restore params from Orbax checkpoint."""
    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(path, item=params_template)
    print(f"  Checkpoint restored from: {path}")
    return restored


def train(model, variables, train_loader, val_loader, cfg: dict):
    """Main training loop."""
    params = variables["params"]
    tx = create_optimizer(cfg)
    state = TrainState(params, tx)

    rng = jax.random.PRNGKey(cfg["seed"])
    train_step = make_train_step(model)
    eval_step = make_eval_step(model)

    print(f"Starting training for {cfg['max_steps']} steps...")
    for step in range(1, cfg["max_steps"] + 1):
        x, y = next(train_loader)
        state, loss, rng = train_step(state, x, y, rng)

        if step % 100 == 0:
            print(f"  Step {step:>6d} | train loss: {loss:.4f}")

        if step % cfg["eval_every"] == 0:
            val_losses = []
            for _ in range(20):
                vx, vy = next(val_loader)
                vl = eval_step(state.params, vx, vy)
                val_losses.append(float(vl))
            avg_val = sum(val_losses) / len(val_losses)
            print(f"  Step {step:>6d} | val loss:   {avg_val:.4f}")

        if step % cfg["save_every"] == 0:
            save_checkpoint(state, cfg["checkpoint_dir"], step)

    if cfg["max_steps"] % cfg["save_every"] != 0:
        save_checkpoint(state, cfg["checkpoint_dir"], cfg["max_steps"])
    print("Training complete.")
    return state