import diffusers
from diffusers import FlaxDDIMScheduler
import jax
import jax.numpy as jnp
# Create a FlaxDDIMScheduler object
scheduler = FlaxDDIMScheduler(clip_sample=False)
scheduler_state = scheduler.create_state()
scheduler_state = scheduler.set_timesteps(scheduler_state, 50)


def compute_snr(timesteps):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = scheduler_state.common.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    alpha = sqrt_alphas_cumprod[timesteps]
    sigma = sqrt_one_minus_alphas_cumprod[timesteps]
    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr
batch = jax.random.normal(jax.random.PRNGKey(0), (4, 10, 3, 256, 256))
timesteps = jax.random.randint(jax.random.PRNGKey(0), (batch.shape[0],batch.shape[1]), 0, scheduler.config.num_train_timesteps)
snr = compute_snr(timesteps)
snr_gamma = 5.0

# snr_loss_weights = jnp.where(snr < snr_gamma, snr, jnp.ones_like(snr) * snr_gamma) / (snr + 1)
# print(snr_loss_weights.round(2))



# MIN SNR with temporal decay , from the DF paper implementation below, will be back with my coffee mug in a moment :D
clipped_snr = jnp.where(snr < snr_gamma, snr, jnp.ones_like(snr) * snr_gamma)
normalized_snr = snr / snr_gamma
normalized_clipped_snr = clipped_snr/ snr_gamma

cum_snr = jnp.zeros_like(normalized_snr)
cum_snr_decay = 0.96
for t in range(0, batch.shape[1]):
    if t == 0:
        cum_snr.at[:, t].set(normalized_clipped_snr[:, t])
    else:
        calculated_drift_snr = cum_snr_decay * cum_snr[:, t - 1] + (1 - cum_snr_decay) * normalized_clipped_snr[:, t]
        cum_snr.at[:, t].set(calculated_drift_snr)

print(cum_snr.shape )
cum_snr = jnp.pad(cum_snr[:, :-1], ((0, 0), (1, 0)), mode="empty")
clipped_fused_snr = 1 - (1 - cum_snr * cum_snr_decay) * (1 - normalized_clipped_snr)
fused_snr = 1 - (1 - cum_snr * cum_snr_decay) * (1 - normalized_snr)

snr_weight = None
pred_objective = "v" # or eps or x0
if pred_objective == "eps":
    snr_weight = clipped_fused_snr/fused_snr
elif pred_objective == "x0":
    snr_weight = clipped_fused_snr * snr_gamma
elif pred_objective == "v":
    snr_weight = clipped_fused_snr * snr_gamma / (fused_snr * snr_gamma + 1)
else:
    raise ValueError(f"unknown objective {pred_objective}")
print(timesteps)
print(snr_weight.round(2), snr_weight.mean())
print(snr_weight.shape)
