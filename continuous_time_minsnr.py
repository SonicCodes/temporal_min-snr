def get_snr_weights(timesteps): # (B, S) [0, 1]
    # Since SNR is (1 - t)
    snr = 1.0 - timesteps  # SNR values between 0 and 1
    normalized_snr = snr
    cum_snr = jnp.zeros_like(normalized_snr)
    cum_snr_decay = 0.96  # Adjust decay rate as needed

    for t in range(timesteps.shape[1]):
        if t == 0:
            cum_snr = cum_snr.at[:, t].set(normalized_snr[:, t])
        else:
            calculated_drift_snr = cum_snr_decay * cum_snr[:, t - 1] + (1 - cum_snr_decay) * normalized_snr[:, t]
            cum_snr = cum_snr.at[:, t].set(calculated_drift_snr)

    cum_snr = jnp.pad(cum_snr[:, :-1], ((0, 0), (1, 0)), mode="constant", constant_values=0)

    clipped_fused_snr = 1 - (1 - cum_snr * cum_snr_decay) * (1 - normalized_snr)
    fused_snr = 1 - (1 - cum_snr * cum_snr_decay) * (1 - normalized_snr)

    snr_weight = fused_snr
    return snr_weight
