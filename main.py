import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

x = jnp.array([1.0, 2.0, 3.0])

print("JAX devices:", jax.devices())
print("dtype:", x.dtype)
print("x^2:", x**2)