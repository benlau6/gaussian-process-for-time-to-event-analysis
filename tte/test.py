from jax import numpy as jnp
import numpy as np

a = jnp.array([0, 1, 2, 3])
b = jnp.array([3, 4])
c = b

print(a.shape)
print(b.shape)
print(b[:, None].shape)
print(b[:, None].T.shape)

print(b)
print(b[:, None])
print(b[:, None].T)
