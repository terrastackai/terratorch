import torch


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def posemb_sincos_2d_with_gsd(h, w, dim, gsd=1.0, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"

    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** (2 * omega / dim)) * (gsd / 1.0)  # Adjusted for g

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def posemb_sincos_1d(pos, dim, temperature: int = 10000, dtype=torch.float32):
    assert dim % 2 == 0, "Feature dimension must be a multiple of 2 for sincos embedding"
    pos = torch.arange(pos) if isinstance(pos, int) else pos

    omega = torch.arange(dim // 2).to(pos) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    scaled_pos = pos[:, None] * omega[None, :]
    pe = torch.cat((scaled_pos.sin(), scaled_pos.cos()), dim=1)

    return pe.type(dtype)
