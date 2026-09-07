# Third-party notices

The gfx12 WMMA fragment-packing approach in
`src/h3_vdn_sage_gfx12.hip` was informed by:

- SageAttention, thu-ml/SageAttention
- pull request #368, commit
  `66f5e64c9e36084c863a4480e570069245e58f90`
- copyright (c) 2024 SageAttention team
- copyright (c) 2026 Advanced Micro Devices, Inc.
- licensed under the Apache License, Version 2.0

The local implementation is modified to remove Torch/ATen, use raw HIP
pointers and a caller-owned workspace/stream, support H3's NHD VDN interval
mask, and execute BF16 PV.

The Apache-2.0 license text is included in the repository root as
[`LICENSE`](LICENSE).
