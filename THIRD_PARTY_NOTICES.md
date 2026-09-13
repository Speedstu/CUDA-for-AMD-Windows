# Third-party notices

This repository documents and orchestrates third-party software. It does not change the license of any dependency.

- **ZLUDA** — upstream `vosen/ZLUDA`; use under the upstream license terms.
- **PyTorch / LibTorch** — copyright and license belong to the PyTorch project and contributors.
- **AMD ROCm / HIP / rocBLAS / hipBLASLt / TheRock** — copyright and licenses belong to AMD and contributors.
- **CUDA headers/runtime packages** — NVIDIA components remain subject to NVIDIA's applicable licenses.

`local-artifacts/` is intentionally excluded from Git. It may contain recovered binary artifacts whose exact source provenance still needs to be established before public redistribution.

Do not upload the recovered multi-gigabyte dependency bundles as repository contents. Prefer official upstream downloads plus version/hash manifests.
