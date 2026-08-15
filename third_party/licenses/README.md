# Redistributed runtime license texts

These files are verbatim copies of the license texts attached to the pinned
upstream versions used by the xhdfe 2.24.0.20260815 release toolchains. Their
line endings are preserved, including CRLF in the dlfcn-win32 file.

| File | Upstream path | SHA-256 |
|---|---|---|
| `GCC-13.2.0-COPYING3` | GCC 13.2.0 `COPYING3` | `8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903` |
| `GCC-13.2.0-COPYING.RUNTIME` | GCC 13.2.0 `COPYING.RUNTIME` | `9d6b43ce4d8de0c878bf16b54d8e7a10d9bd42b75178153e3af6a815bdc90f74` |
| `mingw-w64-11.0.1-winpthreads-COPYING` | mingw-w64 v11.0.1 `mingw-w64-libraries/winpthreads/COPYING` | `63263614cdd29f2f93cba85e992f041b31f9fc7b4033692f31269489a8a1b177` |
| `dlfcn-win32-1.4.1-COPYING` | dlfcn-win32 v1.4.1 `COPYING` | `4cc7ac997b9293db5919baf630100cc09b3508efdfe6a6611c95511fb863b3c7` |

Versioned upstream references:

- <https://gcc.gnu.org/git/?p=gcc.git;a=blob_plain;f=COPYING3;hb=releases/gcc-13.2.0>
- <https://gcc.gnu.org/git/?p=gcc.git;a=blob_plain;f=COPYING.RUNTIME;hb=releases/gcc-13.2.0>
- <https://github.com/mingw-w64/mingw-w64/blob/v11.0.1/mingw-w64-libraries/winpthreads/COPYING>
- <https://github.com/dlfcn-win32/dlfcn-win32/blob/v1.4.1/COPYING>

The separately downloadable corresponding-source bundle records the hashes
of the actual upstream source archives supplied to the release job. This
directory contains license texts only; it is not a substitute for that source.

CUDA plugin release bundles additionally require these exact inputs from the
CUDA 12.6 / CCCL 2.5.0 toolchain custody step:

| File | SHA-256 | Size |
|---|---|---:|
| `NVIDIA-CUDA-12.6-EULA.pdf` | `7c2dc636ad47cf67a0efb97d9c11246efcc471ac9d11eb8efceae3bfd56d8649` | 209850 |
| `NVIDIA-CCCL-2.5.0-LICENSE` | `01b767dcd7d36f42efb608076741cf83f154a995e198028cb698aadc3a43b63b` | 33838 |

They are redistribution/license materials, not GNU GPL Corresponding Source.
