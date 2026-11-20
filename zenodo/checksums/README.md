# Checksum manifests

| File | Covers | Paths relative to | Verify with |
|---|---|---|---|
| `MD5SUMS_zenodo.txt` | every file in the Zenodo records | extracted archive root | `md5sum -c checksums/MD5SUMS_zenodo.txt` |
| `SHA256SUMS_zenodo.txt` | same, SHA-256 | extracted archive root | `shasum -a 256 -c checksums/SHA256SUMS_zenodo.txt` |
| `MD5SUMS_github.txt` | package-root documents and `github/` | repository root | `md5sum -c zenodo/checksums/MD5SUMS_github.txt` |
| `SHA256SUMS_github.txt` | same, SHA-256 | repository root | `shasum -a 256 -c zenodo/checksums/SHA256SUMS_github.txt` |

The manifests are generated from the assembled package and list exactly the files that were published
(every row of `MANIFEST.csv` whose `upload_to` is `github` or `zenodo`).
`MANIFEST.csv` additionally keeps rows marked `skipped(duplicate)` (a byte-identical copy already
published elsewhere) and `not_included` (large third-party raw data) as a record of the packaging decision.
