# Archives

The data that is not in git, packed and encrypted. Ask Martin for the password.

- `results.7z.001`, `results.7z.002` - every saved run (`results/`); one archive in two parts,
  split to stay under GitHub's 100 MB per file
- `sod_prepared.7z` - the prepared survey data (`datasets/sod/prepared/`)

## Unpacking

Run from the repo root; each archive holds its paths from the root, so the files land where
the code reads them.

```
7z x archives/results.7z.001
7z x archives/sod_prepared.7z
```

- **Windows without a command line:** in 7-Zip, open `results.7z.001` (both parts in the same
  folder; it finds the second itself) and extract into the repo root; the same for
  `sod_prepared.7z`.
- **Linux / macOS:** `7z` is in the `7zip` / `p7zip-full` package, or `brew install sevenzip`.
- the file names inside are readable without the password, the contents are not.
