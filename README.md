# Apple Classifier (Rust)

A tiny Rust project that demonstrates binary classification with logistic regression from scratch.

The model learns from two apple features:
- `color` (red/black)
- `size` (big/small)

It trains with gradient descent, uses a sigmoid activation, and tracks Binary Cross-Entropy loss across epochs.

## Run

```bash
cargo run
```

This project is intentionally minimal and educational: no ML frameworks, just core math and Rust.
