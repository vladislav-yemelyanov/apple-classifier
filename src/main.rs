#[derive(Debug)]
enum AppleColor {
    Red,   // Encoded as 1
    Black, // Encoded as 0
}

#[derive(Debug)]
enum AppleSize {
    Big,   // Encoded as 1
    Small, // Encoded as 0
}

#[derive(Debug)]
struct Apple {
    color: AppleColor, // Input feature
    size: AppleSize,   // Input feature
}

/// Compresses any real value into the range [0, 1].
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

// Binary Cross-Entropy loss: measures how far the prediction is from the true label.
fn bce(y_pred: f32, y: f32) -> f32 {
    -(y * y_pred.ln() + (1.0 - y) * (1.0 - y_pred).ln())
}

fn main() {
    // Tiny training dataset with two samples.
    let apples: Vec<Apple> = vec![
        Apple {
            color: AppleColor::Red,
            size: AppleSize::Big,
        },
        Apple {
            color: AppleColor::Black,
            size: AppleSize::Small,
        },
    ];

    // Matrix-like container: each row is [color, size].
    let mut features: Vec<Vec<f32>> = vec![];

    // Encode enum values into numeric feature vectors.
    for apple in &apples {
        let color: f32 = match apple.color {
            AppleColor::Red => 1.0,
            AppleColor::Black => 0.0,
        };

        let state: f32 = match apple.size {
            AppleSize::Big => 1.0,
            AppleSize::Small => 0.0,
        };

        // Keep feature order stable: index 0 -> color, index 1 -> size.
        let row = vec![color, state];

        features.push(row);
    }

    // Initialize model parameters for features: color and size.
    // Start from small values; training will move them to better parameters.
    let mut weights = vec![0.1, 0.1];
    let labels = vec![1.0, 0.0]; // 1.0 = good apple, 0.0 = bad apple
    // Step size for gradient descent.
    let learning_rate = 0.1;
    // Number of full passes over the dataset.
    let epochs = 10000;
    // Separate intercept term.
    let mut bias = 0.0;

    // Training loop (stochastic-style updates per sample).
    for epoch in 0..epochs {
        // Accumulate total loss for the current epoch.
        let mut epoch_loss = 0.0;

        for (i, feature) in features.iter().enumerate() {
            // Linear prediction (logit): color * w1 + size * w2 + bias.
            let y = feature[0] * weights[0] + feature[1] * weights[1] + bias;

            // Convert logit into probability of class "good".
            let y_pred = sigmoid(y);

            // Compare predicted probability against the true class.
            let loss = bce(y_pred, labels[i]);
            epoch_loss += loss;

            // For logistic regression with BCE, this is dL/d(logit).
            let error = y_pred - labels[i];

            if epoch % 2000 == 0 {
                println!("{:?}", apples[i]);

                match labels[i] {
                    1.0 => {
                        // Good apple class
                        println!("Good");
                    }
                    0.0 => {
                        // Bad apple class
                        println!("Bad");
                    }
                    _ => {
                        // Unused branch for safety
                    }
                }
                println!("    predicate: {}", y_pred);
                println!("    error: {}", error);
                println!("-----------------");
            }

            // Update bias using gradient descent.
            bias -= learning_rate * error;

            for j in 0..weights.len() {
                // Weight update: w_j = w_j - lr * error * x_j.
                weights[j] = weights[j] - learning_rate * error * feature[j];
            }
        }

        if epoch % 1000 == 0 {
            // Report average epoch loss for easier progress tracking.
            println!(
                "epoch {} avg_loss {}",
                epoch,
                epoch_loss / features.len() as f32
            );
        }
    }
}
