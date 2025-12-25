// Simple NARX neural network (1 hidden layer, sigmoid activations)
class NARX {
    constructor(inputLag, outputLag, hiddenUnits, options = {}) {
        if (inputLag <= 0 || outputLag <= 0 || hiddenUnits <= 0) {
            throw new Error("inputLag, outputLag, and hiddenUnits must be positive integers.");
        }

        this.inputLag = inputLag;
        this.outputLag = outputLag;
        this.hiddenUnits = hiddenUnits;

        this.learningRate = options.learningRate ?? 0.01;

        this.inputSize = inputLag + outputLag; // combined regressor length

        // Weights: [hiddenUnit][inputIndex]
        this.weightsInputHidden = this.randomMatrix(hiddenUnits, this.inputSize);
        // Bias for each hidden unit
        this.biasHidden = this.randomArray(hiddenUnits);

        // Weights from hidden to single output
        this.weightsHiddenOutput = this.randomArray(hiddenUnits);
        // Output bias (scalar)
        this.biasOutput = this.randomScalar();
    }

    // ---------- Helpers ----------

    randomScalar() {
        return Math.random() * 2 - 1; // [-1, 1]
    }

    randomArray(length) {
        const arr = new Array(length);
        for (let i = 0; i < length; i++) {
            arr[i] = this.randomScalar();
        }
        return arr;
    }

    randomMatrix(rows, cols) {
        const m = new Array(rows);
        for (let r = 0; r < rows; r++) {
            m[r] = this.randomArray(cols);
        }
        return m;
    }

    // Sigmoid activation and derivative (derivative expects y = sigmoid(x))
    activation(x) {
        return 1 / (1 + Math.exp(-x));
    }

    activationDerivative(y) {
        return y * (1 - y);
    }

    // ---------- Core forward pass ----------

    /**
     * Forward pass for a single regressor vector.
     * @param {number[]} combined - length = inputLag + outputLag
     * @returns {{ hidden: number[], output: number, netOutput: number }}
     */
    forward(combined) {
        if (combined.length !== this.inputSize) {
            throw new Error(`Expected combined length ${this.inputSize}, got ${combined.length}`);
        }

        const hidden = new Array(this.hiddenUnits);

        // Hidden layer
        for (let h = 0; h < this.hiddenUnits; h++) {
            let sum = this.biasHidden[h];
            const wRow = this.weightsInputHidden[h];
            for (let k = 0; k < this.inputSize; k++) {
                sum += wRow[k] * combined[k];
            }
            hidden[h] = this.activation(sum);
        }

        // Output layer (single neuron)
        let netOutput = this.biasOutput;
        for (let h = 0; h < this.hiddenUnits; h++) {
            netOutput += this.weightsHiddenOutput[h] * hidden[h];
        }

        const output = this.activation(netOutput);
        return { hidden, output, netOutput };
    }

    // ---------- Training (series-parallel / teacher forcing) ----------

    /**
     * Train on a single input / output sequence.
     * @param {number[]} inputSequence  - exogenous input u(t)
     * @param {number[]} outputSequence - target output y(t)
     * @param {object} options
     *   - epochs: number of passes over the sequence
     *   - learningRate: overrides constructor learningRate
     * @returns {number[]} - array of epoch-wise MSE values
     */
    train(inputSequence, outputSequence, options = {}) {
        if (!Array.isArray(inputSequence) || !Array.isArray(outputSequence)) {
            throw new Error("inputSequence and outputSequence must be arrays.");
        }
        if (inputSequence.length !== outputSequence.length) {
            throw new Error("inputSequence and outputSequence must have the same length.");
        }

        const N = inputSequence.length;
        const epochs = options.epochs ?? 100;
        const lr = options.learningRate ?? this.learningRate;

        const start = Math.max(this.inputLag, this.outputLag);
        if (N <= start) {
            throw new Error("Sequences are too short for the given lags.");
        }

        const epochMSE = [];

        for (let e = 0; e < epochs; e++) {
            let sumSquaredError = 0;

            for (let t = start; t < N; t++) {
                // Build lagged regressor: [u(t-1)...u(t-inputLag), y(t-1)...y(t-outputLag)]
                const laggedInput = inputSequence.slice(t - this.inputLag, t);
                const laggedOutput = outputSequence.slice(t - this.outputLag, t);
                const combined = laggedInput.concat(laggedOutput);

                // Forward pass
                const { hidden, output } = this.forward(combined);

                // Error and output delta
                const target = outputSequence[t];
                const error = target - output;
                const deltaOut = error * this.activationDerivative(output);

                sumSquaredError += error * error;

                // --- Update output weights and bias ---
                for (let h = 0; h < this.hiddenUnits; h++) {
                    this.weightsHiddenOutput[h] += lr * deltaOut * hidden[h];
                }
                this.biasOutput += lr * deltaOut;

                // --- Update hidden weights and biases ---
                for (let h = 0; h < this.hiddenUnits; h++) {
                    // Backprop from output to hidden unit h
                    const deltaHidden =
                        deltaOut *
                        this.weightsHiddenOutput[h] *
                        this.activationDerivative(hidden[h]);

                    const wRow = this.weightsInputHidden[h];
                    for (let k = 0; k < this.inputSize; k++) {
                        wRow[k] += lr * deltaHidden * combined[k];
                    }
                    this.biasHidden[h] += lr * deltaHidden;
                }
            }

            const mse = sumSquaredError / (N - start);
            epochMSE.push(mse);
        }

        return epochMSE;
    }

    // ---------- Single-step prediction ----------

    /**
     * Predict the next output y(t+1) given full history of input and output.
     * Uses teacher forcing style: last true outputs are used as regressors.
     * @param {number[]} inputSequence
     * @param {number[]} outputSequence
     * @returns {number} predicted next output
     */
    predictNext(inputSequence, outputSequence) {
        if (
            inputSequence.length < this.inputLag ||
            outputSequence.length < this.outputLag
        ) {
            throw new Error("Not enough history for the given lags.");
        }

        const laggedInput = inputSequence.slice(-this.inputLag);
        const laggedOutput = outputSequence.slice(-this.outputLag);
        const combined = laggedInput.concat(laggedOutput);

        const { output } = this.forward(combined);
        return output;
    }

    /**
     * Predict outputs over a sequence in series-parallel mode (teacher forcing).
     * Returns predicted y_hat(t) for t = start..N-1.
     */
    predictSequence(inputSequence, outputSequence) {
        const N = inputSequence.length;
        const start = Math.max(this.inputLag, this.outputLag);
        const preds = new Array(N).fill(null);

        for (let t = start; t < N; t++) {
            const laggedInput = inputSequence.slice(t - this.inputLag, t);
            const laggedOutput = outputSequence.slice(t - this.outputLag, t);
            const combined = laggedInput.concat(laggedOutput);

            const { output } = this.forward(combined);
            preds[t] = output;
        }

        return { predictions: preds, startIndex: start };
    }

    // ---------- Multi-step free-run forecast (parallel mode) ----------

    /**
     * Multi-step forecast where past outputs are model predictions
     * instead of ground truth (classical NARX parallel mode).
     *
     * @param {number[]} inputHistory   - known past inputs
     * @param {number[]} outputHistory  - known past outputs
     * @param {number}   steps          - how many steps ahead to forecast
     * @param {number[]} [futureInputs] - optional exogenous inputs for the forecast horizon
     *                                   if omitted, uses the last known input repeatedly
     * @returns {number[]} predicted future outputs (length = steps)
     */
    forecast(inputHistory, outputHistory, steps, futureInputs = null) {
        const inputs = inputHistory.slice();
        const outputs = outputHistory.slice();

        if (
            inputs.length < this.inputLag ||
            outputs.length < this.outputLag
        ) {
            throw new Error("Not enough history for the given lags.");
        }

        const preds = [];

        for (let s = 0; s < steps; s++) {
            // Decide input at this step
            let newInput;
            if (futureInputs && s < futureInputs.length) {
                newInput = futureInputs[s];
            } else {
                newInput = inputs[inputs.length - 1]; // hold last input
            }
            inputs.push(newInput);

            const laggedInput = inputs.slice(-this.inputLag);
            const laggedOutput = outputs.slice(-this.outputLag);
            const combined = laggedInput.concat(laggedOutput);

            const { output } = this.forward(combined);
            preds.push(output);
            outputs.push(output); // feed prediction back as "past output"
        }

        return preds;
    }
}

// Export if used as a Node module
if (typeof module !== "undefined" && module.exports) {
    module.exports = NARX;
}

// ------------------ Demo when run directly ------------------
if (require.main === module) {
    // Simple synthetic system:
    // y(t) = 0.5 * y(t-1) + 0.3 * u(t-1)^2  (nonlinear in input)
    const N = 300;
    const inputSeq = new Array(N);
    const outputSeq = new Array(N);

    // Generate random input in [0, 1]
    for (let t = 0; t < N; t++) {
        inputSeq[t] = Math.random();
    }

    // Generate output with dynamics
    outputSeq[0] = 0;
    for (let t = 1; t < N; t++) {
        const uPrev = inputSeq[t - 1];
        const yPrev = outputSeq[t - 1];
        outputSeq[t] = 0.5 * yPrev + 0.3 * (uPrev * uPrev);
    }

    const inputLag = 2;
    const outputLag = 2;
    const hiddenUnits = 10;

    const model = new NARX(inputLag, outputLag, hiddenUnits, { learningRate: 0.05 });

    console.log("Training...");
    const mseHistory = model.train(inputSeq, outputSeq, { epochs: 100 });

    console.log("Final MSE:", mseHistory[mseHistory.length - 1]);

    const nextPred = model.predictNext(inputSeq, outputSeq);
    console.log("Next true y:", "unknown (we stopped at N)");
    console.log("Next predicted y:", nextPred);

    const future = model.forecast(inputSeq, outputSeq, 5);
    console.log("5-step forecast:", future);
}


// === GA helper methods for weight access ===
NARX.prototype.getNumWeights = function() {
    const hidden = this.hiddenUnits;
    const inSize = this.inputSize;
    // weightsInputHidden: hidden*inSize
    // biasHidden: hidden
    // weightsHiddenOutput: hidden
    // biasOutput: 1
    return hidden * inSize + hidden + hidden + 1;
};

NARX.prototype.getWeightsFlat = function() {
    const arr = [];
    const hidden = this.hiddenUnits;
    const inSize = this.inputSize;
    for (let h = 0; h < hidden; h++) {
        const wRow = this.weightsInputHidden[h];
        for (let k = 0; k < inSize; k++) {
            arr.push(wRow[k]);
        }
    }
    for (let h = 0; h < hidden; h++) {
        arr.push(this.biasHidden[h]);
    }
    for (let h = 0; h < hidden; h++) {
        arr.push(this.weightsHiddenOutput[h]);
    }
    arr.push(this.biasOutput);
    return arr;
};

NARX.prototype.setWeightsFlat = function(flat) {
    const hidden = this.hiddenUnits;
    const inSize = this.inputSize;
    let idx = 0;
    for (let h = 0; h < hidden; h++) {
        const wRow = this.weightsInputHidden[h];
        for (let k = 0; k < inSize; k++) {
            wRow[k] = flat[idx++];
        }
    }
    for (let h = 0; h < hidden; h++) {
        this.biasHidden[h] = flat[idx++];
    }
    for (let h = 0; h < hidden; h++) {
        this.weightsHiddenOutput[h] = flat[idx++];
    }
    this.biasOutput = flat[idx++];
};

// Export for Node/CommonJS
if (typeof module !== "undefined" && module.exports) {
    module.exports = NARX;
}
