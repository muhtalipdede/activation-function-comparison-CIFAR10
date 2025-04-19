# 🧠 Activation Function Comparison with CNN (CIFAR-10)

This project demonstrates how different activation functions (Sigmoid, ReLU, GELU) affect the performance of a convolutional neural network (CNN) on the CIFAR-10 dataset using PyTorch.

---

## 🚀 Setup & Run Instructions

To set up the project environment and run the training script, follow these steps:

```bash
# 1. Create a virtual environment
python3 -m venv venv

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the training script
python __main__.py
```

## 🧮 Activation Functions

### Sigmoid

The sigmoid function maps any real value to the range (0, 1), which is useful for binary classification, but it suffers from vanishing gradients.

$\text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}$

### ReLU

ReLU is widely used in deep learning due to its simplicity and ability to mitigate the vanishing gradient problem.

$\text{ReLU}(x) = \max(0, x)$

### GELU

GELU is a smoother alternative to ReLU that incorporates the input value and its distribution. It's used in many modern architectures (e.g., Transformers).

$\text{GELU}(x) = 0.5 \cdot x \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \cdot (x + 0.044715 \cdot x^3)\right)\right)$

##  Running the Code

To run the individual activation function scripts, you can execute the following commands:

```bash
python sigmoid.py
python relu.py
python gelu.py
```

## 🔗 References

```ref
@misc{hendrycks2023gaussianerrorlinearunits,
    title={Gaussian Error Linear Units (GELUs)}, 
    author={Dan Hendrycks and Kevin Gimpel},
    year={2023},
    eprint={1606.08415},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/1606.08415}, 
}
@misc{lee2023gelu,
    title={GELU Activation Function in Deep Learning: A Comprehensive Mathematical Analysis and Performance},
    author={Minhyeok Lee},
    year={2023},
    eprint={2305.12073},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={<https://arxiv.org/abs/2305.12073}>},
}
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
