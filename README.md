# Image Caption Generator 🖼️

A deep learning project that generates natural language descriptions for images using CNN-LSTM architecture trained on the Flickr8k dataset.

## 📋 Overview

This project implements an image captioning system that:
- Extracts visual features using Xception (pre-trained on ImageNet)
- Generates captions using LSTM-based sequence model
- Provides both CLI and web interface for caption generation
- Uses beam search for improved caption quality

## 🏗️ Architecture

- **Feature Extractor**: Xception CNN (2048-dimensional feature vectors)
- **Caption Generator**: LSTM with embedding layer
- **Decoder**: Dense layers with softmax activation
- **Training**: Categorical cross-entropy loss with Adam optimizer

## 📁 Project Structure
```
IMAGE_CAPTION_GENERATOR/
├── data/
│   ├── Flicker8k_Dataset/       # Image dataset
│   └── Flickr8k_text/           # Caption annotations
├── models/
│   ├── model_0.h5 to model_9.h5 # Trained model weights (10 epochs)
│   └── tokenizer.p              # Tokenizer for text processing
├── src/
│   ├── main.py                  # Training script
│   ├── test.py                  # CLI inference script
│   └── app.py                   # Streamlit web application
├── descriptions.txt              # Cleaned captions
├── features.p                    # Extracted image features
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
TensorFlow 2.x
Keras
Streamlit
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/karthik-vvs/Image-Caption-Generator.git
cd Image-Caption-Generator
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download Flickr8k Dataset**
- Download from [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- Extract to `data/` directory with structure:
  - `data/Flicker8k_Dataset/` (images)
  - `data/Flickr8k_text/` (captions)

4. **Download Pre-trained Models**
- Place model files (`model_0.h5` to `model_9.h5`) in `models/` directory
- Place `tokenizer.p` in `models/` directory

### 📥 Model Files

Due to file size limitations, trained models are not included in this repository. You can:

## Train from scratch
```bash
python src/main.py
```

## 💻 Usage

### Web Application (Recommended)

Launch the Streamlit interface:
```bash
streamlit run src/app.py
```

Features:
- Upload any image (JPG, JPEG, PNG)
- Get instant caption predictions
- View example Flickr8k images
- Beam search for better captions

### Command Line Interface

Generate caption for a single image:
```bash
python src/test.py --image path/to/your/image.jpg
```

Example:
```bash
python src/test.py --image data/Flicker8k_Dataset/1859941832_7faf6e5fa9.jpg
```

### Training

Train the model from scratch:
```bash
python src/main.py
```

Training parameters:
- Epochs: 10
- Batch size: 32
- Vocabulary size: ~8,000 words
- Max caption length: 32 words

## 🎯 Model Performance

- **Dataset**: Flickr8k (8,000 images, 5 captions each)
- **Training Images**: 6,000
- **Vocabulary Size**: ~8,000 unique words
- **Architecture**: Xception + LSTM
- **Best Results**: Model from epoch 9 (`model_9.h5`)

## ⚠️ Important Notes

- The model works best with **Flickr8k-style images**: everyday scenes, people, outdoor activities, animals
- Images significantly different from the training data may produce generic captions
- Ensure uploaded images are in JPG, JPEG, or PNG format
- For images with alpha channel (RGBA), the code automatically converts to RGB

## 🔧 Key Features

### Data Processing
- Automatic text cleaning (lowercase, punctuation removal)
- Vocabulary building with tokenization
- Start/end tokens for sequence generation

### Feature Extraction
- Xception model for robust visual features
- 299x299 image preprocessing
- Feature caching for efficient training

### Caption Generation
- **Greedy Search** (test.py): Fast, single best prediction
- **Beam Search** (app.py): Better quality, top-3 candidates

## 📊 Technical Details

**Model Specifications:**
- Input: 2048-dim image features + variable-length text sequence
- Embedding: 256-dimensional word vectors
- LSTM: 256 hidden units
- Dropout: 0.5 for regularization
- Output: Softmax over vocabulary

**Training Details:**
- Loss: Categorical cross-entropy
- Optimizer: Adam
- Steps per epoch: Calculated from caption sequences
- Checkpointing: Model saved after each epoch

## 🛠️ Requirements

Create a `requirements.txt` file with:
```
tensorflow>=2.10.0
keras>=2.10.0
numpy>=1.23.0
pandas>=1.5.0
pillow>=9.0.0
matplotlib>=3.5.0
streamlit>=1.25.0
tqdm>=4.65.0
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Flickr8k Dataset**: M. Hodosh, P. Young and J. Hockenmaier (2013)
- **Xception Architecture**: François Chollet
- **Inspired by**: "Show and Tell" paper by Google Research

## 📧 Contact

Your Name - vurubindivskarthiksarma@gmail.com

Project Link: [https://github.com/karthik-vvs/Image-Caption-Generator]

## 🐛 Known Issues

- Model requires significant memory during training (~8GB RAM recommended)
- Inference time depends on beam search width (larger = slower but better)
- Dataset download is manual (not automated)

## 🔮 Future Enhancements

- [ ] Attention mechanism for better focus
- [ ] Support for larger datasets (MSCOCO)
- [ ] Multi-language caption generation
- [ ] Real-time video captioning
- [ ] BLEU score evaluation metrics
- [ ] Docker containerization

**Made with ❤️ using TensorFlow and Keras**
