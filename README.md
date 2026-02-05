# Landmark Detector

A modern, production-ready implementation of facial landmark detection using PyTorch. This library provides accurate and efficient detection of facial landmarks with support for multiple detection backends and optimization frameworks.

## Features

- **Multiple Landmark Formats**: Support for 68-point and 39-point facial landmark detection
- **Flexible Architecture**: Modular design supporting multiple backbone networks and face detectors
- **Production Optimizations**: ONNX and OpenVINO inference support for deployment
- **Real-time Performance**: Optimized models achieve real-time inference on CPU
- **Face Alignment**: Automatic face alignment and cropping utilities
- **Multiple Detection Backends**: Integration with MTCNN, FaceBoxes, and RetinaFace detectors

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

Run batch detection on a folder of images:

```bash
python test_batch_detections.py --backbone MobileFaceNet --detector Retinaface
```

**Available Options:**
- `--backbone`: Choose from `MobileNet`, `PFLD`, or `MobileFaceNet`
- `--detector`: Choose from `MTCNN`, `FaceBoxes`, or `Retinaface`

### Real-time Camera Inference

**ONNX-optimized with MTCNN:**
```bash
python test_camera_mtcnn_onnx.py
```

**ONNX-optimized lightweight detector (real-time CPU):**
```bash
python test_camera_light_onnx.py
```

**OpenVINO-optimized (10x faster than ONNX):**
```bash
python test_camera_mobilefacenet_openvino.py
```

## Architecture

### Supported Backbone Networks

- **MobileNet**: Lightweight architecture with good accuracy
- **PFLD**: Practical Facial Landmark Detector, optimized for mobile devices
- **MobileFaceNet**: Efficient architecture designed specifically for face recognition tasks

### Supported Face Detectors

- **MTCNN**: Multi-task Cascaded Convolutional Networks
- **FaceBoxes**: Real-time face detection
- **RetinaFace**: State-of-the-art face detection with high accuracy

### Recommended Configuration

For best performance, use `Retinaface` as the detector with `MobileFaceNet` as the backbone.

## Performance Benchmarks

### 300W Dataset Results

#### Inter-ocular Normalization (ION)

| Algorithm | Common | Challenge | Full Set | Parameters (M) |
|-----------|--------|-----------|----------|----------------|
| ResNet18 (224×224) | 3.73 | 7.14 | 4.39 | 11.76 |
| Res2Net50_v1b (224×224) | 3.43 | 6.77 | 4.07 | 26.00 |
| Res2Net50_v1b_SE (224×224) | 3.37 | 6.67 | 4.01 | 27.05 |
| Res2Net50_v1b_ExternalData (224×224) | 3.30 | 5.92 | 3.81 | 26.00 |
| HRNet_w18_small_v2 (224×224) | 3.57 | 6.85 | 4.20 | 13.83 |

#### Lightweight Models

| Algorithm | Common | Challenge | Full Set | Parameters (M) | CPU Inference (s) |
|-----------|--------|-----------|----------|----------------|-------------------|
| MobileNetV2 (224×224) | 3.70 | 7.27 | 4.39 | 3.74 | 1.2 |
| MobileNetV2_SE (224×224) | 3.63 | 7.01 | 4.28 | 4.15 | - |
| MobileNetV2_SE_RE (224×224) | 3.63 | 6.66 | 4.21 | 4.15 | - |
| MobileNetV2_ExternalData (224×224) | 3.48 | 6.0 | 3.96 | 3.74 | 1.2 |
| MobileNetV2 (56×56) | 4.50 | 8.50 | 5.27 | 3.74 | 0.01 |
| MobileNetV2_SE_ExternalData (56×56) | 4.10 | 6.89 | 4.64 | 4.10 | 0.01 |
| PFLD_ExternalData (112×112) | 3.49 | 6.01 | 3.97 | 0.73 | 0.01 |
| MobileFaceNet_ExternalData (112×112) | 3.30 | 5.69 | 3.76 | 1.01 | - |

*Note: SE = Squeeze-and-Excitation module, RE = Random Erasing module*

#### Heatmap-based Models

| Algorithm | Common | Challenge | Full Set | Parameters (M) |
|-----------|--------|-----------|----------|----------------|
| Hourglass2 | 3.06 | 5.54 | 3.55 | 8.73 |

## Project Structure

```
.
├── common/              # Common utilities and helper functions
├── models/              # Model definitions and architectures
├── FaceBoxes/           # FaceBoxes face detector implementation
├── MTCNN/               # MTCNN face detector implementation
├── Retinaface/          # RetinaFace detector implementation
├── utils/               # Utility functions for alignment and visualization
├── vision/              # Vision utilities and SSD implementations
├── checkpoint/          # Model checkpoints directory
├── onnx/                # ONNX model files
├── openvino/            # OpenVINO model files
└── test_*.py            # Test and inference scripts
```

## Model Checkpoints

Pre-trained model checkpoints should be placed in the `checkpoint/` directory. Refer to the individual test scripts for specific model requirements.

## Output

The library generates two types of outputs:

1. **Detection Results**: Images with detected faces and landmarks visualized
2. **Aligned Faces**: Cropped and aligned face images saved to `results_aligned/`

## Development

This project follows modern Python best practices and is designed for easy extension and customization. The modular architecture allows for easy integration of new detectors or backbone networks.

## License

This project is open source. Please refer to individual component licenses where applicable.

## Citation

If you use this library in your research, please cite:

```bibtex
@misc{LandmarkDetector,
  title={Landmark Detector: A Fast and Accurate Facial Landmark Detection Implementation},
  year={2024},
  note={Open-source software}
}
```

## Acknowledgments

This implementation builds upon several excellent open-source projects:

- PyTorch RetinaFace implementation
- 3DDFA_V2
- PFLD PyTorch implementation