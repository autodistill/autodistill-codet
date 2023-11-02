<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill CoDet Module

This repository contains the code supporting the CoDet base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[CoDet](https://github.com/CVMI-Lab/CoDet) is an open vocabulary zero-shot object detection model. The model was described in the "CoDet: Co-Occurrence Guided Region-Word Alignment for Open-Vocabulary Object Detection" published by Chuofan Ma, Yi Jiang, Xin Wen, Zehuan Yuan, Xiaojuan Qi. The paper was submitted to NeurIPS2023.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [CoDet Autodistill documentation](https://autodistill.github.io/autodistill/base_models/codet/).

## Installation

To use CoDet with autodistill, you need to install the following dependency:

```bash
pip3 install autodistill-codet
```

## Quickstart

When you first run the model, it will download CoDet and its dependencies, as well as the required model configuration and weights. The output during the download process will be verbose. If you stop the download process before it has finished, run `rm -rf ~/.cache/autodistill/CoDet` before running the model again. This ensures that you don't work from a part-installed CoDet setup.

When the `predict()` function runs, the output will also be verbose. You can ignore the output printed to the console that appears when you call `predict()`.

You can only predict classes in the LVIS vocabulary. You can see a list of supported classes in the `class_names.json` file in the [autodistill-codet GitHub repository](https://github.com/autodistill/autodistill-codet).

Use the code snippet below to get started:

```python
from autodistill_codet import CoDet
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import cv2

# define an ontology to map class names to our CoDet prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = CoDet(
    ontology=CaptionOntology(
        {
            "person": "person"
        }
    )
)

# run inference on an image and display the results
# class_names is a list of all classes supported by the model
# class_names can be used to turn the class_id values from the model into human-readable class names
predictions, class_names = base_model.predict("./context_images/1.jpeg")
image = cv2.imread("./context_images/1.jpeg")

plot(
  image=image,
  detections=predictions,
  classes=class_names
)

# run inference on a folder of images and save the results
base_model.label("./context_images", extension=".jpeg")
```


## License

This project is licensed under an [Apache 2.0 license](LICENSE), except where files explicitly note a license.

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!