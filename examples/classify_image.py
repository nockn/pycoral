# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Example using PyCoral to classify a given image using an Edge TPU.
To run this code, you must attach an Edge TPU attached to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.
Example usage:
```
bash examples/install_requirements.sh classify_image.py
python3 examples/classify_image.py \
  --model test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite  \
  --labels test_data/inat_bird_labels.txt \
  --directory test_data/hoge
```
"""

import argparse
import time
import glob

from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
import pandas as pd


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model", required=True, help="File path of .tflite file.")
    parser.add_argument("-l", "--labels", help="File path of labels file.")
    parser.add_argument("-k", "--top_k", type=int, default=1, help="Max number of classification results")
    parser.add_argument("-t", "--threshold", type=float, default=0.0, help="Classification score threshold")
    parser.add_argument("-c", "--count", type=int, default=5, help="Number of times to run inference")
    parser.add_argument("-d", "--directory", required=True, help="File path of images to be classified.")
    parser.add_argument("-o", "--output", help="File path and name of result csv file.")
    args = parser.parse_args()

    print(
        "Note: The first inference on Edge TPU is slow because it includes", "loading the model into Edge TPU memory."
    )

    df = pd.DataFrame(columns=["image", "label", "score"])

    file_glob = args.directory + "/*"
    files = glob.glob(file_glob)
    len_path = len(args.files)

    labels = read_label_file(args.labels) if args.labels else {}

    interpreter = make_interpreter(*args.model.split("@"))
    interpreter.allocate_tensors()

    size = common.input_size(interpreter)

    for file in files:
        print("----INFERENCE TIME----")
        image = Image.open(file).convert("RGB").resize(size, Image.ANTIALIAS)
        common.set_input(interpreter, image)
        for _ in range(args.count):
            start = time.perf_counter()
            interpreter.invoke()
            inference_time = time.perf_counter() - start
            classes = classify.get_classes(interpreter, args.top_k, args.threshold)
            print("%.1fms" % (inference_time * 1000))

        print("-------RESULTS--------")
        for c in classes:
            df = df.append(
                {"image": file[len_path + 1 :], "label": labels.get(c.id, c.id), "score": c.score}, ignore_index=True
            )
            print("%s: %.5f\n" % (labels.get(c.id, c.id), c.score))

    if args.output:
        path_output = args.output
    else:
        path_output = args.model[:-6] + "csv"

    df.to_csv(path_output, index=False)
    print("SAVED RESULT: %s" % (path_output))


if __name__ == "__main__":
    main()
