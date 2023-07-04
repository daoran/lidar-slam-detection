#include "inference.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(inference_ext, m) {
  m.def("inference_init", &inference_init, "inference initialize",
        "scn_file"_a = 1, "rpn_file"_a = 2,
        "voxel_size"_a = 3, "coors_range"_a = 4,
        "max_points"_a = 5, "max_voxels"_a = 6, "max_points_use"_a = 7);

  m.def("inference_forward", &inference_forward, "inference forward",
        "points"_a = 1, "cls_preds"_a = 2, "box_preds"_a = 3, "label_preds"_a = 4);
}