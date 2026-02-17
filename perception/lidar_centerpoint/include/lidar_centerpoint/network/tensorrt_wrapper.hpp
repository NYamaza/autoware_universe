// Copyright 2021 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LIDAR_CENTERPOINT__NETWORK__TENSORRT_WRAPPER_HPP_
#define LIDAR_CENTERPOINT__NETWORK__TENSORRT_WRAPPER_HPP_

#include <lidar_centerpoint/centerpoint_config.hpp>
#include <tensorrt_common/tensorrt_common.hpp>

#include <NvInfer.h>

#include <iostream>
#include <memory>
#include <string>

namespace centerpoint
{

class TensorRTWrapper
{
public:
  explicit TensorRTWrapper(const CenterPointConfig & config);

  ~TensorRTWrapper();

  bool init(
    const std::string & onnx_path, const std::string & engine_path, const std::string & precision);

  tensorrt_common::TrtUniquePtr<nvinfer1::IExecutionContext> context_{nullptr};

bool enqueueV3(void ** bindings, cudaStream_t stream)
{
  // ここで内部のプライベートな engine_ や context_ を使って処理を完結させる
  int32_t nbIO = engine_->getNbIOTensors();
  for (int i = 0; i < nbIO; ++i) {
    context_->setTensorAddress(engine_->getIOTensorName(i), bindings[i]);
  }
  return context_->enqueueV3(stream);
}

// 名前を取得する関数も作っておく
const char* getIOTensorName(int32_t index) const {
  return engine_->getIOTensorName(index);
}



protected:
  virtual bool setProfile(
    nvinfer1::IBuilder & builder, nvinfer1::INetworkDefinition & network,
    nvinfer1::IBuilderConfig & config) = 0;

  CenterPointConfig config_;
  tensorrt_common::Logger logger_;

private:
  bool parseONNX(
    const std::string & onnx_path, const std::string & engine_path, const std::string & precision,
    size_t workspace_size = (1ULL << 30));

  bool saveEngine(const std::string & engine_path);

  bool loadEngine(const std::string & engine_path);

  bool createContext();

  tensorrt_common::TrtUniquePtr<nvinfer1::IRuntime> runtime_{nullptr};
  tensorrt_common::TrtUniquePtr<nvinfer1::IHostMemory> plan_{nullptr};
  tensorrt_common::TrtUniquePtr<nvinfer1::ICudaEngine> engine_{nullptr};
};

}  // namespace centerpoint



#endif  // LIDAR_CENTERPOINT__NETWORK__TENSORRT_WRAPPER_HPP_
