// Copyright 2023 TIER IV, Inc.
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

#include <tensorrt_common/tensorrt_common.hpp>

#include <NvInferPlugin.h>
#include <dlfcn.h>

#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <algorithm>

namespace
{
template <class T>
bool contain(const std::string & s, const T & v)
{
  return s.find(v) != std::string::npos;
}
}  // anonymous namespace

namespace tensorrt_common
{
nvinfer1::Dims get_input_dims(const std::string & onnx_file_path)
{
  Logger logger_;
  auto builder = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
  if (!builder) {
    logger_.log(nvinfer1::ILogger::Severity::kERROR, "Fail to create builder");
  }

  const auto explicitBatch =
    1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  auto network =
    TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
  if (!network) {
    logger_.log(nvinfer1::ILogger::Severity::kERROR, "Fail to create network");
  }

  auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    logger_.log(nvinfer1::ILogger::Severity::kERROR, "Fail to create builder config");
  }

  auto parser = TrtUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger_));
  if (!parser->parseFromFile(
        onnx_file_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR))) {
    logger_.log(nvinfer1::ILogger::Severity::kERROR, "Failed to parse onnx file");
  }

  const auto input = network->getInput(0);
  return input->getDimensions();
}

bool is_valid_precision_string(const std::string & precision)
{
  if (
    std::find(valid_precisions.begin(), valid_precisions.end(), precision) ==
    valid_precisions.end()) {
    std::stringstream message;
    message << "Invalid precision was specified: " << precision << std::endl
            << "Valid string is one of: [";
    for (const auto & s : valid_precisions) {
      message << s << ", ";
    }
    message << "] (case sensitive)" << std::endl;
    std::cerr << message.str();
    return false;
  } else {
    return true;
  }
}

TrtCommon::TrtCommon(
  const std::string & model_path, const std::string & precision,
  std::unique_ptr<nvinfer1::IInt8Calibrator> calibrator, const BatchConfig & batch_config,
  const size_t max_workspace_size, const BuildConfig & build_config,
  const std::vector<std::string> & plugin_paths)
: model_file_path_(model_path),
  calibrator_(std::move(calibrator)),
  precision_(precision),
  batch_config_(batch_config),
  max_workspace_size_(max_workspace_size),
  model_profiler_("Model"),
  host_profiler_("Host")
{
  if (!is_valid_precision_string(precision)) {
    return;
  }
  build_config_ = std::make_unique<const BuildConfig>(build_config);

  for (const auto & plugin_path : plugin_paths) {
    int32_t flags{RTLD_LAZY};
#if ENABLE_ASAN
    flags |= RTLD_NODELETE;
#endif
    void * handle = dlopen(plugin_path.c_str(), flags);
    if (!handle) {
      logger_.log(nvinfer1::ILogger::Severity::kERROR, "Could not load plugin library");
    }
  }
  runtime_ = TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger_));
  if (build_config_->dla_core_id != -1) {
    runtime_->setDLACore(build_config_->dla_core_id);
  }
  initLibNvInferPlugins(&logger_, "");
}

TrtCommon::~TrtCommon()
{
}

void TrtCommon::setup()
{
  if (!fs::exists(model_file_path_)) {
    is_initialized_ = false;
    return;
  }
  std::string engine_path = model_file_path_;
  if (model_file_path_.extension() == ".engine") {
    loadEngine(model_file_path_);
  } else if (model_file_path_.extension() == ".onnx") {
    fs::path cache_engine_path{model_file_path_};
    std::string ext;
    std::string calib_name = "";
    if (precision_ == "int8") {
      if (build_config_->calib_type_str == "Entropy") {
        calib_name = "EntropyV2-";
      } else if (build_config_->calib_type_str == "Legacy" || build_config_->calib_type_str == "Percentile") {
        calib_name = "Legacy-";
      } else {
        calib_name = "MinMax-";
      }
    }
    ext = calib_name + precision_;
    if (build_config_->quantize_first_layer) ext += "-firstFP16";
    if (build_config_->quantize_last_layer) ext += "-lastFP16";
    ext += "-batch" + std::to_string(batch_config_[0]) + ".engine";
    cache_engine_path.replace_extension(ext);

    printNetworkInfo(model_file_path_);

    if (fs::exists(cache_engine_path)) {
      loadEngine(cache_engine_path);
    } else {
      buildEngineFromOnnx(model_file_path_, cache_engine_path);
    }
    engine_path = cache_engine_path;
  } else {
    is_initialized_ = false;
    return;
  }

  context_ = TrtUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
  if (!context_) {
    is_initialized_ = false;
    return;
  }

  if (build_config_->profile_per_layer) {
    context_->setProfiler(&model_profiler_);
  }

  is_initialized_ = true;
}

bool TrtCommon::loadEngine(const std::string & engine_file_path)
{
  std::ifstream engine_file(engine_file_path, std::ios::binary);
  engine_file.seekg(0, std::ios::end);
  size_t size = engine_file.tellg();
  engine_file.seekg(0, std::ios::beg);
  std::vector<char> engine_data(size);
  engine_file.read(engine_data.data(), size);
  engine_ = TrtUniquePtr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(engine_data.data(), size));
  return true;
}

void TrtCommon::printNetworkInfo(const std::string & onnx_file_path)
{
  auto builder = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
  const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
  auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());

  if (precision_ == "fp16" || precision_ == "int8") {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, max_workspace_size_);

  auto parser = TrtUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger_));
  parser->parseFromFile(onnx_file_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR));
  
  std::cout << "Network Info Printed." << std::endl;
}

bool TrtCommon::buildEngineFromOnnx(const std::string & onnx_file_path, const std::string & output_engine_file_path)
{
  auto builder = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
  const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
  auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());

  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, max_workspace_size_);

  auto parser = TrtUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger_));
  parser->parseFromFile(onnx_file_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR));

  // Optimization Profile for TRT 10
  auto profile = builder->createOptimizationProfile();
  const auto input = network->getInput(0);
  const auto input_name = input->getName();
  const auto dims = input->getDimensions();
  
  profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{batch_config_.at(0), dims.d[1], dims.d[2], dims.d[3]});
  profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{batch_config_.at(1), dims.d[1], dims.d[2], dims.d[3]});
  profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{batch_config_.at(2), dims.d[1], dims.d[2], dims.d[3]});
  config->addOptimizationProfile(profile);

  auto plan = TrtUniquePtr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
  engine_ = TrtUniquePtr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(plan->data(), plan->size()));

  std::ofstream file(output_engine_file_path, std::ios::binary);
  file.write(reinterpret_cast<const char *>(plan->data()), plan->size());
  return true;
}

bool TrtCommon::isInitialized() { return is_initialized_; }

nvinfer1::Dims TrtCommon::getBindingDimensions(const int32_t index) const
{
    auto const name = engine_->getIOTensorName(index);
    return context_->getTensorShape(name);
}

int32_t TrtCommon::getNbBindings()
{
  return engine_->getNbIOTensors();
}

bool TrtCommon::setBindingDimensions(const int32_t index, const nvinfer1::Dims & dimensions) const
{
    auto const name = engine_->getIOTensorName(index);
    return context_->setInputShape(name, dimensions);
}

bool TrtCommon::enqueueV2(void ** bindings, cudaStream_t stream, cudaEvent_t * input_consumed)
{
    (void)input_consumed;

    return enqueueV3(bindings, stream);
}

bool TrtCommon::enqueueV3(void ** bindings, cudaStream_t stream)
{
    int32_t nbIO = engine_->getNbIOTensors();
    for (int i = 0; i < nbIO; ++i) {
        context_->setTensorAddress(engine_->getIOTensorName(i), bindings[i]);
    }
    return context_->enqueueV3(stream);
}

void TrtCommon::printProfiling()
{
  std::cout << host_profiler_ << std::endl << model_profiler_;
}

}  // namespace tensorrt_common
