#include "startorch/tensor.hpp"
#include "startorch/common.hpp"
#include "startorch/device.hpp"
#include "startorch/format.hpp"
#include "startorch/layout.hpp"
#include "startorch/memory.hpp"

#include <cstdint>
#include <initializer_list>
#include <memory>

namespace startorch {
Tensor::Tensor(std::initializer_list<darkside::CPPValueToScalarValue> shape,
               std::initializer_list<darkside::CPPValueToScalarValue> storage,
               ScalarType scalar_type, Arena *arena)
    : scalar_type_(scalar_type), arena_(arena) {
  if (arena_ == nullptr)
    return;

  rank_ = shape.size();

  ScalarType shape_scalar_type = shape.begin()->getScalarType();
  Storage shape_storage(rank_, shape_scalar_type, arena_);

  switch (shape_scalar_type) {
#define ASSIGN_SHAPE(scalar_t, cpp_t)                                          \
  case scalar_t: {                                                             \
    cpp_t *ptr = nullptr;                                                      \
    uint64_t bytes = rank_ * darkside::getScalarTypeSize(scalar_t);            \
    Arena temp(bytes, HOST, CPU);                                              \
    bool switched = false;                                                     \
    if (arena_->getDevice().getDeviceType() == DeviceType::CPU)                \
      ptr = (cpp_t *)shape_storage.getData();                                  \
    else {                                                                     \
      switched = true;                                                         \
      ptr = (cpp_t *)temp.getData();                                           \
    }                                                                          \
    uint64_t i = 0;                                                            \
                                                                               \
    for (const auto &val : shape)                                              \
      ptr[i++] = val.getValue<cpp_t>();                                        \
                                                                               \
    if (switched)                                                              \
      Arena::copyData(shape_storage.getData(), ptr, bytes,                     \
                      DevicePair(temp.getDevice(), arena_->getDevice()));      \
    break;                                                                     \
  }

    ASSIGN_SHAPE(ScalarType::INT_8, int8_t)
    ASSIGN_SHAPE(ScalarType::INT_16, int16_t)
    ASSIGN_SHAPE(ScalarType::INT_32, int32_t)
    ASSIGN_SHAPE(ScalarType::INT_64, int64_t)
    ASSIGN_SHAPE(ScalarType::UNSIGNED_INT_8, uint8_t)
    ASSIGN_SHAPE(ScalarType::UNSIGNED_INT_16, uint16_t)
    ASSIGN_SHAPE(ScalarType::UNSIGNED_INT_32, uint32_t)
    ASSIGN_SHAPE(ScalarType::UNSIGNED_INT_64, uint64_t)
    ASSIGN_SHAPE(ScalarType::FLOAT_32, float)
    ASSIGN_SHAPE(ScalarType::FLOAT_64, double)

#undef ASSIGN_SHAPE
  default:
    break;
  }

  layout_ = Layout(shape_storage, Storage(), Storage(), Storage(), arena_);

  uint64_t size = storage.size();
  ScalarType storage_scalar_type = storage.begin()->getScalarType();
  shared_ = std::make_shared<Storage>(size, storage_scalar_type, arena_);

  switch (storage_scalar_type) {
#define ASSIGN_STORAGE(scalar_t, cpp_t)                                        \
  case scalar_t: {                                                             \
    cpp_t *ptr = nullptr;                                                      \
    uint64_t bytes = size * darkside::getScalarTypeSize(scalar_t);             \
    Arena temp(bytes, HOST, CPU);                                              \
    bool switched = false;                                                     \
    if (arena_->getDevice().getDeviceType() == DeviceType::CPU)                \
      ptr = (cpp_t *)shared_->getData();                                       \
    else {                                                                     \
      switched = true;                                                         \
      ptr = (cpp_t *)temp.getData();                                           \
    }                                                                          \
    uint64_t i = 0;                                                            \
                                                                               \
    for (const auto &val : storage)                                            \
      ptr[i++] = val.getValue<cpp_t>();                                        \
                                                                               \
    if (switched)                                                              \
      Arena::copyData(shared_->getData(), ptr, bytes,                          \
                      DevicePair(temp.getDevice(), arena_->getDevice()));      \
    break;                                                                     \
  }

    ASSIGN_STORAGE(ScalarType::INT_8, int8_t)
    ASSIGN_STORAGE(ScalarType::INT_16, int16_t)
    ASSIGN_STORAGE(ScalarType::INT_32, int32_t)
    ASSIGN_STORAGE(ScalarType::INT_64, int64_t)
    ASSIGN_STORAGE(ScalarType::UNSIGNED_INT_8, uint8_t)
    ASSIGN_STORAGE(ScalarType::UNSIGNED_INT_16, uint16_t)
    ASSIGN_STORAGE(ScalarType::UNSIGNED_INT_32, uint32_t)
    ASSIGN_STORAGE(ScalarType::UNSIGNED_INT_64, uint64_t)
    ASSIGN_STORAGE(ScalarType::FLOAT_32, float)
    ASSIGN_STORAGE(ScalarType::FLOAT_64, double)

#undef ASSIGN_STORAGE
  default:
    break;
  }
}

const Layout &Tensor::getLayout() const { return layout_; }

} // namespace startorch
