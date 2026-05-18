#include "startorch/memory.hpp"
#include "startorch/common.hpp"
#include "startorch/device.hpp"
#include "startorch/format.hpp"

#include "darkside/kernel.cuh"

#include <algorithm>
#include <cstdint>

namespace darkside {
uint64_t getScalarTypeSize(startorch::ScalarType scalar_type) {
  switch (scalar_type) {
  case startorch::ScalarType::UNKNOWN_SCALAR:
    return 0;

  case startorch::ScalarType::FLOAT_32:
    return sizeof(float);

  case startorch::ScalarType::FLOAT_64:
    return sizeof(double);

  case startorch::ScalarType::INT_8:
    return sizeof(int8_t);

  case startorch::ScalarType::INT_16:
    return sizeof(int16_t);

  case startorch::ScalarType::INT_32:
    return sizeof(int32_t);

  case startorch::ScalarType::INT_64:
    return sizeof(int64_t);

  case startorch::ScalarType::UNSIGNED_INT_8:
    return sizeof(uint8_t);

  case startorch::ScalarType::UNSIGNED_INT_16:
    return sizeof(uint16_t);

  case startorch::ScalarType::UNSIGNED_INT_32:
    return sizeof(uint32_t);

  case startorch::ScalarType::UNSIGNED_INT_64:
    return sizeof(uint64_t);

  default:
    return 0;
  }
}
} // namespace darkside

namespace startorch {
#define INSTANTIATE(macro)                                                                                                                                     \
  macro(float, ScalarType::FLOAT_32) macro(double, ScalarType::FLOAT_64) macro(int8_t, ScalarType::INT_8) macro(int16_t, ScalarType::INT_16)                   \
      macro(int32_t, ScalarType::INT_32) macro(int64_t, ScalarType::INT_64) macro(uint8_t, ScalarType::UNSIGNED_INT_8)                                         \
          macro(uint16_t, ScalarType::UNSIGNED_INT_16) macro(uint32_t, ScalarType::UNSIGNED_INT_32) macro(uint64_t, ScalarType::UNSIGNED_INT_64)

#define INSTANTIATE_REFERENCE_ELEMENT(T, S)                                                                                                                    \
  Element::Element(T *data, Device *device) : data_(data), device_(device), scalar_type_(S), owner_type_(OwnerType::REFERENCE) {}
INSTANTIATE(INSTANTIATE_REFERENCE_ELEMENT)
#undef INSTANTIATE_REFERENCE_ELEMENT

#define INSTANTIATE_OWNED_ELEMENT(T, S)                                                                                                                        \
  Element::Element(T value, Device *device) {                                                                                                                  \
    if (device == nullptr)                                                                                                                                     \
      return;                                                                                                                                                  \
                                                                                                                                                               \
    data_ = device->makeData(sizeof(T));                                                                                                                       \
                                                                                                                                                               \
    if (data_ == nullptr)                                                                                                                                      \
      return;                                                                                                                                                  \
                                                                                                                                                               \
    DevicePair(device, &AMD5625U).copyData(data_, &value, sizeof(T));                                                                                         \
                                                                                                                                                               \
    device_ = device;                                                                                                                                          \
    scalar_type_ = S;                                                                                                                                          \
    owner_type_ = OwnerType::OWNED;                                                                                                                            \
  }
INSTANTIATE(INSTANTIATE_OWNED_ELEMENT)
#undef INSTANTIATE_OWNED_ELEMENT
#undef INSTANTIATE

Element::Element(const Element &other) {
  switch (other.owner_type_) {
  case OwnerType::OWNED: {
    uint64_t bytes = darkside::getScalarTypeSize(other.scalar_type_);
    data_ = other.device_->makeData(bytes);

    if (data_ == nullptr)
      break;

    DevicePair(other.device_, other.device_).copyData(data_, other.data_, bytes);

    device_ = other.device_;
    scalar_type_ = other.scalar_type_;
    owner_type_ = OwnerType::OWNED;

    break;
  }
  case OwnerType::REFERENCE:
    data_ = other.data_;
    device_ = other.device_;
    scalar_type_ = other.scalar_type_;
    owner_type_ = other.owner_type_;

    break;

  default:
    break;
  }
}

Element::Element(Element &&other) noexcept : data_(other.data_), device_(other.device_), scalar_type_(other.scalar_type_), owner_type_(other.owner_type_) {
  other.data_ = nullptr;
  other.device_ = nullptr;
  other.scalar_type_ = ScalarType::UNKNOWN_SCALAR;
  other.owner_type_ = OwnerType::UNKNOWN_OWNER;
}

Element::~Element() {
  switch (owner_type_) {
  case OwnerType::OWNED: {
    uint64_t bytes = darkside::getScalarTypeSize(scalar_type_);
    uint8_t *tail = static_cast<uint8_t *>(device_->getData()) + device_->getOffset();

    if (static_cast<uint8_t *>(data_) + bytes == tail)
      device_->freeData(bytes);

    break;
  }
  default:
    break;
  }

  data_ = nullptr;
  device_ = nullptr;
  scalar_type_ = ScalarType::UNKNOWN_SCALAR;
  owner_type_ = OwnerType::UNKNOWN_OWNER;
}

Element &Element::operator=(const Element &other) {
  if (this == &other)
    return *this;

  if (other.owner_type_ == OwnerType::OWNED && data_ != nullptr && scalar_type_ == other.scalar_type_) {
    uint64_t bytes = darkside::getScalarTypeSize(scalar_type_);

    DevicePair(device_, other.device_).copyData(data_, other.data_, bytes);

    return *this;
  }

  Element temp(other);

  std::swap(data_, temp.data_);
  std::swap(device_, temp.device_);
  std::swap(scalar_type_, temp.scalar_type_);
  std::swap(owner_type_, temp.owner_type_);

  return *this;
}

Element &Element::operator=(Element &&other) noexcept {
  if (this == &other)
    return *this;

  if (owner_type_ == OwnerType::REFERENCE && other.data_ != nullptr && scalar_type_ == other.scalar_type_) {
    uint64_t bytes = darkside::getScalarTypeSize(scalar_type_);

    DevicePair(device_, other.device_).copyData(data_, other.data_, bytes);

    if (other.owner_type_ == OwnerType::OWNED) {
      bytes = darkside::getScalarTypeSize(other.scalar_type_);
      uint8_t *tail = static_cast<uint8_t *>(other.device_->getData()) + other.device_->getOffset();

      if (static_cast<uint8_t *>(other.data_) + bytes == tail)
        other.device_->freeData(bytes);
    }

    return *this;
  }

  if (owner_type_ == OwnerType::OWNED) {
    uint64_t bytes = darkside::getScalarTypeSize(scalar_type_);
    uint8_t *tail = static_cast<uint8_t *>(device_->getData()) + device_->getOffset();

    if (static_cast<uint8_t *>(data_) + bytes == tail)
      device_->freeData(bytes);
  }

  data_ = other.data_;
  device_ = other.device_;
  scalar_type_ = other.scalar_type_;
  owner_type_ = other.owner_type_;

  other.data_ = nullptr;
  other.device_ = nullptr;
  other.scalar_type_ = ScalarType::UNKNOWN_SCALAR;
  other.owner_type_ = OwnerType::UNKNOWN_OWNER;

  return *this;
}

void *Element::getData() { return data_; }
const void *Element::getData() const { return data_; }
Device *Element::getDevice() const { return device_; }
ScalarType Element::getScalarType() const { return scalar_type_; }
OwnerType Element::getOwnerType() const { return owner_type_; }

template <ScalarType S> Element element_cast(const Element &element, Device *device) {
  if (element.getData() == nullptr)
    return Element();

  if (element.getScalarType() == S && element.getDevice() == device)
    return Element(element);

  Element result;

  darkside::ScalarTypeToCPPType(element.getScalarType(), [&]<typename N>(darkside::CPPTypeToScalarType<N>) {
    darkside::ScalarTypeToCPPType(S, [&]<typename T>(darkside::CPPTypeToScalarType<T>) {
      result = Element(static_cast<T>(0), device);

      if (result.getData<T>() == nullptr)
        return;

      darkside::castData<T, N>(result.getData<T>(), element.getData<N>(), 1, device, element.getDevice());
    });
  });

  return result;
}

#define INSTANTIATE(macro)                                                                                                                                     \
  macro(ScalarType::FLOAT_32) macro(ScalarType::FLOAT_64) macro(ScalarType::INT_8) macro(ScalarType::INT_16) macro(ScalarType::INT_32)                         \
      macro(ScalarType::INT_64) macro(ScalarType::UNSIGNED_INT_8) macro(ScalarType::UNSIGNED_INT_16) macro(ScalarType::UNSIGNED_INT_32)                        \
          macro(ScalarType::UNSIGNED_INT_64)

#define INSTANTIATE_ELEMENT_CAST(S) template Element element_cast<S>(const Element &element, Device *device);
INSTANTIATE(INSTANTIATE_ELEMENT_CAST)
#undef INSTANTIATE_ELEMENT_CAST
#undef INSTANTIATE

Storage::Storage(uint64_t size, ScalarType scalar_type, Device *device) {
  if (size == 0 || device == nullptr || scalar_type == ScalarType::UNKNOWN_SCALAR)
    return;

  uint64_t bytes = size * darkside::getScalarTypeSize(scalar_type);
  data_ = device_->makeData(bytes);

  if (data_ == nullptr)
    return;

  size_ = size;
  scalar_type_ = scalar_type;
  device_ = device;
}

Storage::Storage(const Storage &other) {
  if (other.size_ == 0)
    return;

  uint64_t bytes = other.size_ * darkside::getScalarTypeSize(other.scalar_type_);
  data_ = other.device_->makeData(bytes);

  if (data_ == nullptr)
    return;

  DevicePair(other.device_, other.device_).copyData(data_, other.data_, bytes);

  size_ = other.size_;
  scalar_type_ = other.scalar_type_;
  device_ = other.device_;
}

Storage::Storage(Storage &&other) noexcept : data_(other.data_), size_(other.size_), scalar_type_(other.scalar_type_), device_(other.device_) {
  other.data_ = nullptr;
  other.size_ = 0;
  other.scalar_type_ = ScalarType::UNKNOWN_SCALAR;
  other.device_ = nullptr;
}

Storage::~Storage() {
  if (size_ == 0)
    return;

  uint64_t bytes = size_ * darkside::getScalarTypeSize(scalar_type_);
  uint8_t *tail = static_cast<uint8_t *>(device_->getData()) + device_->getOffset();

  if (static_cast<uint8_t *>(data_) + bytes == tail)
    device_->freeData(bytes);

  data_ = nullptr;
  size_ = 0;
  scalar_type_ = ScalarType::UNKNOWN_SCALAR;
  device_ = nullptr;
}

Storage &Storage::operator=(const Storage &other) {
  if (this == &other)
    return *this;

  if (other.size_ == 0) {
    if (size_ > 0) {
      uint64_t bytes = size_ * darkside::getScalarTypeSize(scalar_type_);
      uint8_t *tail = static_cast<uint8_t *>(device_->getData()) + device_->getOffset();

      if (static_cast<uint8_t *>(data_) + bytes == tail)
        device_->freeData(bytes);
    }

    data_ = nullptr;
    size_ = 0;
    scalar_type_ = ScalarType::UNKNOWN_SCALAR;
    device_ = nullptr;

    return *this;
  }

  uint64_t bytes = other.size_ * darkside::getScalarTypeSize(other.scalar_type_);

  if (device_ == other.device_ && size_ == other.size_ && scalar_type_ == other.scalar_type_) {
    DevicePair(device_, other.device_).copyData(data_, other.data_, bytes);

    return *this;
  }

  void *data = other.device_->makeData(bytes);

  if (data == nullptr)
    return *this;

  DevicePair(other.device_, other.device_).copyData(data, other.data_, bytes);

  if (size_ > 0) {
    bytes = size_ * darkside::getScalarTypeSize(scalar_type_);
    uint8_t *tail = static_cast<uint8_t *>(device_->getData()) + device_->getOffset();

    if (static_cast<uint8_t *>(data_) + bytes == tail)
      device_->freeData(bytes);
  }

  device_ = other.device_;
  size_ = other.size_;
  scalar_type_ = other.scalar_type_;
  data_ = data;

  return *this;
}

Storage &Storage::operator=(Storage &&other) noexcept {
  if (this == &other)
    return *this;

  if (size_ > 0) {
    uint64_t bytes = size_ * darkside::getScalarTypeSize(scalar_type_);
    uint8_t *tail = static_cast<uint8_t *>(device_->getData()) + device_->getOffset();

    if (static_cast<uint8_t *>(data_) + bytes == tail)
      device_->freeData(bytes);
  }

  data_ = other.data_;
  size_ = other.size_;
  scalar_type_ = other.scalar_type_;
  device_ = other.device_;

  other.data_ = nullptr;
  other.size_ = 0;
  other.scalar_type_ = ScalarType::UNKNOWN_SCALAR;
  other.device_ = nullptr;

  return *this;
}

Element Storage::operator[](uint64_t index) {
  if (index >= size_)
    return Element();

  Element element;

  darkside::ScalarTypeToCPPType(scalar_type_, [&]<typename T>(darkside::CPPTypeToScalarType<T>) {
    uint8_t *pointer = static_cast<uint8_t *>(data_) + index * sizeof(T);
    element = Element(static_cast<T *>(static_cast<void *>(pointer)), device_);
  });

  return element;
}

const Element Storage::operator[](uint64_t index) const {
  if (index >= size_)
    return Element();

  Element element;

  darkside::ScalarTypeToCPPType(scalar_type_, [&]<typename T>(darkside::CPPTypeToScalarType<T>) {
    uint8_t *pointer = static_cast<uint8_t *>(data_) + index * sizeof(T);
    element = Element(static_cast<T *>(static_cast<void *>(pointer)), device_);
  });

  return element;
}

void *Storage::getData() { return data_; }

const void *Storage::getData() const { return data_; }
uint64_t Storage::getSize() const { return size_; }
ScalarType Storage::getScalarType() const { return scalar_type_; }
Device *Storage::getDevice() const { return device_; }

void Storage::fillData(const Element &value) {
  if (size_ == 0)
    return;

  darkside::ScalarTypeToCPPType(scalar_type_, [&]<typename T>(darkside::CPPTypeToScalarType<T>) {
    Element value_host;
    const Element *value_pointer;

    if (value.getDevice()->getDeviceType() == DeviceType::CPU && value.getScalarType() == scalar_type_)
      value_pointer = &value;
    else {
      value_host = element_cast<darkside::CPPTypeToScalarType<T>::getScalarType>(value, &AMD5625U);
      value_pointer = &value_host;
    }
    darkside::fillData<T>(static_cast<T *>(data_), size_, *value_pointer->getData<T>(), device_);
  });
}

void Storage::fillIncreasedData(const Element &start, const Element &step) {
  if (size_ == 0)
    return;

  darkside::ScalarTypeToCPPType(scalar_type_, [&]<typename T>(darkside::CPPTypeToScalarType<T>) {
    Element start_host;
    const Element *start_pointer;

    if (start.getDevice()->getDeviceType() == DeviceType::CPU && start.getScalarType() == scalar_type_)
      start_pointer = &start;
    else {
      start_host = element_cast<darkside::CPPTypeToScalarType<T>::getScalarType>(start, &AMD5625U);
      start_pointer = &start_host;
    }

    Element step_host;
    const Element *step_pointer;

    if (step.getDevice()->getDeviceType() == DeviceType::CPU && step.getScalarType() == scalar_type_)
      step_pointer = &step;
    else {
      step_host = element_cast<darkside::CPPTypeToScalarType<T>::getScalarType>(step, &AMD5625U);
      step_pointer = &step_host;
    }

    darkside::fillIncreasedData<T>(static_cast<T *>(data_), size_, *start_pointer->getData<T>(), *step_pointer->getData<T>(), device_);
  });
}

void Storage::fillDecreasedData(const Element &start, const Element &step) {
  if (size_ == 0)
    return;

  darkside::ScalarTypeToCPPType(scalar_type_, [&]<typename T>(darkside::CPPTypeToScalarType<T>) {
    Element start_host;
    const Element *start_pointer;

    if (start.getDevice()->getDeviceType() == DeviceType::CPU && start.getScalarType() == scalar_type_)
      start_pointer = &start;
    else {
      start_host = element_cast<darkside::CPPTypeToScalarType<T>::getScalarType>(start, &AMD5625U);
      start_pointer = &start_host;
    }

    Element step_host;
    const Element *step_pointer;

    if (step.getDevice()->getDeviceType() == DeviceType::CPU && step.getScalarType() == scalar_type_)
      step_pointer = &step;
    else {
      step_host = element_cast<darkside::CPPTypeToScalarType<T>::getScalarType>(step, &AMD5625U);
      step_pointer = &step_host;
    }

    darkside::fillDecreasedData<T>(static_cast<T *>(data_), size_, *start_pointer->getData<T>(), *step_pointer->getData<T>(), device_);
  });
}
} // namespace startorch
