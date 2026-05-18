#pragma once

#include "startorch/common.hpp"
#include "startorch/device.hpp"

#include <cstdint>

namespace startorch {

class Element {
private:
  void *data_ = nullptr;
  Device *device_ = nullptr;
  ScalarType scalar_type_ = ScalarType::UNKNOWN_SCALAR;
  OwnerType owner_type_ = OwnerType::UNKNOWN_OWNER;

public:
  Element() = default;

  Element(float *data, Device *device);
  Element(double *data, Device *device);
  Element(int8_t *data, Device *device);
  Element(int16_t *data, Device *device);
  Element(int32_t *data, Device *device);
  Element(int64_t *data, Device *device);
  Element(uint8_t *data, Device *device);
  Element(uint16_t *data, Device *device);
  Element(uint32_t *data, Device *device);
  Element(uint64_t *data, Device *device);

  Element(float value, Device *device = &AMD5625U);
  Element(double value, Device *device = &AMD5625U);
  Element(int8_t value, Device *device = &AMD5625U);
  Element(int16_t value, Device *device = &AMD5625U);
  Element(int32_t value, Device *device = &AMD5625U);
  Element(int64_t value, Device *device = &AMD5625U);
  Element(uint8_t value, Device *device = &AMD5625U);
  Element(uint16_t value, Device *device = &AMD5625U);
  Element(uint32_t value, Device *device = &AMD5625U);
  Element(uint64_t value, Device *device = &AMD5625U);

  Element(const Element &other);
  Element(Element &&other) noexcept;

  ~Element();

  Element &operator=(const Element &other);
  Element &operator=(Element &&other) noexcept;

  template <typename T> T *getData() { return static_cast<T *>(data_); }
  template <typename T> const T *getData() const { return static_cast<T *>(data_); }

  void *getData();
  const void *getData() const;
  Device *getDevice() const;
  ScalarType getScalarType() const;
  OwnerType getOwnerType() const;
};

template <ScalarType S> Element element_cast(const Element &element, Device *device = &AMD5625U);

class Storage {
private:
  void *data_ = nullptr;
  uint64_t size_ = 0;
  ScalarType scalar_type_ = ScalarType::UNKNOWN_SCALAR;
  Device *device_ = nullptr;

public:
  Storage() = default;
  Storage(uint64_t size, ScalarType scalar_type, Device *device = &AMD5625U);

  Storage(const Storage &other);
  Storage(Storage &&other) noexcept;

  ~Storage();

  Storage &operator=(const Storage &other);
  Storage &operator=(Storage &&other) noexcept;

  Element operator[](uint64_t index);
  const Element operator[](uint64_t index) const;

  template <typename T> T *getData() { return static_cast<T *>(data_); };
  template <typename T> const T *getData() const { return static_cast<T *>(data_); };

  void *getData();
  Device *getDevice();

  const void *getData() const;
  uint64_t getSize() const;
  ScalarType getScalarType() const;
  const Device *getDevice() const;

  void fillData(const Element &value);
  void fillIncreasedData(const Element &start, const Element &step);
  void fillDecreasedData(const Element &start, const Element &step);
};
} // namespace startorch
