#include "startorch/layout.hpp"
#include "startorch/common.hpp"
#include "startorch/device.hpp"
#include "startorch/format.hpp"
#include "startorch/memory.hpp"

#include "darkside/kernel.cuh"

#include <cstdint>

namespace startorch {
Layout::Layout(const Storage &shape, const Storage &order, const Storage &strides, const Storage &offsets) {
  uint64_t size = shape.getSize();

  if (size == 0)
    return;

  Device *device = shape.getDevice();

  if (size == order.getSize()) {
    if (device->getMemoryType() < order.getDevice()->getMemoryType())
      device = order.getDevice();
    if (size == strides.getSize()) {
      if (device->getMemoryType() < strides.getDevice()->getMemoryType())
        device = strides.getDevice();
      if (size == offsets.getSize()) {
        if (device->getMemoryType() < offsets.getDevice()->getMemoryType())
          device = offsets.getDevice();
      }
    }
  }

  if (device->getMemoryType() < MemoryType::HOST)
    device = &AMD5625U;

  ScalarType scalar_type = shape.getScalarType();

  if (size == order.getSize()) {
    if (scalar_type < order.getScalarType())
      scalar_type = order.getScalarType();
    if (size == strides.getSize()) {
      if (scalar_type < strides.getScalarType())
        scalar_type = strides.getScalarType();
      if (size == offsets.getSize()) {
        if (scalar_type < offsets.getScalarType())
          scalar_type = offsets.getScalarType();
      }
    }
  }

  if (scalar_type < ScalarType::INT_8)
    scalar_type = ScalarType::UNSIGNED_INT_64;

  if (device != shape.getDevice() || scalar_type != shape.getScalarType()) {
    shape_ = Storage(size, scalar_type, device);

    darkside::ScalarTypeToCPPType(shape.getScalarType(), [&]<typename N>(darkside::CPPTypeToScalarType<N>) {
      darkside::ScalarTypeToCPPType(scalar_type, [&]<typename T>(darkside::CPPTypeToScalarType<T>) {
        darkside::castData<T, N>(shape_.getData<T>(), shape.getData<N>(), size, device, shape.getDevice());
      });
    });
  } else
    shape_ = shape;

  bool valid = true;

  if (valid == false || size != order.getSize()) {
    valid = false;
    order_ = Storage(size, scalar_type, device);

    order_.fillIncreasedData(Element(0, &AMD5625U), Element(1, &AMD5625U));
  } else if (device != order.getDevice() || scalar_type != order.getScalarType()) {
    order_ = Storage(size, scalar_type, device);

    darkside::ScalarTypeToCPPType(order.getScalarType(), [&]<typename N>(darkside::CPPTypeToScalarType<N>) {
      darkside::ScalarTypeToCPPType(scalar_type, [&]<typename T>(darkside::CPPTypeToScalarType<T>) {
        darkside::castData<T, N>(order_.getData<T>(), order.getData<N>(), size, device, order.getDevice());
      });
    });
  } else
    order_ = order;

  if (valid == false || size != strides.getSize()) {
    valid = false;
    strides_ = Storage(size, scalar_type, device);

    darkside::ScalarTypeToCPPType(scalar_type, [&]<typename T>(darkside::CPPTypeToScalarType<T>) {
      T stride = 1;
      T *order_ptr = order_.getData<T>();
      T *shape_ptr = shape_.getData<T>();
      T *strides_ptr = strides_.getData<T>();

      for (uint64_t i = size; i > 0; i--) {
        uint64_t dim = static_cast<uint64_t>(order_ptr[i - 1]);
        strides_ptr[dim] = stride;
        stride *= shape_ptr[dim];
      }
    });
  } else if (device != strides.getDevice() || scalar_type != strides.getScalarType()) {
    strides_ = Storage(size, scalar_type, device);

    darkside::ScalarTypeToCPPType(strides.getScalarType(), [&]<typename N>(darkside::CPPTypeToScalarType<N>) {
      darkside::ScalarTypeToCPPType(scalar_type, [&]<typename T>(darkside::CPPTypeToScalarType<T>) {
        darkside::castData<T, N>(strides_.getData<T>(), strides.getData<N>(), size, device, strides.getDevice());
      });
    });
  } else
    strides_ = strides;

  if (valid == false || size != offsets.getSize()) {
    valid = false;
    offsets_ = Storage(size, scalar_type, device);

    offsets_.fillData(Element(0, &AMD5625U));
  } else if (device != offsets.getDevice() || scalar_type != offsets.getScalarType()) {
    offsets_ = Storage(size, scalar_type, device);

    darkside::ScalarTypeToCPPType(offsets.getScalarType(), [&]<typename N>(darkside::CPPTypeToScalarType<N>) {
      darkside::ScalarTypeToCPPType(scalar_type, [&]<typename T>(darkside::CPPTypeToScalarType<T>) {
        darkside::castData<T, N>(offsets_.getData<T>(), offsets.getData<N>(), size, device, offsets.getDevice());
      });
    });
  } else
    offsets_ = offsets;
}

Layout::Layout(const Storage &shape, OrderType order_type, const Storage &strides, const Storage &offsets) {
  uint64_t size = shape.getSize();

  if (size == 0)
    return;

  Device *device = shape.getDevice();

  if (size == strides.getSize()) {
    if (device->getMemoryType() < strides.getDevice()->getMemoryType())
      device = strides.getDevice();
    if (size == offsets.getSize()) {
      if (device->getMemoryType() < offsets.getDevice()->getMemoryType())
        device = offsets.getDevice();
    }
  }

  if (device->getMemoryType() < MemoryType::HOST)
    device = &AMD5625U;

  ScalarType scalar_type = shape.getScalarType();

  if (size == strides.getSize()) {
    if (scalar_type < strides.getScalarType())
      scalar_type = strides.getScalarType();
    if (size == offsets.getSize()) {
      if (scalar_type < offsets.getScalarType())
        scalar_type = offsets.getScalarType();
    }
  }

  if (scalar_type < ScalarType::INT_8)
    scalar_type = ScalarType::UNSIGNED_INT_64;

  if (device != shape.getDevice() || scalar_type != shape.getScalarType()) {
    shape_ = Storage(size, scalar_type, device);

    darkside::ScalarTypeToCPPType(shape.getScalarType(), [&]<typename N>(darkside::CPPTypeToScalarType<N>) {
      darkside::ScalarTypeToCPPType(scalar_type, [&]<typename T>(darkside::CPPTypeToScalarType<T>) {
        darkside::castData<T, N>(shape_.getData<T>(), shape.getData<N>(), size, device, shape.getDevice());
      });
    });
  } else
    shape_ = shape;

  order_ = Storage(size, scalar_type, device);

  switch (order_type) {
  case OrderType::COLUMN_MAJOR:
    order_.fillDecreasedData(Element(size - 1, &AMD5625U), Element(1, &AMD5625U));
    break;

  case OrderType::ROW_MAJOR:
    order_.fillIncreasedData(Element(0, &AMD5625U), Element(1, &AMD5625U));
    break;

  default:
    order_.fillIncreasedData(Element(0, &AMD5625U), Element(1, &AMD5625U));
    break;
  }

  bool valid = true;

  if (valid == false || size != strides.getSize()) {
    valid = false;
    strides_ = Storage(size, scalar_type, device);

    darkside::ScalarTypeToCPPType(scalar_type, [&]<typename T>(darkside::CPPTypeToScalarType<T>) {
      T stride = 1;
      T *strides_ptr = strides_.getData<T>();
      const T *order_ptr = order_.getData<T>();
      const T *shape_ptr = shape_.getData<T>();

      for (uint64_t i = size; i > 0; i--) {
        uint64_t dim = static_cast<uint64_t>(order_ptr[i - 1]);
        strides_ptr[dim] = stride;
        stride *= shape_ptr[dim];
      }
    });
  } else if (device != strides.getDevice() || scalar_type != strides.getScalarType()) {
    strides_ = Storage(size, scalar_type, device);

    darkside::ScalarTypeToCPPType(strides.getScalarType(), [&]<typename N>(darkside::CPPTypeToScalarType<N>) {
      darkside::ScalarTypeToCPPType(scalar_type, [&]<typename T>(darkside::CPPTypeToScalarType<T>) {
        darkside::castData<T, N>(strides_.getData<T>(), strides.getData<N>(), size, device, strides.getDevice());
      });
    });
  } else
    strides_ = strides;

  if (valid == false || size != offsets.getSize()) {
    valid = false;
    offsets_ = Storage(size, scalar_type, device);

    offsets_.fillData(Element(0, &AMD5625U));
  } else if (device != offsets.getDevice() || scalar_type != offsets.getScalarType()) {
    offsets_ = Storage(size, scalar_type, device);

    darkside::ScalarTypeToCPPType(offsets.getScalarType(), [&]<typename N>(darkside::CPPTypeToScalarType<N>) {
      darkside::ScalarTypeToCPPType(scalar_type, [&]<typename T>(darkside::CPPTypeToScalarType<T>) {
        darkside::castData<T, N>(offsets_.getData<T>(), offsets.getData<N>(), size, device, offsets.getDevice());
      });
    });
  } else
    offsets_ = offsets;
}

uint64_t Layout::getIndex(const Storage &indices) const {
  if (strides_.getSize() == 0 || strides_.getSize() != indices.getSize())
    return 0;

  const Storage *indices_ = nullptr;
  Storage temp = Storage();

  if (indices.getDevice()->getDeviceType() != DeviceType::CPU) {
    temp = Storage(indices.getSize(), indices.getScalarType(), strides_.getDevice());
    indices_ = &temp;

    DevicePair(strides_.getDevice(), indices.getDevice())
        .copyData(temp.getData(), indices.getData(), indices.getSize() * darkside::getScalarTypeSize(indices.getScalarType()));
  } else
    indices_ = &indices;

  uint64_t index = 0;

  darkside::ScalarTypeToCPPType(indices.getScalarType(), [&]<typename T>(darkside::CPPTypeToScalarType<T>) {
    darkside::ScalarTypeToCPPType(strides_.getScalarType(), [&]<typename N>(darkside::CPPTypeToScalarType<N>) {
      const T *indices_ptr = indices_->getData<T>();
      const N *offsets_ptr = offsets_.getData<N>();
      const N *strides_ptr = strides_.getData<N>();
      uint64_t size = shape_.getSize();

      for (uint64_t i = 0; i < size; i++)
        index += (static_cast<uint64_t>(indices_ptr[i]) + offsets_ptr[i]) * strides_ptr[i];
    });
  });

  return index;
}
} // namespace startorch
