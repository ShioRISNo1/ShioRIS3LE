# GPU Acceleration on macOS

## Current Status

### OpenCL on macOS

**⚠️ OpenCL is DEPRECATED on macOS since macOS 10.14 (Mojave)**

Apple has deprecated OpenCL in favor of Metal. While OpenCL framework is still present in macOS for backward compatibility, it has significant limitations:

#### Limitations on macOS:

1. **No GPU Support on Apple Silicon (M1/M2/M3)**
   - OpenCL on Apple Silicon Macs only exposes CPU devices
   - GPU devices are NOT enumerated through OpenCL API
   - This is by design - Apple wants developers to use Metal

2. **Limited GPU Support on Intel Macs**
   - May work on older Intel-based Macs with AMD/Intel GPUs
   - However, performance is not optimized
   - No updates or improvements from Apple

3. **Deprecation Warnings**
   - Xcode shows deprecation warnings when using OpenCL
   - May be removed entirely in future macOS versions

### Current Behavior

When you run ShioRIS3 on macOS with OpenCL enabled:

```
=== GPU Dose Calculation Configuration ===
✓ OpenCL found:
  OpenCL Include: /System/Library/Frameworks/OpenCL.framework
  OpenCL Library: /System/Library/Frameworks/OpenCL.framework
✓ GPU dose calculation enabled with OpenCL backend
```

**BUT**, when the application initializes:

```
OpenCL: Platform detection started
OpenCL: Found 1 platform(s)
OpenCL: Platform 0: Apple Apple
OpenCL: Platform 0 has 0 GPU device(s) (error: Device not found)
OpenCL: No GPU devices found, trying CPU fallback
OpenCL: Platform 0 has 1 CPU device(s) (error: Success)
  OpenCL: CPU Device 0: Apple M2 (or Intel CPU)
OpenCL: ⚠ Selected CPU device (no GPU available)
```

**Result**: OpenCL falls back to CPU, providing minimal or no speedup.

## Recommended Solution for macOS: Metal Backend

### Why Metal?

1. **Native macOS GPU API** - Officially supported and maintained by Apple
2. **Full GPU Access** - Works on both Intel and Apple Silicon
3. **Better Performance** - Optimized for macOS hardware
4. **Unified Memory** - Efficient memory sharing on Apple Silicon
5. **Future-Proof** - Active development and support from Apple

### Metal Backend Status

| Platform | Status | Performance |
|----------|--------|-------------|
| **macOS (Intel)** | 🚧 Not Yet Implemented | Expected: 20-50x speedup |
| **macOS (Apple Silicon)** | 🚧 Not Yet Implemented | Expected: 50-100x speedup |

## Workaround: Use Linux or Windows for GPU

If you need GPU acceleration NOW:

1. **Linux** + NVIDIA GPU: OpenCL works perfectly
2. **Linux** + AMD GPU: OpenCL works well
3. **Windows** + NVIDIA GPU: OpenCL works perfectly
4. **Windows** + AMD GPU: OpenCL works well

## Implementation Roadmap

### Phase 1: Metal Backend (High Priority for macOS)

```cpp
// Proposed implementation
class MetalDoseBackend : public IGPUDoseBackend {
    id<MTLDevice> device_;
    id<MTLCommandQueue> commandQueue_;
    id<MTLComputePipelineState> doseKernel_;
    id<MTLBuffer> ctBuffer_;
    id<MTLBuffer> doseBuffer_;
    // ...
};
```

### Phase 2: Fallback Strategy

When GPU not available:
1. Detect platform (macOS/Windows/Linux)
2. Try best backend (Metal on macOS, OpenCL on others)
3. Fall back to CPU if GPU unavailable
4. Display clear message to user

## How to Check OpenCL Devices on Your Mac

Run this command in Terminal:

```bash
# List all OpenCL platforms and devices
/System/Library/Frameworks/OpenCL.framework/Libraries/openclinfo
```

**On Apple Silicon Macs (M1/M2/M3)**, you'll typically see:

```
Platform Name: Apple
Number of devices: 1

Device 0:
  Device Type: CPU
  Device Name: Apple M2
  Device Vendor: Apple
  Max Compute Units: 8
```

**Notice**: No GPU device listed!

**On Intel Macs**, you might see:

```
Platform Name: Apple
Number of devices: 2

Device 0:
  Device Type: GPU
  Device Name: AMD Radeon Pro 5500M
  Device Vendor: AMD

Device 1:
  Device Type: CPU
  Device Name: Intel(R) Core(TM) i9
```

## Temporary Workaround

Until Metal backend is implemented, on macOS:

1. **Use CPU mode** - Current OpenCL backend falls back to CPU
2. **Run on Linux VM** - Use virtualization with GPU passthrough
3. **Use Windows Boot Camp** - For Intel Macs only
4. **Wait for Metal implementation** - Coming soon!

## Developer Notes

### Building with OpenCL on macOS

The OpenCL framework is detected by CMake, but:

```cmake
# CMake Output
-- ✓ OpenCL found:
-- ✓ GPU dose calculation enabled with OpenCL backend
```

This is **misleading** because:
- OpenCL framework exists ✓
- OpenCL GPU devices do NOT exist ✗

### Testing OpenCL on macOS

To see detailed device detection logs, run:

```bash
./ShioRIS3 2>&1 | grep OpenCL
```

You should see output like:
```
OpenCL: Platform detection started
OpenCL: Found 1 platform(s)
OpenCL: Platform 0: Apple Apple
OpenCL: Platform 0 has 0 GPU device(s) (error: Device not found)
OpenCL: ⚠ Selected CPU device (no GPU available)
OpenCL: Note - macOS has deprecated OpenCL for GPU. Consider using Metal backend for GPU acceleration.
```

## References

- [Apple Metal Documentation](https://developer.apple.com/metal/)
- [OpenCL Deprecation Notice](https://developer.apple.com/opencl/)
- [Metal Compute Programming Guide](https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/)

## Summary

| Feature | OpenCL (macOS) | Metal (macOS) | OpenCL (Linux/Win) |
|---------|----------------|---------------|-------------------|
| **GPU Support** | ❌ (Apple Silicon)<br>⚠️ (Intel) | ✅ | ✅ |
| **Performance** | 1x (CPU fallback) | 50-100x | 20-50x |
| **Status** | Deprecated | Recommended | Supported |
| **Implementation** | ✅ Complete | 🚧 Planned | ✅ Complete |

**Recommendation for macOS users**:
- Wait for Metal backend implementation (coming soon)
- Or use Linux/Windows for GPU acceleration now
