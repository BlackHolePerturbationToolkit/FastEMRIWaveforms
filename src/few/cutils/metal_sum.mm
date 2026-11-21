// Apple GPU implementation of the interpolated time-domain mode sum.

#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <string>

#include "metal_sum.hh"

namespace {

thread_local std::string last_error;

struct SumContext {
  __strong id<MTLDevice> device;
  __strong id<MTLCommandQueue> queue;
  __strong id<MTLComputePipelineState> pipeline;
};

struct SumParameters {
  std::uint32_t init_length;
  std::uint32_t output_length;
  std::uint32_t mode_count;
  std::uint32_t padding;
};

const char *metal_source = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct SumParameters {
  uint init_length;
  uint output_length;
  uint mode_count;
  uint padding;
};

struct DSComplex {
  float2 real;
  float2 imag;
};

inline float2 ds_normalize(float high, float low) {
  const float sum = high + low;
  return float2(sum, low - (sum - high));
}

inline float2 ds_add(float2 left, float2 right) {
  const float sum = left.x + right.x;
  const float virtual_right = sum - left.x;
  const float error = (left.x - (sum - virtual_right)) +
                      (right.x - virtual_right) + left.y + right.y;
  return ds_normalize(sum, error);
}

inline float2 ds_negate(float2 value) {
  return float2(-value.x, -value.y);
}

inline float2 ds_subtract(float2 left, float2 right) {
  return ds_add(left, ds_negate(right));
}

inline float2 ds_multiply(float2 left, float2 right) {
  const float product = left.x * right.x;
  float error = fma(left.x, right.x, -product);
  error = fma(left.x, right.y, error);
  error = fma(left.y, right.x, error);
  error = fma(left.y, right.y, error);
  return ds_normalize(product, error);
}

inline float2 ds_multiply_float(float2 value, float factor) {
  return ds_multiply(value, float2(factor, 0.0f));
}

inline DSComplex ds_complex_add(DSComplex left, DSComplex right) {
  return DSComplex{ds_add(left.real, right.real),
                   ds_add(left.imag, right.imag)};
}

inline DSComplex ds_complex_conjugate(DSComplex value) {
  return DSComplex{value.real, ds_negate(value.imag)};
}

inline DSComplex ds_complex_multiply(DSComplex left, DSComplex right) {
  return DSComplex{
      ds_subtract(ds_multiply(left.real, right.real),
                  ds_multiply(left.imag, right.imag)),
      ds_add(ds_multiply(left.real, right.imag),
             ds_multiply(left.imag, right.real))};
}

inline float2 ds_polynomial_even(float2 x2, thread const float2 *coefficients,
                                 int highest_index) {
  float2 result = coefficients[highest_index];
  for (int index = highest_index - 1; index >= 0; --index) {
    result = ds_add(ds_multiply(result, x2), coefficients[index]);
  }
  return result;
}

inline void ds_sincos(float2 angle, thread float2 &sine,
                      thread float2 &cosine) {
  constexpr float inverse_two_pi = 0.15915494309189535f;
  constexpr float2 two_pi =
      float2(6.283185482025146484375f, -1.7484555314695172e-7f);
  constexpr float2 pi =
      float2(3.1415927410125732421875f, -8.7422776573475858e-8f);
  constexpr float2 half_pi =
      float2(1.57079637050628662109375f, -4.3711388286737929e-8f);

  // 2026-09-01 23:16 CST (mac): The host reduces each fundamental phase, so
  // the combined mode angle is small. The quotient is therefore an exact
  // integer in FP32 for every FEW Kerr mode index currently supported.
  const float quotient = rint((angle.x + angle.y) * inverse_two_pi);
  float2 reduced =
      ds_subtract(angle, ds_multiply_float(two_pi, quotient));
  if (reduced.x > pi.x) {
    reduced = ds_subtract(reduced, two_pi);
  } else if (reduced.x < -pi.x) {
    reduced = ds_add(reduced, two_pi);
  }

  float cosine_sign = 1.0f;
  if (reduced.x > half_pi.x) {
    reduced = ds_subtract(pi, reduced);
    cosine_sign = -1.0f;
  } else if (reduced.x < -half_pi.x) {
    reduced = ds_subtract(ds_negate(pi), reduced);
    cosine_sign = -1.0f;
  }

  // Taylor coefficients are split from their FP64 values. On
  // [-pi/2, pi/2], the first omitted terms are below 6e-13 for cosine and
  // 5e-14 for sine; including x^18 below moves both below 5e-14.
  const float2 sine_coefficients[9] = {
      float2(1.0f, 0.0f),
      float2(-0.1666666716337204f, 4.9670538793122887e-9f),
      float2(0.008333333767950535f, -4.3461720333759502e-10f),
      float2(-0.00019841270113829523f, 2.7255968749334558e-12f),
      float2(2.7557318844628753e-6f, 3.7935712242972291e-14f),
      float2(-2.5052107943679403e-8f, -4.4176230446483665e-16f),
      float2(1.6059044372074283e-10f, -5.3525265115627256e-18f),
      float2(-7.6471636098127127e-13f, -1.2200710471178288e-20f),
      float2(2.8114573589663704e-15f, -1.0462084739763658e-22f)};
  const float2 cosine_coefficients[10] = {
      float2(1.0f, 0.0f),
      float2(-0.5f, 0.0f),
      float2(0.0416666679084301f, -1.2417634698280722e-9f),
      float2(-0.0013888889225199819f, 3.3631094437103215e-11f),
      float2(2.4801587642286904e-5f, -3.4069960936668198e-13f),
      float2(-2.755731998149713e-7f, 7.5751122090511949e-15f),
      float2(2.0876755879584152e-9f, 1.1082839809204342e-16f),
      float2(-1.147074536050896e-11f, -2.3722076892312381e-19f),
      float2(4.7794772561329454e-14f, 7.6254440444864298e-22f),
      float2(-1.5619206814541513e-16f, -1.5404471465941993e-24f)};

  const float2 x2 = ds_multiply(reduced, reduced);
  sine = ds_multiply(
      reduced, ds_polynomial_even(x2, sine_coefficients, 8));
  cosine = ds_multiply_float(
      ds_polynomial_even(x2, cosine_coefficients, 9), cosine_sign);
}

inline float2 load_ds(const device float *high, const device float *low,
                      uint index) {
  return float2(high[index], low[index]);
}

inline DSComplex load_ds_complex(const device float2 *high,
                                 const device float2 *low, uint index) {
  return DSComplex{float2(high[index].x, low[index].x),
                   float2(high[index].y, low[index].y)};
}

inline float2 reduced_mode_phase(float4 high, float4 low,
                                 int m, int k, int n) {
  float2 phase =
      ds_multiply_float(float2(high.x, low.x), float(m));
  phase = ds_add(
      phase, ds_multiply_float(float2(high.y, low.y), float(k)));
  return ds_add(
      phase, ds_multiply_float(float2(high.z, low.z), float(n)));
}

kernel void few_mode_sum_double_single(
    device float4 *waveform [[buffer(0)]],
    const device float *interpolation_high [[buffer(1)]],
    const device float *interpolation_low [[buffer(2)]],
    const device int *m_values [[buffer(3)]],
    const device int *k_values [[buffer(4)]],
    const device int *n_values [[buffer(5)]],
    const device float2 *ylm_high [[buffer(6)]],
    const device float2 *ylm_low [[buffer(7)]],
    const device uint *segments [[buffer(8)]],
    const device float *local_time_high [[buffer(9)]],
    const device float *local_time_low [[buffer(10)]],
    const device float4 *phase_high [[buffer(11)]],
    const device float4 *phase_low [[buffer(12)]],
    constant SumParameters &parameters [[buffer(13)]],
    uint sample [[thread_position_in_grid]]) {
  if (sample >= parameters.output_length) {
    return;
  }

  const uint segment = segments[sample];
  const float2 x =
      float2(local_time_high[sample], local_time_low[sample]);
  const uint mode_count = parameters.mode_count;
  const uint base_count = parameters.init_length * 2 * mode_count;
  const uint segment_offset = segment * 2 * mode_count;
  DSComplex total{float2(0.0f), float2(0.0f)};

  for (uint mode = 0; mode < mode_count; ++mode) {
    const uint real_index = segment_offset + mode;
    const uint imag_index = real_index + mode_count;
    float2 amplitude_real =
        load_ds(interpolation_high, interpolation_low,
                3 * base_count + real_index);
    amplitude_real = ds_add(
        ds_multiply(amplitude_real, x),
        load_ds(interpolation_high, interpolation_low,
                2 * base_count + real_index));
    amplitude_real = ds_add(
        ds_multiply(amplitude_real, x),
        load_ds(interpolation_high, interpolation_low,
                base_count + real_index));
    amplitude_real = ds_add(
        ds_multiply(amplitude_real, x),
        load_ds(interpolation_high, interpolation_low, real_index));

    float2 amplitude_imag =
        load_ds(interpolation_high, interpolation_low,
                3 * base_count + imag_index);
    amplitude_imag = ds_add(
        ds_multiply(amplitude_imag, x),
        load_ds(interpolation_high, interpolation_low,
                2 * base_count + imag_index));
    amplitude_imag = ds_add(
        ds_multiply(amplitude_imag, x),
        load_ds(interpolation_high, interpolation_low,
                base_count + imag_index));
    amplitude_imag = ds_add(
        ds_multiply(amplitude_imag, x),
        load_ds(interpolation_high, interpolation_low, imag_index));

    const float2 phase = reduced_mode_phase(
        phase_high[sample], phase_low[sample], m_values[mode], k_values[mode],
        n_values[mode]);
    float2 sine;
    float2 cosine;
    ds_sincos(phase, sine, cosine);
    const DSComplex exponential{cosine, ds_negate(sine)};
    const DSComplex amplitude{amplitude_real, amplitude_imag};
    const DSComplex partial = ds_complex_multiply(exponential, amplitude);
    total = ds_complex_add(
        total, ds_complex_multiply(partial,
                                   load_ds_complex(ylm_high, ylm_low, mode)));
    if (m_values[mode] != 0) {
      total = ds_complex_add(
          total,
          ds_complex_multiply(
              ds_complex_conjugate(partial),
              load_ds_complex(ylm_high, ylm_low, mode_count + mode)));
    }
  }
  waveform[sample] =
      float4(total.real.x, total.real.y, total.imag.x, total.imag.y);
}
)METAL";

void set_error(const std::string &message) { last_error = message; }

void set_error(NSString *prefix, NSError *error) {
  NSString *description = error ? error.localizedDescription : @"unknown error";
  set_error(std::string(prefix.UTF8String) + ": " + description.UTF8String);
}

id<MTLBuffer> make_buffer(id<MTLDevice> device, std::size_t bytes) {
  bytes = std::max<std::size_t>(bytes, sizeof(float));
  if (bytes > device.maxBufferLength) {
    set_error("Metal summation buffer exceeds the device limit");
    return nil;
  }
  id<MTLBuffer> buffer = [device
      newBufferWithLength:static_cast<NSUInteger>(bytes)
                  options:MTLResourceStorageModeShared];
  if (!buffer) {
    set_error("Metal failed to allocate a strict summation buffer");
  }
  return buffer;
}

void split_to_float(const double *source, id<MTLBuffer> high_buffer,
                    id<MTLBuffer> low_buffer, std::size_t count) {
  auto *high = static_cast<float *>(high_buffer.contents);
  auto *low = static_cast<float *>(low_buffer.contents);
  vDSP_vdpsp(source, 1, high, 1, static_cast<vDSP_Length>(count));
  for (std::size_t index = 0; index < count; ++index) {
    low[index] = static_cast<float>(source[index] -
                                    static_cast<double>(high[index]));
  }
}

// Keep a large additive knot phase separate from its within-segment change.
double wrapped_phase(double angle) {
  if (std::abs(angle) <= M_PI) return angle;
  // hi + lo approximates 2*pi beyond binary64. FMA preserves the residual
  // when the quotient is moderate. libm handles very large arguments.
  if (std::abs(angle) < 0x1p40) {
    constexpr double hi = 2.0 * M_PI;
    constexpr double lo = 2.4492935982947064e-16;
    const double cycles = std::nearbyint(angle / hi);
    return std::fma(-cycles, lo, std::fma(-cycles, hi, angle));
  }
  return std::atan2(std::sin(angle), std::cos(angle));
}

double phase_increment(const double *coefficients, std::size_t segment_count,
                       std::size_t index, double s) {
  const double s1 = 1.0 - s;
  const double c1 = coefficients[segment_count + index];
  const double c2 = coefficients[2 * segment_count + index];
  const double c3 = coefficients[3 * segment_count + index];
  const double c4 = coefficients[4 * segment_count + index];
  const double c5 = coefficients[5 * segment_count + index];
  const double c6 = coefficients[6 * segment_count + index];
  const double c7 = coefficients[7 * segment_count + index];
  return s * (c1 + s1 * (c2 + s * (c3 + s1 * (c4 + s * (c5 + s1 *
         (c6 + s * c7))))));
}

void split_scalar(double value, float *high, float *low) {
  *high = static_cast<float>(value);
  *low = static_cast<float>(value - static_cast<double>(*high));
}

bool prepare_samples(
    const double *phase_times, const double *phase_coefficients,
    int init_length, int output_length, double delta_t,
    const double *trajectory_times, id<MTLBuffer> segments_buffer,
    id<MTLBuffer> local_time_high_buffer,
    id<MTLBuffer> local_time_low_buffer, id<MTLBuffer> phase_high_buffer,
    id<MTLBuffer> phase_low_buffer) {
  auto *segments = static_cast<std::uint32_t *>(segments_buffer.contents);
  auto *local_time_high =
      static_cast<float *>(local_time_high_buffer.contents);
  auto *local_time_low =
      static_cast<float *>(local_time_low_buffer.contents);
  auto *phase_high = static_cast<float *>(phase_high_buffer.contents);
  auto *phase_low = static_cast<float *>(phase_low_buffer.contents);
  const std::size_t phase_segment_count = 3ULL * (init_length - 1);
  for (int i = 0; i < init_length; ++i) {
    if (!std::isfinite(phase_times[i]) || !std::isfinite(trajectory_times[i]) ||
        (i && (phase_times[i] <= phase_times[i - 1] ||
               trajectory_times[i] <= trajectory_times[i - 1]))) {
      set_error("Metal summation requires finite, increasing time knots");
      return false;
    }
  }
  const double end_time = (output_length - 1) * delta_t;
  const double last_knot = trajectory_times[init_length - 1];
  if (!std::isfinite(end_time) ||
      end_time > std::nextafter(last_knot, std::numeric_limits<double>::infinity()) ||
      trajectory_times[0] > 0.0) {
    set_error("Metal output samples extend outside the trajectory knots");
    return false;
  }
  int previous_segment = -1;
  double offset[3];

  int segment = 0;
  for (int sample = 0; sample < output_length; ++sample) {
    while (segment + 1 < init_length - 1 &&
           trajectory_times[segment + 1] < end_time &&
           sample >= std::ceil(trajectory_times[segment + 1] / delta_t)) {
      ++segment;
    }
    if (segment != previous_segment) {
      for (int phase = 0; phase < 3; ++phase) {
        offset[phase] = wrapped_phase(phase_coefficients[3ULL * segment + phase]);
      }
      previous_segment = segment;
    }
    const double time = delta_t * sample;
    segments[sample] = static_cast<std::uint32_t>(segment);
    split_scalar(time - trajectory_times[segment],
                 local_time_high + sample, local_time_low + sample);
    const double width = phase_times[segment + 1] - phase_times[segment];
    const double s = (time - phase_times[segment]) / width;
    const std::size_t phase_index = static_cast<std::size_t>(sample) * 4;
    for (int phase = 0; phase < 3; ++phase) {
      const double step = phase_increment(phase_coefficients, phase_segment_count,
                                          3ULL * segment + phase, s);
      const double value = offset[phase] + wrapped_phase(step);
      split_scalar(value, phase_high + phase_index + phase,
                   phase_low + phase_index + phase);
    }
    phase_high[phase_index + 3] = 0.0f;
    phase_low[phase_index + 3] = 0.0f;
  }
  return true;
}

} // namespace

extern "C" {

const char *few_metal_sum_last_error() { return last_error.c_str(); }

void *few_metal_sum_context_create() {
  @autoreleasepool {
    last_error.clear();
    auto *context = new (std::nothrow) SumContext();
    if (!context) {
      set_error("Failed to allocate strict Metal summation context");
      return nullptr;
    }
    context->device = MTLCreateSystemDefaultDevice();
    context->queue = [context->device newCommandQueue];
    if (!context->device || !context->queue) {
      set_error("Metal device or strict summation queue is unavailable");
      delete context;
      return nullptr;
    }
    MTLCompileOptions *options = [MTLCompileOptions new];
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *)) {
      options.mathMode = MTLMathModeSafe;
      options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    } else
#endif
    {
      options.fastMathEnabled = NO;
    }
    NSError *error = nil;
    id<MTLLibrary> library = [context->device
        newLibraryWithSource:[NSString stringWithUTF8String:metal_source]
                     options:options
                       error:&error];
    if (!library) {
      set_error(@"Metal strict summation runtime compilation failed", error);
      delete context;
      return nullptr;
    }
    id<MTLFunction> function =
        [library newFunctionWithName:@"few_mode_sum_double_single"];
    if (!function) {
      set_error("Metal summation function was not found in the compiled library");
      delete context;
      return nullptr;
    }
    context->pipeline =
        [context->device newComputePipelineStateWithFunction:function
                                                       error:&error];
    if (!context->pipeline) {
      set_error(@"Metal strict summation pipeline creation failed", error);
      delete context;
      return nullptr;
    }
    return context;
  }
}

void few_metal_sum_context_destroy(void *opaque_context) {
  @autoreleasepool { delete static_cast<SumContext *>(opaque_context); }
}

int few_metal_sum_evaluate(
    void *opaque_context, double *waveform, const double *interpolation,
    const double *phase_times, const double *phase_coefficients,
    const int *m_values, const int *k_values, const int *n_values,
    int init_length, int output_length, int mode_count, const double *ylms,
    double delta_t, const double *trajectory_times, double *gpu_seconds) {
  @autoreleasepool {
    last_error.clear();
    if (!opaque_context || !waveform || !interpolation || !phase_times ||
        !phase_coefficients || !m_values || !k_values || !n_values || !ylms ||
        !trajectory_times || init_length < 2 || output_length < 1 ||
        mode_count < 1) {
      set_error("Invalid strict Metal summation arguments");
      return -1;
    }
    auto *context = static_cast<SumContext *>(opaque_context);
    const std::size_t interpolation_count =
        4ULL * init_length * 2ULL * mode_count;
    const std::size_t ylm_float_count = 4ULL * mode_count;
    id<MTLBuffer> output_buffer =
        make_buffer(context->device, output_length * 4ULL * sizeof(float));
    id<MTLBuffer> interpolation_high_buffer =
        make_buffer(context->device, interpolation_count * sizeof(float));
    id<MTLBuffer> interpolation_low_buffer =
        make_buffer(context->device, interpolation_count * sizeof(float));
    id<MTLBuffer> m_buffer =
        make_buffer(context->device, mode_count * sizeof(int));
    id<MTLBuffer> k_buffer =
        make_buffer(context->device, mode_count * sizeof(int));
    id<MTLBuffer> n_buffer =
        make_buffer(context->device, mode_count * sizeof(int));
    id<MTLBuffer> ylm_high_buffer =
        make_buffer(context->device, ylm_float_count * sizeof(float));
    id<MTLBuffer> ylm_low_buffer =
        make_buffer(context->device, ylm_float_count * sizeof(float));
    id<MTLBuffer> segments_buffer = make_buffer(
        context->device, output_length * sizeof(std::uint32_t));
    id<MTLBuffer> local_time_high_buffer =
        make_buffer(context->device, output_length * sizeof(float));
    id<MTLBuffer> local_time_low_buffer =
        make_buffer(context->device, output_length * sizeof(float));
    id<MTLBuffer> phase_high_buffer =
        make_buffer(context->device, output_length * 4ULL * sizeof(float));
    id<MTLBuffer> phase_low_buffer =
        make_buffer(context->device, output_length * 4ULL * sizeof(float));
    if (!output_buffer || !interpolation_high_buffer ||
        !interpolation_low_buffer || !m_buffer || !k_buffer || !n_buffer ||
        !ylm_high_buffer || !ylm_low_buffer || !segments_buffer ||
        !local_time_high_buffer || !local_time_low_buffer ||
        !phase_high_buffer || !phase_low_buffer) {
      return -1;
    }

    split_to_float(interpolation, interpolation_high_buffer,
                   interpolation_low_buffer, interpolation_count);
    split_to_float(ylms, ylm_high_buffer, ylm_low_buffer, ylm_float_count);
    std::memcpy(m_buffer.contents, m_values, mode_count * sizeof(int));
    std::memcpy(k_buffer.contents, k_values, mode_count * sizeof(int));
    std::memcpy(n_buffer.contents, n_values, mode_count * sizeof(int));
    if (!prepare_samples(
            phase_times, phase_coefficients, init_length, output_length,
            delta_t, trajectory_times, segments_buffer, local_time_high_buffer,
            local_time_low_buffer, phase_high_buffer, phase_low_buffer)) {
      return -1;
    }

    SumParameters parameters{static_cast<std::uint32_t>(init_length),
                             static_cast<std::uint32_t>(output_length),
                             static_cast<std::uint32_t>(mode_count), 0};
    id<MTLCommandBuffer> command_buffer = [context->queue commandBuffer];
    // 2026-09-02 14:08 CST (mac): Turn rare command-allocation failures into
    // Python exceptions instead of silently returning an all-zero waveform.
    if (!command_buffer) {
      set_error("Metal failed to allocate a strict summation command buffer");
      return -1;
    }
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    if (!encoder) {
      set_error("Metal failed to allocate a strict summation command encoder");
      return -1;
    }
    [encoder setComputePipelineState:context->pipeline];
    [encoder setBuffer:output_buffer offset:0 atIndex:0];
    [encoder setBuffer:interpolation_high_buffer offset:0 atIndex:1];
    [encoder setBuffer:interpolation_low_buffer offset:0 atIndex:2];
    [encoder setBuffer:m_buffer offset:0 atIndex:3];
    [encoder setBuffer:k_buffer offset:0 atIndex:4];
    [encoder setBuffer:n_buffer offset:0 atIndex:5];
    [encoder setBuffer:ylm_high_buffer offset:0 atIndex:6];
    [encoder setBuffer:ylm_low_buffer offset:0 atIndex:7];
    [encoder setBuffer:segments_buffer offset:0 atIndex:8];
    [encoder setBuffer:local_time_high_buffer offset:0 atIndex:9];
    [encoder setBuffer:local_time_low_buffer offset:0 atIndex:10];
    [encoder setBuffer:phase_high_buffer offset:0 atIndex:11];
    [encoder setBuffer:phase_low_buffer offset:0 atIndex:12];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:13];
    const NSUInteger width = context->pipeline.threadExecutionWidth;
    const NSUInteger maximum = context->pipeline.maxTotalThreadsPerThreadgroup;
    const NSUInteger desired = std::min<NSUInteger>(256, maximum);
    const NSUInteger threads = std::max<NSUInteger>(
        width, desired - desired % std::max<NSUInteger>(width, 1));
    [encoder dispatchThreads:MTLSizeMake(output_length, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status == MTLCommandBufferStatusError) {
      set_error(@"Metal strict summation command failed", command_buffer.error);
      return -1;
    }
    const double start_time = command_buffer.GPUStartTime;
    const double end_time = command_buffer.GPUEndTime;
    if (gpu_seconds) {
      *gpu_seconds = end_time >= start_time && start_time > 0.0
                         ? end_time - start_time
                         : 0.0;
    }
    const auto *parts = static_cast<const float *>(output_buffer.contents);
    for (std::size_t sample = 0; sample < static_cast<std::size_t>(output_length); ++sample) {
      waveform[2 * sample] += static_cast<double>(parts[4 * sample]) +
                              static_cast<double>(parts[4 * sample + 1]);
      waveform[2 * sample + 1] += static_cast<double>(parts[4 * sample + 2]) +
                                  static_cast<double>(parts[4 * sample + 3]);
    }
    return 0;
  }
}

} // extern "C"
