/* convert_probe.m — verify h3_bf16_to_f16_inplace at production sizes.
 * Compiles h3_shaders.metal, fills a private buffer with known BF16 values,
 * runs the in-place conversion exactly as h3_gpu_dispatch_1d does, reads back,
 * and checks EVERY element against the CPU conversion.
 *
 * xcrun clang -fobjc-arc -O2 -framework Foundation -framework Metal \
 *   src/convert_probe.m -o bin/convert_probe
 * ./bin/convert_probe "6900" src/h3.c/h3_shaders.metal
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float bf16_to_f32(uint16_t bits) {
    uint32_t wide = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &wide, sizeof(value));
    return value;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *want = argc > 1 ? argv[1] : "6900";
        const char *shader_path = argc > 2 ? argv[2] : "src/h3.c/h3_shaders.metal";
        id<MTLDevice> device = nil;
        for (id<MTLDevice> candidate in MTLCopyAllDevices())
            if ([candidate.name rangeOfString:@(want)].location != NSNotFound)
                device = candidate;
        if (!device) { fprintf(stderr, "no device\n"); return 1; }
        printf("device %s  maxThreadsPerThreadgroup=%lu\n",
               device.name.UTF8String,
               (unsigned long)device.maxThreadsPerThreadgroup.width);
        NSError *error = nil;
        NSString *source = [NSString stringWithContentsOfFile:@(shader_path)
            encoding:NSUTF8StringEncoding error:&error];
        if (!source) { fprintf(stderr, "no shader source\n"); return 1; }
        id<MTLLibrary> library = [device newLibraryWithSource:source
            options:nil error:&error];
        if (!library) {
            fprintf(stderr, "compile: %s\n",
                    error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:
                [library newFunctionWithName:@"h3_bf16_to_f16_inplace"]
                error:&error];
        if (!pipeline) {
            fprintf(stderr, "pipeline: %s\n",
                    error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        /* fc1 is the largest streamed matrix: 28672 x 5376 */
        size_t sizes[] = {21504u * 5376u, 7168u * 5376u,
                          28672u * 5376u, 5376u * 14336u};
        for (size_t s = 0; s < 4; s++) {
            size_t elements = sizes[s];
            uint16_t *host = malloc(elements * 2);
            srand((int)s + 7);
            for (size_t i = 0; i < elements; i++) {
                float value = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;
                host[i] = f32_to_bf16(value);
            }
            id<MTLBuffer> staging = [device newBufferWithBytes:host
                length:elements * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> target = [device newBufferWithLength:elements * 2
                options:MTLResourceStorageModePrivate];
            id<MTLBuffer> counter = [device newBufferWithLength:4
                options:MTLResourceStorageModeShared];
            memset(counter.contents, 0, 4);
            id<MTLBuffer> readback = [device newBufferWithLength:elements * 2
                options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
            [blit copyFromBuffer:staging sourceOffset:0 toBuffer:target
                destinationOffset:0 size:elements * 2];
            [blit endEncoding];
            uint32_t count = (uint32_t)elements;
            id<MTLComputeCommandEncoder> encoder =
                [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:target offset:0 atIndex:0];
            [encoder setBuffer:counter offset:0 atIndex:1];
            [encoder setBytes:&count length:sizeof(count) atIndex:2];
            NSUInteger width = MIN((NSUInteger)256,
                                   pipeline.maxTotalThreadsPerThreadgroup);
            [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
            [encoder endEncoding];
            id<MTLBlitCommandEncoder> back = [command blitCommandEncoder];
            [back copyFromBuffer:target sourceOffset:0 toBuffer:readback
                destinationOffset:0 size:elements * 2];
            [back endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status == MTLCommandBufferStatusError) {
                printf("size %zu: COMMAND ERROR %s\n", elements,
                       command.error.localizedDescription.UTF8String);
                free(host);
                continue;
            }
            const uint16_t *result = readback.contents;
            size_t wrong = 0, first_wrong = SIZE_MAX;
            for (size_t i = 0; i < elements; i++) {
                float value = bf16_to_f32(host[i]);
                __fp16 expected = (__fp16)value;
                uint16_t expected_bits;
                memcpy(&expected_bits, &expected, 2);
                if (result[i] != expected_bits) {
                    wrong++;
                    if (first_wrong == SIZE_MAX) first_wrong = i;
                }
            }
            uint32_t saturated;
            memcpy(&saturated, counter.contents, 4);
            printf("size %9zu: wrong=%zu first_wrong=%zd saturated=%u %s\n",
                   elements, wrong, first_wrong == SIZE_MAX ? -1 : (ssize_t)first_wrong,
                   saturated, wrong ? "<-- BROKEN" : "ok");
            free(host);
        }
    }
    return 0;
}
