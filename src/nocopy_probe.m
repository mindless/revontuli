/* nocopy_probe.m
 *
 * h3.c loads weights by mmap'ing a region of a safetensors file and wrapping it
 * with -newBufferWithBytesNoCopy:. It computes the pointer as
 * (mapping + file_offset % pagesize), so the pointer it hands Metal is only
 * page-aligned when the tensor's file offset happens to be page-aligned.
 *
 * Metal documents that this API needs a page-aligned pointer. On Apple Silicon
 * the driver is forgiving. This is a DISCRETE eGPU over Thunderbolt, so I need
 * to know what actually happens here before I download ~100 GiB of weights and
 * discover the SSD-streaming path cannot map anything.
 *
 * I test, on the explicitly selected RX 6900 XT:
 *   1. BytesNoCopy with a page-ALIGNED pointer and page-multiple length
 *   2. BytesNoCopy with a page-aligned pointer and NON-multiple length
 *   3. BytesNoCopy with an UNALIGNED pointer (h3.c's real-world case)
 *   4. that the GPU actually reads the mapped bytes correctly
 *   5. a Shared allocation larger than maxBufferLength, to see the failure mode
 *
 * Build:
 *   xcrun clang -fobjc-arc -O2 -framework Foundation -framework Metal \
 *     src/nocopy_probe.m -o bin/nocopy_probe
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>

static int g_pass = 0, g_fail = 0;
static void result(const char *label, int ok, const char *detail) {
    printf("  [%s] %-52s %s\n", ok ? "PASS" : "FAIL", label, detail ? detail : "");
    if (ok) g_pass++; else g_fail++;
}

static NSString *const kSource = @"\
#include <metal_stdlib>\n\
using namespace metal;\n\
/* I sum the mapped bytes on the GPU so I can prove it really read them. */\n\
kernel void sum_u32(device const uint *src [[buffer(0)]],\n\
                    device atomic_uint *acc [[buffer(1)]],\n\
                    constant uint &count [[buffer(2)]],\n\
                    uint i [[thread_position_in_grid]]) {\n\
    if (i < count)\n\
        atomic_fetch_add_explicit(acc, src[i], memory_order_relaxed);\n\
}\n";

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *wanted = (argc > 1) ? argv[1] : "6900";
        id<MTLDevice> device = nil;
        for (id<MTLDevice> d in MTLCopyAllDevices())
            if ([d.name rangeOfString:[NSString stringWithUTF8String:wanted]].location
                != NSNotFound) { device = d; break; }
        if (!device) { printf("I could not find device \"%s\".\n", wanted); return 2; }

        size_t page = (size_t)getpagesize();
        printf("I am probing newBufferWithBytesNoCopy on: %s\n", device.name.UTF8String);
        printf("  page size       = %zu bytes\n", page);
        printf("  maxBufferLength = %.2f GiB\n\n",
               (double)device.maxBufferLength / 1073741824.0);

        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError *err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:kSource options:nil error:&err];
        if (!lib) { printf("compile failed: %s\n", err.localizedDescription.UTF8String); return 3; }
        id<MTLComputePipelineState> pipe =
            [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"sum_u32"]
                                                 error:&err];
        if (!pipe) { printf("pipeline failed: %s\n", err.localizedDescription.UTF8String); return 3; }

        /* I build a scratch file and mmap it, exactly like h3.c does. */
        const char *path = "/tmp/h3_nocopy_probe.bin";
        size_t fileBytes = 64 * page;
        FILE *fh = fopen(path, "wb");
        if (!fh) { printf("cannot write %s\n", path); return 4; }
        uint32_t *seed = malloc(fileBytes);
        for (size_t i = 0; i < fileBytes / 4; i++) seed[i] = (uint32_t)(i % 7) + 1;
        fwrite(seed, 1, fileBytes, fh);
        fclose(fh);

        int fd = open(path, O_RDONLY);
        void *mapping = mmap(NULL, fileBytes, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (mapping == MAP_FAILED) { printf("mmap failed\n"); return 5; }

        printf("=== newBufferWithBytesNoCopy behaviour ===\n");

        /* --- Case 1: aligned pointer, page-multiple length --- */
        {
            id<MTLBuffer> buf = [device newBufferWithBytesNoCopy:mapping
                                                         length:8 * page
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
            result("aligned pointer + page-multiple length", buf != nil,
                   buf ? "buffer created" : "returned nil");
        }

        /* --- Case 2: aligned pointer, length NOT a page multiple --- */
        {
            id<MTLBuffer> buf = nil;
            @try {
                buf = [device newBufferWithBytesNoCopy:mapping
                                               length:8 * page + 1234
                                              options:MTLResourceStorageModeShared
                                          deallocator:nil];
            } @catch (NSException *ex) {
                printf("      exception: %s\n", ex.reason.UTF8String);
            }
            result("aligned pointer + non-page-multiple length", buf != nil,
                   buf ? "buffer created (length rounded internally)"
                       : "returned nil / threw");
        }

        /* --- Case 3: UNALIGNED pointer -- this is h3.c's real case whenever a
         *     tensor's offset inside the safetensors file is not a multiple of
         *     the page size. --- */
        {
            size_t deltas[] = { 2, 64, 128, 512, 1024, 2048, 4095 };
            int aligned_required = 0, created = 0;
            for (size_t d = 0; d < sizeof(deltas) / sizeof(deltas[0]); d++) {
                void *p = (unsigned char *)mapping + deltas[d];
                id<MTLBuffer> buf = nil;
                @try {
                    buf = [device newBufferWithBytesNoCopy:p
                                                   length:4 * page
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
                } @catch (NSException *ex) {
                    /* I record it rather than dying, so I learn the whole shape. */
                }
                if (buf) created++; else aligned_required++;
                printf("      offset +%-5zu : %s\n", deltas[d],
                       buf ? "buffer created" : "NIL (page alignment enforced)");
            }
            char detail[192];
            snprintf(detail, sizeof(detail),
                     "%d/%zu unaligned offsets accepted", created,
                     sizeof(deltas) / sizeof(deltas[0]));
            /* Either answer is informative; I only fail if it is inconsistent. */
            result("unaligned pointer accepted by this driver",
                   created == (int)(sizeof(deltas) / sizeof(deltas[0])), detail);
        }

        /* --- Case 4: does the GPU actually read the mapped data correctly? --- */
        {
            size_t bytes = 8 * page;
            uint32_t count = (uint32_t)(bytes / 4);
            id<MTLBuffer> src = [device newBufferWithBytesNoCopy:mapping
                                                         length:bytes
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
            id<MTLBuffer> acc = [device newBufferWithLength:4
                                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> cnt = [device newBufferWithLength:4
                                                   options:MTLResourceStorageModeShared];
            if (src && acc && cnt) {
                *(uint32_t *)acc.contents = 0;
                *(uint32_t *)cnt.contents = count;
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pipe];
                [enc setBuffer:src offset:0 atIndex:0];
                [enc setBuffer:acc offset:0 atIndex:1];
                [enc setBuffer:cnt offset:0 atIndex:2];
                [enc dispatchThreads:MTLSizeMake(count, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                uint64_t ref = 0;
                for (uint32_t i = 0; i < count; i++) ref += seed[i];
                uint32_t got = *(uint32_t *)acc.contents;
                char detail[192];
                snprintf(detail, sizeof(detail),
                         "GPU sum=%u  CPU sum(mod 2^32)=%u  status=%ld",
                         got, (uint32_t)ref, (long)cb.status);
                result("GPU reads mmap'd file bytes correctly",
                       got == (uint32_t)ref, detail);
            } else {
                result("GPU reads mmap'd file bytes correctly", 0,
                       "buffer setup failed");
            }
        }

        /* --- Case 5: what happens above maxBufferLength --- */
        {
            uint64_t over = device.maxBufferLength + (1ull << 20);
            id<MTLBuffer> buf = nil;
            @try {
                buf = [device newBufferWithLength:over
                                         options:MTLResourceStorageModeShared];
            } @catch (NSException *ex) {
                printf("      exception: %s\n", ex.reason.UTF8String);
            }
            char detail[192];
            snprintf(detail, sizeof(detail),
                     "requested %.2f GiB -> %s (h3.c would report "
                     "\"cannot allocate ... Metal buffer\")",
                     (double)over / 1073741824.0, buf ? "SUCCEEDED" : "nil");
            /* I want this to be a clean nil, not a crash. */
            result("over-cap allocation fails cleanly (returns nil)",
                   buf == nil, detail);
        }

        munmap(mapping, fileBytes);
        free(seed);
        unlink(path);
        printf("\nI ran %d checks: %d passed, %d failed.\n",
               g_pass + g_fail, g_pass, g_fail);
        return g_fail ? 1 : 0;
    }
}
