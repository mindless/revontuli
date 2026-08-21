/* metal_probe.m
 *
 * I use this to get a definitive answer about which GPUs macOS actually
 * exposes to Metal on this 2018 Intel Mac mini, and what each one can do.
 * I deliberately do not trust System Information for this: I want the same
 * MTLDevice objects that h3.c itself would receive.
 *
 * Build:
 *   xcrun clang -fobjc-arc -framework Foundation -framework Metal \
 *     src/metal_probe.m -o bin/metal_probe
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static const char *yesno(BOOL v) { return v ? "yes" : "no"; }

static double gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

int main(void) {
    @autoreleasepool {
        /* I enumerate every Metal device, not just the system default one.
         * On an Intel Mac with an eGPU there is more than one. */
        NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
        id<MTLDevice> systemDefault = MTLCreateSystemDefaultDevice();

        printf("I found %lu Metal device(s).\n", (unsigned long)devices.count);
        printf("MTLCreateSystemDefaultDevice() returned: %s\n\n",
               systemDefault ? systemDefault.name.UTF8String : "(nil)");

        NSUInteger index = 0;
        for (id<MTLDevice> device in devices) {
            printf("==================== Metal device %lu ====================\n",
                   (unsigned long)index);
            printf("  name                        : %s\n", device.name.UTF8String);
            printf("  registryID                  : 0x%llx (%llu)\n",
                   (unsigned long long)device.registryID,
                   (unsigned long long)device.registryID);
            printf("  is system default           : %s\n",
                   yesno(systemDefault &&
                         systemDefault.registryID == device.registryID));
            printf("  lowPower                    : %s\n", yesno(device.isLowPower));
            printf("  removable (eGPU)            : %s\n", yesno(device.isRemovable));
            printf("  headless                    : %s\n", yesno(device.isHeadless));
            printf("  hasUnifiedMemory            : %s\n", yesno(device.hasUnifiedMemory));
            printf("  location                    : %ld (0=builtin 1=slot 2=external 3=unspecified)\n",
                   (long)device.location);
            printf("  locationNumber              : %lu\n",
                   (unsigned long)device.locationNumber);
            printf("  recommendedMaxWorkingSetSize: %llu bytes (%.2f GiB)\n",
                   (unsigned long long)device.recommendedMaxWorkingSetSize,
                   gib(device.recommendedMaxWorkingSetSize));
            printf("  maxBufferLength             : %llu bytes (%.2f GiB)\n",
                   (unsigned long long)device.maxBufferLength,
                   gib(device.maxBufferLength));
            printf("  currentAllocatedSize        : %llu bytes (%.2f GiB)\n",
                   (unsigned long long)device.currentAllocatedSize,
                   gib(device.currentAllocatedSize));
            printf("  maxThreadgroupMemoryLength  : %lu bytes\n",
                   (unsigned long)device.maxThreadgroupMemoryLength);
            printf("  maxThreadsPerThreadgroup    : %lu x %lu x %lu\n",
                   (unsigned long)device.maxThreadsPerThreadgroup.width,
                   (unsigned long)device.maxThreadsPerThreadgroup.height,
                   (unsigned long)device.maxThreadsPerThreadgroup.depth);
            printf("  maxTransferRate             : %llu bytes/s\n",
                   (unsigned long long)device.maxTransferRate);
            if (@available(macOS 14.0, *)) {
                printf("  architecture.name           : %s\n",
                       device.architecture.name.UTF8String);
            }
            printf("  supports32BitFloatFiltering : %s\n",
                   yesno(device.supports32BitFloatFiltering));
            printf("  supportsBCTextureCompression: %s\n",
                   yesno(device.supportsBCTextureCompression));

            /* h3_metal.m probes MTLGPUFamily 1001..1010 (the Apple families)
             * and reports the highest hit as "apple_gpu_family". I replicate
             * that here so I can see exactly what h3.c would conclude. */
            printf("  -- Apple GPU families (what h3_metal.m probes) --\n");
            int appleFamily = 0;
            for (int family = 10; family >= 1; family--) {
                MTLGPUFamily candidate = (MTLGPUFamily)(1000 + family);
                if ([device supportsFamily:candidate]) {
                    if (!appleFamily) appleFamily = family;
                    printf("     MTLGPUFamilyApple%-2d       : yes\n", family);
                }
            }
            printf("     -> h3 apple_gpu_family    : %d %s\n", appleFamily,
                   appleFamily ? "" : "(none — this is not an Apple GPU)");

            printf("  -- Common/Mac/Metal families --\n");
            struct { const char *label; MTLGPUFamily f; } fams[] = {
                { "MTLGPUFamilyCommon1", MTLGPUFamilyCommon1 },
                { "MTLGPUFamilyCommon2", MTLGPUFamilyCommon2 },
                { "MTLGPUFamilyCommon3", MTLGPUFamilyCommon3 },
                { "MTLGPUFamilyMac2",    MTLGPUFamilyMac2    },
                { "MTLGPUFamilyMetal3",  MTLGPUFamilyMetal3  },
            };
            for (size_t i = 0; i < sizeof(fams) / sizeof(fams[0]); i++) {
                printf("     %-24s : %s\n", fams[i].label,
                       yesno([device supportsFamily:fams[i].f]));
            }
            if (@available(macOS 26.0, *)) {
                printf("     %-24s : %s\n", "MTLGPUFamilyMetal4",
                       yesno([device supportsFamily:MTLGPUFamilyMetal4]));
            } else {
                printf("     %-24s : n/a (SDK/OS < macOS 26)\n",
                       "MTLGPUFamilyMetal4");
            }

            /* h3_gpu.m turns on the M5 TensorOps fast path purely by looking
             * for the substring "M5" in the device name. I check that here
             * because it decides whether the bfloat-heavy shader block is
             * even compiled. */
            BOOL m5 = [device.name rangeOfString:@"M5"].location != NSNotFound;
            printf("  -- h3.c decision points --\n");
            printf("     name contains \"M5\"       : %s -> H3_METAL_HAS_TENSOR %s\n",
                   yesno(m5), m5 ? "WOULD be defined" : "will NOT be defined");
            printf("     argument buffers tier    : %ld\n",
                   (long)device.argumentBuffersSupport);
            printf("\n");
            index++;
        }

        /* I make it easy for a shell script to assert that the eGPU is present. */
        BOOL found6900 = NO;
        for (id<MTLDevice> device in devices) {
            if ([device.name rangeOfString:@"6900"].location != NSNotFound) {
                found6900 = YES;
                printf("I detected the RX 6900 XT as Metal device: %s "
                       "(registryID 0x%llx)\n", device.name.UTF8String,
                       (unsigned long long)device.registryID);
            }
        }
        if (!found6900) {
            printf("I did NOT find any Metal device whose name contains \"6900\".\n");
            return 2;
        }
        return 0;
    }
}
