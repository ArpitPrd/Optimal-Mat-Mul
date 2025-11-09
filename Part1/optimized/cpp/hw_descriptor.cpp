#include <iostream>
#include <thread>
#include <sys/sysinfo.h>
#include <cpuid.h>

void print_system_info() {
    std::cout << "Cores available: " << std::thread::hardware_concurrency() << "\n";

    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(7, &eax, &ebx, &ecx, &edx)) {

        std::cout << eax << ebx << ecx << edx << std::endl;
        if (ebx & (1 << 16)) std::cout << "AVX-512 supported\n";
        else if (ecx & (1 << 5)) std::cout << "AVX2 supported\n";
        else std::cout << "SSE supported\n";
    }
}


int main() {
    print_system_info();
}