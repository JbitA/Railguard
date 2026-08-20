#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
namespace railguard {
class PosixSerialSource {
public:
    PosixSerialSource(const std::string& device, int baud = 921600);
    ~PosixSerialSource();
    PosixSerialSource(const PosixSerialSource&) = delete;
    PosixSerialSource& operator=(const PosixSerialSource&) = delete;
    std::size_t read_some(std::span<std::uint8_t> dst, int timeout_ms = 50);
private:
    int fd_{-1};
};
}
