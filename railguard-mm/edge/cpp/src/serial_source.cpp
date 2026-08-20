#include "railguard/serial_source.hpp"
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <stdexcept>
#include <termios.h>
#include <unistd.h>
namespace railguard {
static speed_t baud_constant(int baud){switch(baud){case 115200:return B115200;case 230400:return B230400;case 460800:return B460800;case 921600:return B921600;default:throw std::invalid_argument("unsupported serial baud");}}
PosixSerialSource::PosixSerialSource(const std::string& device,int baud){fd_=::open(device.c_str(),O_RDWR|O_NOCTTY|O_NONBLOCK);if(fd_<0)throw std::runtime_error("open serial "+device+": "+std::strerror(errno));termios t{};if(tcgetattr(fd_,&t)!=0)throw std::runtime_error("tcgetattr failed");cfmakeraw(&t);const auto b=baud_constant(baud);cfsetispeed(&t,b);cfsetospeed(&t,b);t.c_cflag|=CLOCAL|CREAD;t.c_cflag&=~CSTOPB;t.c_cflag&=~CRTSCTS;if(tcsetattr(fd_,TCSANOW,&t)!=0)throw std::runtime_error("tcsetattr failed");tcflush(fd_,TCIFLUSH);}
PosixSerialSource::~PosixSerialSource(){if(fd_>=0)::close(fd_);}
std::size_t PosixSerialSource::read_some(std::span<std::uint8_t> dst,int timeout_ms){pollfd p{fd_,POLLIN,0};const int r=poll(&p,1,timeout_ms);if(r<0&&errno!=EINTR)throw std::runtime_error("serial poll failed");if(r<=0||!(p.revents&POLLIN))return 0;const auto n=::read(fd_,dst.data(),dst.size());if(n<0&&(errno==EAGAIN||errno==EINTR))return 0;if(n<0)throw std::runtime_error("serial read failed");return static_cast<std::size_t>(n);}
}
