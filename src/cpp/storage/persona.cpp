#include "persona.h"
#include <fstream>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#endif

namespace quila {

Persona::Persona(int dim) : vector(dim, 0.0f), last_update(0) {}

void Persona::update(const std::vector<float>& new_vector) {
    vector = new_vector;
    last_update++;
}

const std::vector<float>& Persona::get() const {
    return vector;
}

void Persona::save(const char* path) {
    // NVRAM (Intel Optane) storage with memory-mapped I/O (Req 9.1.2)
    size_t size = vector.size();
    size_t total_size = sizeof(size) + size * sizeof(float);

#ifdef _WIN32
    HANDLE hFile = CreateFileA(path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS,
                               FILE_FLAG_WRITE_THROUGH | FILE_FLAG_NO_BUFFERING, NULL);
    if (hFile == INVALID_HANDLE_VALUE) return;

    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READWRITE, 0, total_size, NULL);
    if (hMap) {
        void* mapped = MapViewOfFile(hMap, FILE_MAP_WRITE, 0, 0, total_size);
        if (mapped) {
            memcpy(mapped, &size, sizeof(size));
            memcpy((char*)mapped + sizeof(size), vector.data(), size * sizeof(float));
            FlushViewOfFile(mapped, total_size);
            UnmapViewOfFile(mapped);
        }
        CloseHandle(hMap);
    }
    CloseHandle(hFile);
#else
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC | O_DIRECT, 0644);
    if (fd < 0) return;

    ftruncate(fd, total_size);
    void* mapped = mmap(NULL, total_size, PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapped != MAP_FAILED) {
        memcpy(mapped, &size, sizeof(size));
        memcpy((char*)mapped + sizeof(size), vector.data(), size * sizeof(float));
        msync(mapped, total_size, MS_SYNC);
        munmap(mapped, total_size);
    }
    close(fd);
#endif
}

void Persona::load(const char* path) {
    // Load Persona vector from NVRAM storage with memory-mapped I/O (Req 9.1.2)
#ifdef _WIN32
    HANDLE hFile = CreateFileA(path, GENERIC_READ, 0, NULL, OPEN_EXISTING,
                               FILE_FLAG_NO_BUFFERING, NULL);
    if (hFile == INVALID_HANDLE_VALUE) return;

    DWORD fileSize = GetFileSize(hFile, NULL);
    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, fileSize, NULL);
    if (hMap) {
        void* mapped = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, fileSize);
        if (mapped) {
            size_t size;
            memcpy(&size, mapped, sizeof(size));
            vector.resize(size);
            memcpy(vector.data(), (char*)mapped + sizeof(size), size * sizeof(float));
            UnmapViewOfFile(mapped);
        }
        CloseHandle(hMap);
    }
    CloseHandle(hFile);
#else
    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) return;

    struct stat st;
    fstat(fd, &st);
    void* mapped = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mapped != MAP_FAILED) {
        size_t size;
        memcpy(&size, mapped, sizeof(size));
        vector.resize(size);
        memcpy(vector.data(), (char*)mapped + sizeof(size), size * sizeof(float));
        munmap(mapped, st.st_size);
    }
    close(fd);
#endif
}

} // namespace quila
